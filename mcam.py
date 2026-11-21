#!/usr/bin/env python3
"""mcam — 複数の動画・音声を目と耳で揃え、全部が重なる範囲だけを無劣化で切り出す。

使い方:
    python3 mcam.py cam1.mp4 cam2.mov rec.wav ...     # ブラウザで位置合わせ UI を開く
    python3 mcam.py -p show.mcam.json                  # 保存したプロジェクトを開き直す
    python3 mcam.py export show.mcam.json [-o OUTDIR]  # UI なしで書き出しだけ行う

出力:
    動画ファイル → -map 0:V -c:v copy -an
    音声ファイル → -map 0:a -c:a copy

依存: ffmpeg / ffprobe（PATH 上）、Python 3.8+ 標準ライブラリのみ。
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import mimetypes
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI_FILE = HERE / "mcam_ui.html"
PEAKS_PER_SEC = 1000          # 波形の解像度（1 ms）
PEAK_DECODE_SR = 16000
PROXY_HEIGHT = 360

FFMPEG = os.environ.get("MCAM_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("MCAM_FFPROBE", "ffprobe")


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _f(v, default=None):
    try:
        x = float(v)
        return x if x == x else default  # NaN 除外
    except (TypeError, ValueError):
        return default


def _rate(s):
    if not s or s == "0/0":
        return None
    if "/" in s:
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else None
    return float(s)


def probe(path: str) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    fmt = info.get("format", {})
    fstart = _f(fmt.get("start_time"), 0.0)
    fdur = _f(fmt.get("duration"), 0.0)

    def span(st):
        s = _f(st.get("start_time"), fstart) - fstart
        d = _f(st.get("duration"))
        if d is None:
            d = max(0.0, fdur - s)
        return s, d

    video = audio = None
    for st in info.get("streams", []):
        kind = st.get("codec_type")
        if kind == "video" and video is None:
            if st.get("disposition", {}).get("attached_pic"):
                continue  # カバーアートは無視
            s, d = span(st)
            fps = _rate(st.get("avg_frame_rate")) or _rate(st.get("r_frame_rate"))
            video = {"codec": st.get("codec_name"), "start": s, "dur": d,
                     "fps": fps, "w": st.get("width"), "h": st.get("height")}
        elif kind == "audio" and audio is None:
            s, d = span(st)
            audio = {"codec": st.get("codec_name"), "start": s, "dur": d,
                     "sr": int(st.get("sample_rate") or 48000),
                     "sample_fmt": st.get("sample_fmt"),
                     "ch": int(st.get("channels") or 2)}
    if not video and not audio:
        raise SystemExit(f"{path}: 動画・音声ストリームが見つかりません")
    kind = "video" if video else "audio"
    main = video or audio
    return {
        "path": str(Path(path).resolve()),
        "name": Path(path).name,
        "kind": kind,
        "format": fmt.get("format_name"),
        "fstart": fstart,
        "duration": fdur,
        "video": video,
        "audio": audio,
        # 書き出し対象ストリームがファイル時間軸上で占める範囲
        "span": [main["start"], main["start"] + main["dur"]],
    }


# ---------------------------------------------------------------------------
# キャッシュ（波形・プロキシ）
# ---------------------------------------------------------------------------

def cache_key(path: str) -> str:
    st = os.stat(path)
    h = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    return h[:16]


def make_peaks(path: str, dst: Path) -> None:
    """mono 16 kHz の u8 PCM から 1 ms ごとの (min, max) を作る。値は 0..255。"""
    tmp = dst.with_suffix(".part")
    block = PEAK_DECODE_SR // PEAKS_PER_SEC
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1",
         "-ar", str(PEAK_DECODE_SR), "-c:a", "pcm_u8", "-f", "u8", "-"],
        stdout=subprocess.PIPE)
    out = bytearray()
    rest = b""
    chunk = block * 4096
    while True:
        buf = proc.stdout.read(chunk)
        if not buf:
            break
        buf = rest + buf
        n = len(buf) // block * block
        mv = memoryview(buf)
        for i in range(0, n, block):
            b = mv[i:i + block]
            out.append(min(b))
            out.append(max(b))
        rest = bytes(buf[n:])
    if rest:
        out.append(min(rest))
        out.append(max(rest))
    proc.wait()
    tmp.write_bytes(bytes(out))
    tmp.replace(dst)


def make_proxy(path: str, dst: Path) -> None:
    tmp = dst.with_name(dst.stem + ".part.mp4")
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-i", path, "-map", "0:V:0", "-an", "-sn", "-dn",
         "-vf", f"scale=-2:'min({PROXY_HEIGHT},ih)'",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", "12",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        check=True)
    tmp.replace(dst)


# ---------------------------------------------------------------------------
# プロジェクト
# ---------------------------------------------------------------------------

class Project:
    def __init__(self, path: Path, tracks: list[dict], out_dir: Path):
        self.path = path
        self.tracks = tracks
        self.out_dir = out_dir
        self.lock = threading.Lock()

    @classmethod
    def create(cls, files, project_path, out_dir):
        tracks = []
        for f in files:
            t = probe(f)
            t["offset"] = 0.0
            t["marker"] = None
            t["gain"] = 1.0
            t["mute"] = False
            tracks.append(t)
        return cls(project_path, tracks, out_dir)

    @classmethod
    def load(cls, project_path: Path, out_dir=None):
        data = json.loads(project_path.read_text(encoding="utf-8"))
        base = project_path.resolve().parent
        tracks = []
        for saved in data["tracks"]:
            p = Path(saved["path"])
            if not p.is_absolute():
                p = base / p
            t = probe(str(p))
            for k in ("offset", "marker", "gain", "mute"):
                if k in saved:
                    t[k] = saved[k]
            t.setdefault("offset", 0.0)
            t.setdefault("marker", None)
            t.setdefault("gain", 1.0)
            t.setdefault("mute", False)
            tracks.append(t)
        od = Path(out_dir) if out_dir else Path(data.get("out_dir") or base / "synced")
        if not od.is_absolute():
            od = base / od
        return cls(project_path, tracks, od)

    def save(self):
        data = {
            "version": 1,
            "out_dir": str(self.out_dir),
            "tracks": [{k: t[k] for k in ("path", "offset", "marker", "gain", "mute")}
                       for t in self.tracks],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def overlap(self):
        """タイムライン上で全トラックが重なる区間 [t0, t1]。重ならなければ None。"""
        t0 = max(t["offset"] + t["span"][0] for t in self.tracks)
        t1 = min(t["offset"] + t["span"][1] for t in self.tracks)
        return (t0, t1) if t1 > t0 else None


# ---------------------------------------------------------------------------
# 書き出しと検証
# ---------------------------------------------------------------------------

LOSSLESS = {"flac", "alac", "wavpack", "tta"}


def export_plan(proj: Project, video_ext: str | None = None, lossless_reencode=False):
    ov = proj.overlap()
    if ov is None:
        raise ValueError("全トラックが重なる区間がありません")
    t0, t1 = ov
    length = t1 - t0
    proj.out_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    used = set()
    for t in proj.tracks:
        src = Path(t["path"])
        ext = src.suffix
        if t["kind"] == "video" and video_ext:
            ext = video_ext if video_ext.startswith(".") else "." + video_ext
        stem = src.stem + ".sync"
        out = proj.out_dir / (stem + ext)
        n = 2
        while out in used:
            out = proj.out_dir / f"{stem}{n}{ext}"
            n += 1
        used.add(out)
        start = t0 - t["offset"]          # ファイル時間（format start 基準）
        cmd = [FFMPEG, "-hide_banner", "-v", "error", "-y"]
        if start > 1e-6:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", str(src), "-t", f"{length:.6f}", "-map_metadata", "0"]
        note = None
        if t["kind"] == "video":
            cmd += ["-map", "0:V:0", "-c:v", "copy", "-an", "-sn", "-dn"]
        else:
            codec = t["audio"]["codec"] or ""
            if lossless_reencode and codec in LOSSLESS:
                # FLAC/ALAC の copy はフレーム境界でしか切れない。可逆なので同コーデックで
                # 作り直しても音は変わらず、サンプル単位で切れる（ファイル自体は別物になる）。
                cmd += ["-map", "0:a", "-c:a", codec]
                fmt = t["audio"].get("sample_fmt")
                if fmt:
                    cmd += ["-sample_fmt", fmt]
                note = f"可逆形式のため -c:a {codec} で再エンコード（--lossless-reencode）"
            else:
                cmd += ["-map", "0:a", "-c:a", "copy"]
            cmd += ["-vn", "-sn", "-dn"]
        cmd.append(str(out))
        plan.append({"track": t, "start": start, "length": length,
                     "out": out, "cmd": cmd, "note": note})
    return (t0, t1), plan


def _framemd5(args):
    # -enc_time_base -1: 出力時刻を 1/fps に丸めず、元のタイムベースのまま出す
    out = subprocess.run([FFMPEG, "-v", "error", *args, "-enc_time_base", "-1",
                          "-fps_mode", "passthrough", "-f", "framemd5", "-"],
                         capture_output=True, text=True).stdout
    tb = None
    frames = []
    for line in out.splitlines():
        m = re.match(r"#tb 0: (\d+)/(\d+)", line)
        if m:
            tb = int(m.group(1)) / int(m.group(2))
            continue
        if line.startswith("#") or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6 and tb:
            frames.append((int(parts[2]) * tb, parts[5]))
    return frames


def measure_video(src: str, start: float, out: str, fps: float | None):
    """出力の先頭フレームが元ファイルのどの時刻のフレームかを、デコード結果の一致で調べる。"""
    first = _framemd5(["-i", out, "-map", "0:v:0", "-frames:v", "1"])
    if not first:
        return None
    target = first[0][1]
    a = max(0.0, start - 3.0)
    frames = _framemd5(["-ss", f"{a:.6f}", "-i", src, "-map", "0:V:0", "-t", "6"])
    hits = [a + pts for pts, h in frames if h == target]
    if not hits:
        return None
    best = min(hits, key=lambda x: abs(x - start))
    durs = [b[0] - a_[0] for a_, b in zip(frames, frames[1:]) if b[0] > a_[0]]
    frame = (sorted(durs)[len(durs) // 2] if durs else (1 / fps if fps else 1 / 30))
    # ffmpeg / エディットリストの解釈では「start 以降で最初のフレーム」が先頭になるので、
    # 0〜+1 フレームの差はフレームの刻みによるもの。負の大きな差はキーフレームまで戻った印。
    return {"actual": best, "error": best - start, "frame": frame,
            "ambiguous": len(hits) > 1}


def _pcm_mono(args, sr):
    import array
    raw = subprocess.run(
        [FFMPEG, "-v", "error", *args, "-map", "0:a:0", "-ar", str(sr), "-ac", "1",
         "-c:a", "pcm_f32le", "-f", "f32le", "-"], capture_output=True).stdout
    a = array.array("f")
    a.frombytes(raw[:len(raw) // 4 * 4])
    if sys.byteorder != "little":
        a.byteswap()
    return a


def _sad(x, y, xo, yo, n, step=1):
    return sum(abs(x[xo + k] - y[yo + k]) for k in range(0, n, step))


def measure_audio(src: str, start: float, out: str, sr: int, ch: int):
    """出力の 0.25 秒目付近 2048 サンプルが、元ファイルのどこと一致するかを探す。
    非可逆コーデックでもデコード結果はほぼ同じになるので、差の絶対値和が最小の位置を取る。"""
    probe_at, n, search = 0.25, 2048, 0.2
    o = _pcm_mono(["-i", out, "-t", "0.6"], sr)
    n0 = int(probe_at * sr)
    if len(o) < n0 + n:
        return None
    energy = sum(abs(o[n0 + k]) for k in range(n))
    if energy < 1e-4 * n:
        return None  # 無音で照合不能
    a = max(0.0, start + probe_at - search)
    s = _pcm_mono(["-ss", f"{a:.6f}", "-i", src, "-t", f"{2 * search + n / sr + 0.05:.6f}"], sr)
    # ① 16 サンプル平均で粗く探す
    D = 16

    def pool(x, off, cnt):
        return [sum(x[off + i * D: off + i * D + D]) for i in range(cnt)]
    m = n // D
    op = pool(o, n0, m)
    span = len(s) - n
    if span <= 0:
        return None
    coarse = []
    for lag in range(0, span, D // 2):
        sp = [sum(s[lag + i * D: lag + i * D + D]) for i in range(m)]
        coarse.append((sum(abs(p - q) for p, q in zip(op, sp)), lag))
    coarse.sort()
    # ② 上位候補の周辺をサンプル単位で
    best = None
    for _, c in coarse[:3]:
        for lag in range(max(0, c - D), min(span, c + D) + 1):
            v = _sad(s, o, lag, n0, n, 4)
            if best is None or v < best[0]:
                best = (v, lag)
    v, lag = best
    v = _sad(s, o, lag, n0, n)
    rel = v / max(energy, 1e-9)
    actual = a + lag / sr - probe_at
    return {"actual": actual, "error": actual - start, "residual": rel,
            "ambiguous": rel > 0.3}


def run_export(proj: Project, video_ext=None, lossless_reencode=False, verify=True, log=print):
    (t0, t1), plan = export_plan(proj, video_ext, lossless_reencode)
    log(f"重なり区間: タイムライン {t0:.6f} 〜 {t1:.6f} 秒（長さ {t1 - t0:.3f} 秒）")
    results = []
    for p in plan:
        t = p["track"]
        log(f"→ {t['name']}  開始 {p['start']:.6f} 秒  → {p['out']}")
        if p["note"]:
            log(f"   ({p['note']})")
        r = subprocess.run(p["cmd"], capture_output=True, text=True)
        res = {"name": t["name"], "out": str(p["out"]), "start": p["start"],
               "length": p["length"], "cmd": p["cmd"], "ok": r.returncode == 0,
               "stderr": r.stderr.strip(), "measured": None}
        if not res["ok"]:
            log("   失敗: " + r.stderr.strip())
        elif verify:
            try:
                if t["kind"] == "video":
                    m = measure_video(t["path"], p["start"], str(p["out"]), t["video"]["fps"])
                else:
                    m = measure_audio(t["path"], p["start"], str(p["out"]),
                                      t["audio"]["sr"], t["audio"]["ch"])
            except Exception as e:  # 検証失敗は書き出し失敗ではない
                m = {"error_msg": str(e)}
            res["measured"] = m
            log("   " + describe_measure(t, m))
        results.append(res)
    report = {
        "overlap": [t0, t1],
        "tracks": [{"name": t["name"], "path": t["path"], "offset": t["offset"]}
                   for t in proj.tracks],
        "results": results,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rp = proj.out_dir / "sync_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")
    log(f"レポート: {rp}")
    return report


def describe_measure(t, m):
    if not m:
        return "実測: 照合できず（無音・静止画などで一致箇所を特定できない）"
    if "error_msg" in m:
        return "実測: 検証中にエラー " + m["error_msg"]
    ms = m["error"] * 1000
    if t["kind"] == "video":
        fr = m["frame"]
        fe = m["error"] / fr
        if -0.02 < fe < 1.0:
            return (f"実測: 先頭フレーム = 元の {m['actual']:.4f} 秒（理想 {ms:+.1f} ms、"
                    "1 フレーム未満の刻み）")
        hint = ("キーフレームまで戻って切れています。--video-ext mov / mp4 で"
                "エディットリスト付きにすると正確になります") if fe < 0 else "想定外のずれです"
        return f"実測: 先頭が理想より {ms:+.1f} ms（{fe:+.2f} フレーム）← {hint}"
    flag = "（一致度が低いため参考値）" if m.get("ambiguous") else ""
    sr = t["audio"]["sr"]
    smp = m["error"] * sr
    if abs(smp) < 0.5:
        return f"実測: サンプル単位で一致{flag}"
    hint = ""
    if t["audio"]["codec"] in LOSSLESS:
        hint = "  ← フレーム境界で切れています。--lossless-reencode で正確に切れます"
    return f"実測: 先頭が理想より {ms:+.3f} ms（{smp:+.0f} サンプル）ずれています{flag}{hint}"


# ---------------------------------------------------------------------------
# HTTP サーバ
# ---------------------------------------------------------------------------

class State:
    def __init__(self, proj: Project, cache_dir: Path, use_proxy: bool, opts):
        self.proj = proj
        self.cache = cache_dir
        self.use_proxy = use_proxy
        self.opts = opts
        self.jobs: dict[str, str] = {}      # key -> "pending" | "ready" | "error: ..."
        self.export_log: list[str] = []
        self.exporting = False
        self.cache.mkdir(parents=True, exist_ok=True)
        self.keys = [cache_key(t["path"]) for t in proj.tracks]
        for i, t in enumerate(proj.tracks):
            if t["audio"]:
                self.jobs[f"peaks{i}"] = "pending"
            if t["video"] and use_proxy:
                self.jobs[f"proxy{i}"] = "pending"
        threading.Thread(target=self._prepare, daemon=True).start()

    def peaks_path(self, i):
        return self.cache / f"{self.keys[i]}.peaks"

    def proxy_path(self, i):
        return self.cache / f"{self.keys[i]}.proxy.mp4"

    def _prepare(self):
        # 波形を先に全部作る（軽い）→ 次にプロキシ（重い）
        for i, t in enumerate(self.proj.tracks):
            if t["audio"]:
                self._job(f"peaks{i}", self.peaks_path(i), make_peaks, t["path"])
        for i, t in enumerate(self.proj.tracks):
            if t["video"] and self.use_proxy:
                self._job(f"proxy{i}", self.proxy_path(i), make_proxy, t["path"])

    def _job(self, name, dst, fn, src):
        if dst.exists():
            self.jobs[name] = "ready"
            return
        self.jobs[name] = "pending"
        try:
            fn(src, dst)
            self.jobs[name] = "ready"
        except Exception as e:
            self.jobs[name] = f"error: {e}"

    def snapshot(self):
        p = self.proj
        tracks = []
        for i, t in enumerate(p.tracks):
            d = {k: t[k] for k in ("name", "kind", "offset", "marker", "gain", "mute",
                                   "span", "duration", "video", "audio")}
            d["peaks"] = self.jobs.get(f"peaks{i}") if t["audio"] else None
            if t["video"]:
                d["proxy"] = self.jobs.get(f"proxy{i}") if self.use_proxy else "ready"
            tracks.append(d)
        ov = p.overlap()
        return {"tracks": tracks, "overlap": ov, "out_dir": str(p.out_dir),
                "project": str(p.path), "peaks_per_sec": PEAKS_PER_SEC,
                "exporting": self.exporting, "export_log": self.export_log}


def make_handler(state: State):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def _file(self, path: Path, ctype):
            size = path.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            code = 200
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        if m.group(2):
                            end = min(int(m.group(2)), size - 1)
                    elif m.group(2):
                        start = max(0, size - int(m.group(2)))
                    code = 206
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        buf = f.read(min(left, 1 << 20))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        left -= len(buf)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            tracks = state.proj.tracks
            try:
                if u.path == "/":
                    return self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/state":
                    return self._json(state.snapshot())
                i = int(q.get("i", -1))
                if not 0 <= i < len(tracks):
                    return self._send(404, b"not found", "text/plain")
                t = tracks[i]
                if u.path == "/peaks":
                    p = state.peaks_path(i)
                    if not p.exists():
                        return self._send(404, b"pending", "text/plain")
                    return self._send(200, p.read_bytes(), "application/octet-stream")
                if u.path == "/video":
                    if state.use_proxy:
                        p = state.proxy_path(i)
                        if not p.exists():
                            return self._send(404, b"pending", "text/plain")
                        return self._file(p, "video/mp4")
                    p = Path(t["path"])
                    return self._file(p, mimetypes.guess_type(p.name)[0] or "video/mp4")
                if u.path == "/pcm":
                    return self._pcm(t, q)
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _pcm(self, t, q):
            """ファイル時間 s 秒から n サンプル分の f32le インターリーブ PCM。"""
            sr = int(q.get("sr", 48000))
            ch = 2
            s = float(q["s"])
            n = int(q["n"])
            cmd = [FFMPEG, "-v", "error"]
            if s > 0:
                cmd += ["-ss", f"{s:.6f}"]
            # first_pts=0 + async で、音声ストリームの開始が遅いファイルでも先頭を無音で埋める
            cmd += ["-i", t["path"], "-map", "0:a:0",
                    "-af", f"aresample={sr}:async=1:first_pts=0",
                    "-ac", str(ch), "-t", f"{n / sr + 0.05:.6f}",
                    "-c:a", "pcm_f32le", "-f", "f32le", "-"]
            data = subprocess.run(cmd, capture_output=True).stdout
            want = n * ch * 4
            data = data[:want]
            if len(data) < want:
                data += bytes(want - len(data))
            self._send(200, data, "application/octet-stream")

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            ln = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(ln) or b"{}")
            proj = state.proj
            if u.path == "/api/update":
                with proj.lock:
                    for i, patch in body.get("tracks", {}).items():
                        t = proj.tracks[int(i)]
                        for k in ("offset", "marker", "gain", "mute"):
                            if k in patch:
                                t[k] = patch[k]
                    proj.save()
                return self._json(state.snapshot())
            if u.path == "/api/export":
                if state.exporting:
                    return self._json({"error": "書き出し中です"}, 409)
                if proj.overlap() is None:
                    return self._json({"error": "全トラックが重なる区間がありません"}, 400)
                state.exporting = True
                state.export_log = []

                def work():
                    try:
                        run_export(proj, state.opts.video_ext, state.opts.lossless_reencode,
                                   not state.opts.no_verify, log=state.export_log.append)
                        state.export_log.append("完了")
                    except Exception as e:
                        state.export_log.append(f"エラー: {e}")
                    finally:
                        state.exporting = False
                threading.Thread(target=work, daemon=True).start()
                return self._json({"ok": True})
            self._send(404, b"not found", "text/plain")

    return H


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def check_tools():
    for tool in (FFMPEG, FFPROBE):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} が見つかりません。PATH に ffmpeg/ffprobe を置いてください。")


def add_export_opts(ap):
    ap.add_argument("-o", "--out", help="出力フォルダ（既定: プロジェクトと同じ場所の synced/）")
    ap.add_argument("--video-ext", help="動画の出力コンテナ拡張子を変更（例: mov）")
    ap.add_argument("--lossless-reencode", action="store_true",
                    help="FLAC/ALAC など可逆形式の音声だけ同じコーデックで再エンコードし、"
                         "サンプル単位で切る（copy だとフレーム境界でしか切れない）")
    ap.add_argument("--no-verify", action="store_true", help="書き出し後の実測検証を省略")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    check_tools()
    if argv and argv[0] == "export":
        ap = argparse.ArgumentParser(prog="mcam.py export")
        ap.add_argument("project")
        add_export_opts(ap)
        a = ap.parse_args(argv[1:])
        proj = Project.load(Path(a.project), a.out)
        report = run_export(proj, a.video_ext, a.lossless_reencode, not a.no_verify)
        return 0 if all(r["ok"] for r in report["results"]) else 1

    ap = argparse.ArgumentParser(prog="mcam.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="動画・音声ファイル")
    ap.add_argument("-p", "--project", help="プロジェクト JSON（既定: 先頭ファイルの隣に作成）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-proxy", action="store_true",
                    help="プレビュー用プロキシを作らず元動画を直接ブラウザで再生する")
    ap.add_argument("--no-browser", action="store_true")
    add_export_opts(ap)
    a = ap.parse_args(argv)

    if a.files:
        first = Path(a.files[0]).resolve()
        pp = Path(a.project) if a.project else first.parent / (first.stem + ".mcam.json")
        if pp.exists() and not a.project:
            print(f"既存のプロジェクト {pp} があります。続きから開くには -p {pp} を、"
                  "作り直すなら削除してください。", file=sys.stderr)
            return 2
        out = Path(a.out) if a.out else pp.resolve().parent / "synced"
        proj = Project.create(a.files, pp.resolve(), out.resolve())
        proj.save()
    elif a.project:
        proj = Project.load(Path(a.project).resolve(), a.out)
    else:
        ap.print_help()
        return 2

    cache = proj.path.parent / ".mcam-cache"
    state = State(proj, cache, not a.no_proxy, a)
    srv = Server((a.host, a.port), make_handler(state))
    url = f"http://{a.host}:{a.port}/"
    print(f"プロジェクト: {proj.path}")
    print(f"UI: {url}   （Ctrl+C で終了。位置は自動保存されます）")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
