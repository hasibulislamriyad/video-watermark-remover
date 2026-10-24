"""Run: python app.py  ->  http://127.0.0.1:5000"""
import os
import shutil
import subprocess
import threading
import uuid

from flask import Flask, jsonify, request, send_file

import cv2
import remover

# Keep the PC cool: fewer CPU threads + lower priority so Windows stays responsive.
cv2.setNumThreads(2)
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
DATA = os.path.join(BASE, "jobs")
os.makedirs(DATA, exist_ok=True)
JOBS = {}


def jp(job, name):
    return os.path.join(DATA, job, name)


@app.get("/")
def index():
    for p in (os.path.join(BASE, "templates", "index.html"), os.path.join(BASE, "index.html")):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    return "index.html not found. Put it next to app.py or inside a 'templates' folder. Looked in: " + BASE, 500


def make_preview(job):
    """Browser-friendly copy (H.264 + AAC, max 960px wide) so the video always plays."""
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", jp(job, "input.mp4"),
                        "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                        "-movflags", "+faststart", jp(job, "preview.mp4")], check=True)
        return True
    except Exception:
        return False


@app.post("/upload")
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify(error="No file"), 400
    job = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(DATA, job))
    f.save(jp(job, "input.mp4"))
    try:
        info = remover.video_info(jp(job, "input.mp4"))
        if info["w"] == 0:
            raise ValueError("Could not read this video")
    except Exception as e:
        return jsonify(error=str(e)), 400
    if not make_preview(job):
        return jsonify(error="ffmpeg not found or failed. Install ffmpeg and add it to PATH."), 400
    JOBS[job] = {"status": "ready", "progress": 0}
    return jsonify(job=job, **info)


@app.get("/video/<job>/<kind>")
def video(job, kind):
    name = {"original": "preview.mp4", "result": "output_preview.mp4"}.get(kind)
    if not name or not os.path.isfile(jp(job, name)):
        return "not found", 404
    return send_file(jp(job, name), mimetype="video/mp4", conditional=True)


@app.post("/detect/<job>")
def detect(job):
    return jsonify(boxes=remover.auto_detect(jp(job, "input.mp4")))


@app.post("/process/<job>")
def process(job):
    body = request.get_json(force=True)
    info = remover.video_info(jp(job, "input.mp4"))
    mask = remover.boxes_to_mask(body.get("boxes", []), info["w"], info["h"])
    if not mask.any():
        return jsonify(error="Draw at least one box over the watermark"), 400
    if body.get("precise", True):   # repair only the text strokes, not the whole box
        mask = remover.refine_text_mask(jp(job, "input.mp4"), mask, sens=body.get("sens", 2))
    mode = body.get("mode", "smart")

    def run():
        JOBS[job] = {"status": "working", "progress": 0}
        cb = lambda p: JOBS[job].update(progress=round(p, 3))
        src, dst = jp(job, "input.mp4"), jp(job, "output.mp4")
        try:
            if mode == "propainter":
                remover.remove_propainter(src, dst, mask, cb)
            elif mode == "fast":
                remover.remove_fast(src, dst, mask, cb)
            else:
                remover.remove_temporal(src, dst, mask, cb)
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", dst,
                                "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "ultrafast",
                                "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                                "-movflags", "+faststart", jp(job, "output_preview.mp4")])
            if r.returncode != 0:  # preview is optional: fall back to the full file
                shutil.copyfile(dst, jp(job, "output_preview.mp4"))
            JOBS[job].update(status="done", progress=1)
        except Exception as e:
            JOBS[job] = {"status": "error", "error": str(e), "progress": 0}

    threading.Thread(target=run, daemon=True).start()
    return jsonify(ok=True)


@app.get("/status/<job>")
def status(job):
    return jsonify(JOBS.get(job, {"status": "unknown"}))


@app.get("/download/<job>")
def download(job):
    return send_file(jp(job, "output.mp4"), as_attachment=True, download_name="clean.mp4")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
