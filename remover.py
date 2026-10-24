"""Core video watermark removal logic (OpenCV + ffmpeg, optional ProPainter)."""
import os
import subprocess
import tempfile

import cv2
import numpy as np


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def first_frame(path, out_jpg):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read the video")
    cv2.imwrite(out_jpg, frame)


def boxes_to_mask(boxes, w, h, grow=4):
    """boxes: list of [x, y, w, h] in video pixels -> uint8 mask (255 = remove)."""
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        mask[y0:y1, x0:x1] = 255
    if grow and mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k)
    return mask


def refine_text_mask(path, box_mask, samples=48, sens=2):
    """Shrink the rough boxes to only the watermark's strokes.

    Stroke score = how much each pixel differs from its local median, measured
    in EVERY sampled frame. We then take a high percentile over time instead of
    the median image, so letters that blink, fade or appear in only some frames
    are still caught, while a hand/object that merely passes through is not.
    Hysteresis keeps faint strokes only when connected to a strong one.
    sens: 1 normal, 2 strong, 3 max (also catches marks visible <45% of the time).
    """
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gs = []
    for i in np.linspace(0, max(0, n - 1), min(samples, max(n, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            gs.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not gs:
        return box_mask
    lo_k, hi_k = {1: (3.0, 6.0), 2: (2.2, 4.5), 3: (1.6, 3.5)}.get(int(sens), (2.2, 4.5))
    pct = {1: 80, 2: 70, 3: 55}.get(int(sens), 70)
    out = np.zeros_like(box_mask)
    cnts, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        k = max(7, (h // 4) | 1)
        scs = []
        for g in gs:
            crop = g[y:y + h, x:x + w]
            bl = cv2.medianBlur(cv2.copyMakeBorder(crop, k, k, k, k, cv2.BORDER_REPLICATE), k)[k:-k, k:-k]
            scs.append(np.abs(crop.astype(np.float32) - bl.astype(np.float32)))
        sc = cv2.GaussianBlur(np.percentile(np.stack(scs), pct, axis=0).astype(np.float32), (0, 0), 0.7)
        base = np.median(sc)
        mad = 1.4826 * np.median(np.abs(sc - base)) + 0.5
        weak = (sc > max(3.0, base + lo_k * mad)).astype(np.uint8)
        strong = sc > max(8.0, base + hi_k * mad)
        _, lab = cv2.connectedComponents(weak, connectivity=8)
        keep = np.unique(lab[strong]); keep = keep[keep > 0]
        m = np.isin(lab, keep).astype(np.uint8) * 255
        d = max(5, (h // 9) | 1)   # grow to cover soft edges and glow
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
        m = cv2.bitwise_and(m, box_mask[y:y + h, x:x + w])
        if m.sum() / 255 < 0.02 * w * h:
            m = box_mask[y:y + h, x:x + w]
        out[y:y + h, x:x + w] |= m
    return out


def auto_detect(path, samples=40):
    """Find fixed (static) overlays: logos, corner marks, timestamps.

    Idea: a static watermark has near-zero change across frames while the
    scene behind it moves, and it has crisp edges. Works best on clips with
    motion. Returns list of [x, y, w, h].
    """
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 5:
        return []
    idxs = np.linspace(0, n - 1, min(samples, n)).astype(int)
    grays = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(grays) < 5:
        return []
    stack = np.stack(grays)
    std = stack.std(axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    edges = cv2.Canny(median, 80, 180)
    h, w = median.shape
    cand = ((std < max(3.0, np.percentile(std, 8))) & (edges > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 60), max(9, h // 60)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    cand = cv2.dilate(cand, k)
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.0004 * w * h or area > 0.25 * w * h:
            continue
        boxes.append([x, y, bw, bh])
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:5]


def _remux(video_only, original, out_path):
    """Encode to H.264 and bring the original audio back.

    Some Windows ffmpeg builds crash (0xC0000005) with certain settings, so we
    try several safer settings in turn and report ffmpeg's real message if all fail.
    """
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only, "-i", original,
            "-map", "0:v:0", "-map", "1:a:0?"]
    tail = ["-shortest", "-movflags", "+faststart", out_path]
    attempts = [
        base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-threads", "2", "-c:a", "aac", "-b:a", "192k"] + tail,
        base + ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p",
                "-threads", "1", "-x264-params", "asm=0", "-c:a", "aac", "-b:a", "192k"] + tail,
        base + ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"] + tail,
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only, "-c:v", "copy", out_path],
    ]
    last = ""
    for cmd in attempts:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return
        last = "exit %s %s" % (r.returncode, (r.stderr or "").strip()[-300:])
    raise RuntimeError("ffmpeg failed on every setting. Last: " + last +
                       "  (try updating ffmpeg: winget upgrade ffmpeg)")


def remove_fast(src, dst, mask, progress=None, method="telea", stabilize=0.25):
    """Per-frame OpenCV inpainting limited to the mask's bounding box.

    stabilize: 0..0.6 blends the repaired patch with the previous one to
    reduce flicker (higher = smoother but can ghost on fast motion).
    """
    info = video_info(src)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    pad = 24
    x0, x1 = max(0, xs.min() - pad), min(info["w"], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(info["h"], ys.max() + pad)
    roi_mask = mask[y0:y1, x0:x1]
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    radius = 5

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cap = cv2.VideoCapture(src)
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"),
                             info["fps"], (info["w"], info["h"]))
    soft = cv2.GaussianBlur(roi_mask, (0, 0), 2).astype(np.float32)[..., None] / 255.0
    prev, i = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            roi = frame[y0:y1, x0:x1]
            fixed = cv2.inpaint(roi, roi_mask, radius, flag).astype(np.float32)
            if stabilize and prev is not None:
                fixed = fixed * (1 - stabilize) + prev * stabilize
            prev = fixed
            blended = fixed * soft + roi.astype(np.float32) * (1 - soft)
            frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
            writer.write(frame)
            i += 1
            if progress and i % 5 == 0:
                progress(i / max(1, info["frames"]))
    finally:
        cap.release()
        writer.release()
    _remux(tmp.name, src, dst)
    os.unlink(tmp.name)
    if progress:
        progress(1.0)


def remove_propainter(src, dst, mask, progress=None):
    """Higher quality: runs ProPainter (https://github.com/sczhou/ProPainter).

    Set env PROPAINTER_DIR to your cloned repo (with weights downloaded).
    Needs a CUDA GPU for reasonable speed. Not run in the demo environment.
    """
    repo = os.environ.get("PROPAINTER_DIR")
    if not repo or not os.path.isdir(repo):
        raise RuntimeError("Set PROPAINTER_DIR to your ProPainter folder")
    work = tempfile.mkdtemp()
    mask_path = os.path.join(work, "mask.png")
    cv2.imwrite(mask_path, mask)
    out_dir = os.path.join(work, "out")
    subprocess.run(["python", "inference_propainter.py", "--video", os.path.abspath(src),
                    "--mask", mask_path, "--output", out_dir, "--fp16"],
                   cwd=repo, check=True)
    name = os.path.splitext(os.path.basename(src))[0]
    result = os.path.join(out_dir, name, "inpaint_out.mp4")
    _remux(result, src, dst)
    if progress:
        progress(1.0)


def remove_temporal(src, dst, mask, progress=None, reach=None):
    """Fill the hole with REAL pixels from neighbouring frames (not a blur).

    When the camera pans/moves, the background hidden behind a fixed logo in
    this frame was visible elsewhere in nearby frames. We align neighbour
    frames to the current one (ORB features + homography), copy the
    uncovered pixels in, and only blur-fill what is still missing.
    Best for moving camera. For a fully static shot there is nothing
    behind the logo in any frame, so it falls back to normal inpainting.
    """
    from collections import OrderedDict

    info = video_info(src)
    W, H = info["w"], info["h"]
    if reach is None:   # big frames = fewer buffered neighbours, so RAM stays low (4K: ~6 frames)
        reach = 16 if W * H <= 1280 * 720 else 10 if W * H <= 1920 * 1080 else 5
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    pad = 30
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    rw, rh = x1 - x0, y1 - y0
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)

    scale = min(1.0, 800.0 / W)
    sw, sh = int(W * scale), int(H * scale)
    S = np.diag([scale, scale, 1.0])
    Si = np.linalg.inv(S)
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    feat_mask = cv2.resize(255 - cv2.dilate(mask, k(21)), (sw, sh), interpolation=cv2.INTER_NEAREST)
    valid_src = 255 - cv2.dilate(mask, k(7))            # trustworthy source pixels
    hole = mask[y0:y1, x0:x1] > 0
    band = (cv2.dilate(mask, k(41)) > 0) & (cv2.dilate(mask, k(9)) == 0)
    band = band[y0:y1, x0:x1]
    soft = cv2.GaussianBlur(mask[y0:y1, x0:x1], (0, 0), 2).astype(np.float32)[..., None] / 255.0

    orb = cv2.ORB_create(2000, fastThreshold=8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def load(frame):
        g = cv2.cvtColor(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(g, feat_mask)
        pts = np.float32([p.pt for p in kp]) if kp else np.zeros((0, 2), np.float32)
        return {"f": frame, "pts": pts, "des": des}

    def align(cur, nb):
        """3x3 matrix mapping neighbour pixel coords -> current pixel coords."""
        if cur["des"] is None or nb["des"] is None or len(cur["pts"]) < 20 or len(nb["pts"]) < 20:
            return None
        good = []
        for pair in bf.knnMatch(nb["des"], cur["des"], k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 20:
            return None
        a = nb["pts"][[m.queryIdx for m in good]]
        b = cur["pts"][[m.trainIdx for m in good]]
        Hs, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        if Hs is None or inl.sum() < 20:
            return None
        return Si @ Hs @ S

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cap = cv2.VideoCapture(src)
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (W, H))
    buf = OrderedDict()
    state = {"next": 0, "eof": False}

    def fill_buffer(upto):
        while not state["eof"] and state["next"] <= upto:
            ok, fr = cap.read()
            if not ok:
                state["eof"] = True
                break
            buf[state["next"]] = load(fr)
            state["next"] += 1

    t = 0
    try:
        while True:
            fill_buffer(t + reach)
            if t not in buf:
                break
            for old in [i for i in buf if i < t - reach]:
                del buf[old]
            cur = buf[t]
            frame = cur["f"]
            roi = frame[y0:y1, x0:x1].copy()
            remaining = hole.copy()
            for d in range(1, reach + 1):
                if not remaining.any():
                    break
                for j in (t - d, t + d):
                    if j not in buf or not remaining.any():
                        continue
                    Hm = align(cur, buf[j])
                    if Hm is None:
                        continue
                    M = T @ Hm
                    warped = cv2.warpPerspective(buf[j]["f"], M, (rw, rh), flags=cv2.INTER_LINEAR)
                    valid = cv2.warpPerspective(valid_src, M, (rw, rh), flags=cv2.INTER_NEAREST) > 250
                    take = remaining & valid
                    if take.sum() < 5:
                        continue
                    ok_band = band & valid
                    off = np.zeros(3, np.float32)
                    if ok_band.sum() > 200:
                        off = (roi[ok_band].mean(0) - warped[ok_band].mean(0)).clip(-25, 25)
                    roi[take] = np.clip(warped[take].astype(np.float32) + off, 0, 255).astype(np.uint8)
                    remaining &= ~take
            if remaining.any():  # whatever no frame could show: classic fill
                m = cv2.dilate(remaining.astype(np.uint8) * 255, k(3))
                roi = cv2.inpaint(roi, m, 5, cv2.INPAINT_TELEA)
            out = roi.astype(np.float32) * soft + frame[y0:y1, x0:x1].astype(np.float32) * (1 - soft)
            frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
            writer.write(frame)
            t += 1
            if progress and t % 3 == 0:
                progress(t / max(1, info["frames"]))
    finally:
        cap.release()
        writer.release()
    _remux(tmp.name, src, dst)
    os.unlink(tmp.name)
    if progress:
        progress(1.0)
