**video-watermark-remover**

**A free, local video watermark remover for text and logo watermarks.** Upload a video, box the watermark, and get a repaired clip. It only fixes the watermark's own pixels, so the rest of the frame stays sharp. Audio is untouched. Everything runs on your computer, and nothing is uploaded anywhere.

> Only use this on videos you own or have permission to edit.

## Features

- **Text-only (precise) mode:** detects the actual letter strokes inside your box and repairs only those pixels, instead of smudging the whole rectangle.
- **Catches blinking or fading text:** finds strokes that appear in only some frames.
- **Three modes:**
  - **Smart:** pulls real background pixels from neighbouring frames (best when the camera moves).
  - **Fast:** OpenCV inpainting, CPU only, works on any video.
  - **High:** [ProPainter](https://github.com/sczhou/ProPainter), best quality, needs a GPU.
- **Sensitivity setting** (Normal / Strong / Max) for faint or thin watermarks.
- **Before/after player** in the browser, plus "Detect fixed marks" auto-detection.
- Keeps the CPU/RAM load low (limited threads, buffered frames scale with resolution).



## Install

Requires **Python 3.9+** and **ffmpeg** on your PATH.

```bash
git clone https://github.com/hasibulislamriyad/video-watermark-remover
cd video-watermark-remover
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>.

Install ffmpeg: Windows `winget install ffmpeg`, macOS `brew install ffmpeg`, Ubuntu `sudo apt install ffmpeg`.

## How to use

1. Upload an MP4 / M4V.
2. Scrub to a frame where the watermark is visible.
3. Drag a box **slightly larger than the whole watermark** (all letters and decorations). Text-only mode keeps the rest of the box untouched.
4. Pick **Smart** or **Fast**, keep **Text-only** on, and press **Remove watermark**.
5. If a faint trace is left, set Sensitivity to **Max** and run again. Compare with the tabs, then download.

## Limitations

- On a completely static shot, nothing shows what is behind the watermark in any frame, so the fill is an estimate. Thin text on simple backgrounds looks great; big logos on busy backgrounds may show artifacts.
- Large watermarks, or ones covering faces or fine detail, will not be perfect.
- The demos above are synthetic, so results on real videos vary.
- High mode requires cloning ProPainter and a CUDA GPU:
  `export PROPAINTER_DIR=/path/to/ProPainter` before `python app.py`.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Flask server and job handling |
| `remover.py` | Mask detection and removal algorithms (OpenCV + ffmpeg) |
| `index.html` | Browser UI |

## License

MIT. See [LICENSE](LICENSE).
