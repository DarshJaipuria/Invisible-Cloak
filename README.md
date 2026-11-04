# Invisible Cloak

Show a **peace sign** to your webcam and you **fade out of the picture**, leaving no trace.
Show it again and you fade back in. Works live in Zoom, Google Meet, Teams, Discord, OBS and
any other app that can use a camera, through a virtual camera.

## How it works

1. **Background capture** - at startup you step out of frame and the script records a few
   seconds of the empty room (median of many frames = a clean background image).
2. **Gesture detection** - MediaPipe Hands tracks your hand. Index + middle finger up with ring
   and pinky folded, held for ~0.3 s, toggles the effect.
3. **Finding you** - the mask is the combination of MediaPipe Selfie Segmentation (AI body mask)
   and "any pixel that differs from the saved background", then filled, grown and softened
   so no edge is left behind.
4. **Brightness matching** - the saved background is matched to the live feed's brightness and
   color every frame (using only the parts of the room you are not covering), so webcam
   auto-exposure changes do not reveal where you were.
5. **Fade** - a smooth eased blend between the live frame and the background.
6. **Output** - frames go to a virtual camera (`pyvirtualcam`), which you select in any app.

## Setup

Requires **Python 3.9 - 3.12** (MediaPipe's `mp.solutions` API is not available on newer versions).

```bash
python -m venv venv
# Windows:  venv\Scripts\activate      macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

Virtual camera driver (needed to use it in other apps):
- **Windows / macOS:** install [OBS Studio](https://obsproject.com/) once (you do not need to keep
  it open, and do not start its virtual camera yourself).
- **Linux:** install `v4l2loopback`.

## Usage

```bash
python invisible_cloak.py                # sends output to the virtual camera
python invisible_cloak.py --no-virtual   # preview window only
python invisible_cloak.py --cam 1        # use a different camera
python invisible_cloak.py --width 960 --height 540   # lighter on slower PCs
```

1. Step completely out of frame during the countdown, wait for the background capture to finish.
2. Walk back in. Make a peace sign to fade out; make it again to fade in.
3. In Zoom / Meet / Teams, choose **OBS Virtual Camera** as your camera.

**Keys (preview window):** `q` quit | `b` recapture background | `space` manual toggle

## Tuning (top of `invisible_cloak.py`)

| Setting | What it does |
|---|---|
| `HOLD_TIME` | seconds you must hold the sign to trigger |
| `FADE_SECONDS` | fade duration |
| `MASK_GROW` | raise it if an outline of you is still visible |
| `DIFF_THRESH` | lower = catches more of you, higher = replaces less of the room |

## Tips for a clean result

- Do not move the camera after capturing the background.
- Use steady, even lighting; press `b` to recapture if the lighting changes.
- Keep your hand fully in frame and well lit for the gesture to be detected.

## Tech

Python, OpenCV, MediaPipe (Hands + Selfie Segmentation), NumPy, pyvirtualcam.