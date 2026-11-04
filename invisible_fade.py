"""
Invisible Fade v3 - show a peace sign (index + middle finger up) and you fade out of the camera
with no trace left. Show it again and you fade back in.

Output goes to a VIRTUAL CAMERA (select "OBS Virtual Camera" in Zoom / Meet / Teams / etc).

Keys (preview window):  q = quit | b = re-capture background | space = manual toggle

How "no trace" works:
  mask = AI body mask  OR  "anything that differs from the saved background"
  + the saved background is brightness/color-matched to the live feed every frame
    (so webcam auto-exposure changes don't reveal where you were).
"""
import argparse
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None

# ----------------------------- settings -----------------------------
HOLD_TIME = 0.3        # seconds to hold the peace sign to trigger
COOLDOWN = 1.0         # minimum seconds between toggles
FADE_SECONDS = 1.2     # fade in / fade out duration
BG_COUNTDOWN = 5       # seconds to step out of frame before capture (lets auto-exposure settle)
BG_CAPTURE = 3         # seconds spent capturing the background
MASK_GROW = 31         # pixels to expand your silhouette (raise if you still see an edge)
DIFF_THRESH = 40       # how different from the background a pixel must be to count as "you"
                       # (lower = more aggressive, raise if too much of the room gets replaced)
# ---------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_seg = mp.solutions.selfie_segmentation


def put_text(img, text, y=60, scale=1.2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    x = max(10, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)


def capture_background(cap, w, h):
    """Median of many frames = clean image of the empty room."""
    start = time.time()
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera returned no frame - is another app using it? Try --cam 1")
            time.sleep(0.5)
            continue
        frame = cv2.resize(frame, (w, h))
        el = time.time() - start
        view = frame.copy()
        if el < BG_COUNTDOWN:
            put_text(view, f"Step OUT of frame... {int(BG_COUNTDOWN - el) + 1}", scale=1.0)
        elif el < BG_COUNTDOWN + BG_CAPTURE:
            put_text(view, "Capturing background - stay out!", scale=1.0)
            i += 1
            if i % 3 == 0:
                frames.append(frame)
        else:
            break
        cv2.imshow("Invisible Fade (preview)", cv2.resize(view, (w // 2, h // 2)))
        cv2.waitKey(1)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def peace_sign(lm):
    """Index + middle extended, ring + pinky folded. Returns (is_peace, flags)."""
    def p(i):
        return np.array([lm[i].x, lm[i].y])

    wrist = p(0)
    ratio = lambda tip, mcp: np.linalg.norm(p(tip) - wrist) / (np.linalg.norm(p(mcp) - wrist) + 1e-6)

    flags = (ratio(8, 5) > 1.6, ratio(12, 9) > 1.6, ratio(16, 13) < 1.5, ratio(20, 17) < 1.5)
    return all(flags), flags


def open_camera(index, w, h, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="real camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-virtual", action="store_true", help="preview only")
    args = ap.parse_args()
    W, H, FPS = args.width, args.height, args.fps
    sw, sh = W // 2, H // 2                                    # masks are computed at half size

    cap = open_camera(args.cam, W, H, FPS)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Close other apps using it or try --cam 1")

    bg = capture_background(cap, W, H)
    bg_f = bg.astype(np.float32)
    bg_small = cv2.GaussianBlur(cv2.resize(bg, (sw, sh)), (5, 5), 0).astype(np.float32)

    vcam = None
    if not args.no_virtual:
        if pyvirtualcam is None:
            print("pyvirtualcam not installed -> preview only.")
        else:
            try:
                vcam = pyvirtualcam.Camera(width=W, height=H, fps=FPS,
                                           fmt=pyvirtualcam.PixelFormat.BGR)
                print(f"Virtual camera running: {vcam.device}  <- select this in your meeting app")
            except Exception as e:
                print(f"Virtual camera failed ({e}). Install OBS (Win/Mac) or v4l2loopback (Linux). Preview only.")

    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    seg = mp_seg.SelfieSegmentation(model_selection=0)

    k = max(3, MASK_GROW // 2)
    grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blur_k = (k | 1, k | 1)

    vanished = False
    progress = 0.0                                             # 0 = visible, 1 = gone
    hold_start, armed = None, True
    last_toggle, last_t = 0.0, time.time()
    mask_prev = np.zeros((sh, sw), np.float32)
    gain = np.ones(3, np.float32)                              # brightness/color match for the background
    flags = None

    def toggle():
        nonlocal vanished, last_toggle
        vanished = not vanished
        last_toggle = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (W, H))
        now = time.time()
        dt, last_t = now - last_t, now

        rgb_small = cv2.cvtColor(cv2.resize(frame, (640, int(640 * H / W))), cv2.COLOR_BGR2RGB)

        # ---- gesture ----
        hres = hands.process(rgb_small)
        gesture, flags = False, None
        for h in hres.multi_hand_landmarks or []:
            is_peace, fl = peace_sign(h.landmark)
            flags = fl
            if is_peace:
                gesture = True
        if gesture and armed and now - last_toggle > COOLDOWN:
            hold_start = hold_start or now
            if now - hold_start >= HOLD_TIME:
                toggle()
                armed, hold_start = False, None
        elif not gesture:
            armed, hold_start = True, None

        # ---- fade progress ----
        target = 1.0 if vanished else 0.0
        step = dt / FADE_SECONDS
        progress += float(np.clip(target - progress, -step, step))

        # ---- render ----
        if progress <= 0.001:
            out = frame
            mask_prev *= 0.5
        else:
            small = cv2.GaussianBlur(cv2.resize(frame, (sw, sh)), (5, 5), 0)

            # 1) AI body mask (generous threshold)
            sres = seg.process(rgb_small)
            ml = cv2.resize(sres.segmentation_mask, (sw, sh))
            ml_bin = (ml > 0.2).astype(np.uint8) * 255

            # 2) match background brightness/color to the live feed, using pixels that are NOT you
            outside = cv2.bitwise_not(cv2.dilate(ml_bin, grow_k, iterations=2))
            if cv2.countNonZero(outside) > 1500:
                f_mean = np.array(cv2.mean(small, mask=outside)[:3], np.float32)
                b_mean = np.array(cv2.mean(bg_small.astype(np.uint8), mask=outside)[:3], np.float32)
                ratio = np.clip(f_mean / (b_mean + 1e-6), 0.6, 1.6)
                gain = 0.7 * gain + 0.3 * ratio
            bg_adj_small = np.clip(bg_small * gain, 0, 255)

            # 3) anything that differs from the (matched) background also counts as "you"
            diff = np.abs(small.astype(np.float32) - bg_adj_small).max(axis=2)
            diff_bin = (diff > DIFF_THRESH).astype(np.uint8) * 255
            diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, open_k)

            # 4) combine, fill holes, grow, soften
            m = cv2.bitwise_or(ml_bin, diff_bin)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
            m = cv2.dilate(m, grow_k)
            m = cv2.GaussianBlur(m, blur_k, 0).astype(np.float32) / 255
            m = np.clip(m * 1.8, 0, 1)                         # harden the soft edge so nothing half-shows
            m = np.maximum(m, mask_prev * 0.6)                 # hold through detection flicker
            mask_prev = m
            mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

            a = progress * progress * (3 - 2 * progress)       # smooth ease in/out
            alpha = (mask * a)[..., None]
            bg_adj = np.clip(bg_f * gain, 0, 255)
            out = (frame.astype(np.float32) * (1 - alpha) + bg_adj * alpha).astype(np.uint8)

        # ---- output ----
        if vcam:
            vcam.send(out)

        view = cv2.resize(out, (sw, sh))                       # preview-only overlays below
        if hold_start:
            cv2.rectangle(view, (10, 10), (10 + int(200 * (now - hold_start) / HOLD_TIME), 24),
                          (255, 255, 0), -1)
        status = "INVISIBLE" if vanished and progress > 0.99 else ("fading..." if 0.001 < progress < 0.999 else "visible")
        cv2.putText(view, status, (10, sh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        dbg = "no hand" if flags is None else "I{} M{} R{} P{}".format(*["+" if f else "-" for f in flags])
        cv2.putText(view, dbg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Invisible Fade (preview)", view)
        if vcam:
            vcam.sleep_until_next_frame()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            bg = capture_background(cap, W, H)
            bg_f = bg.astype(np.float32)
            bg_small = cv2.GaussianBlur(cv2.resize(bg, (sw, sh)), (5, 5), 0).astype(np.float32)
            gain = np.ones(3, np.float32)
        elif key == ord(" "):
            toggle()

    cap.release()
    if vcam:
        vcam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()