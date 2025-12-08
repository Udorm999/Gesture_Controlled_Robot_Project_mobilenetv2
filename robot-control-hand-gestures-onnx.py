#!/usr/bin/env python3
import time, threading, socket
import cv2, numpy as np
from flask import Flask, Response
from auppbot import AUPPBot
import RPi.GPIO as GPIO

import onnxruntime as ort


# ---------------- Configuration (tweak these) ----------------
CAM_INDEX = 0
W, H = 640, 480
ROTATE_90_CW = True

# Drive params
PORT = "/dev/ttyUSB0"
BAUD = 115200
BASE = 15  # base forward PWM (tweak)
DELTA = 8  # steering delta (tweak)
SEARCH_SPIN = 15
SEARCH_DURATION = 0.5

LEFT_SIGN = +1
RIGHT_SIGN = +1

# Ultrasonic
TRIG = 21
ECHO = 20
STOP_THRESHOLD_CM = 1.0
ULTRA_MIN_INTERVAL = 0.05
_ultra_last_time = 0
# require N consecutive low readings before acting
ULTRA_CONSECUTIVE = 2
_ultra_buffer = []

# MobileNetV2 gesture classifier config (ONNX)
ONNX_MODEL_PATH = "/home/aupp/Documents/hand_gestures_mobilenetv2.onnx"  # TODO: update path
# Must be in the SAME order as training
CLASS_NAMES = ['fist', 'one-finger-up', 'open-palm', 'thumps-up', 'two-fingers']
IMG_SIZE = (224, 224)

# Debug
DEBUG = True

# ---------------- Globals ----------------
app = Flask(__name__)
current_frame = None
frame_lock = threading.Lock()
shutdown_flag = False

# Ultrasonic state
_ultra_buf = []

# ONNX model (attempt load)
try:
    gesture_sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = gesture_sess.get_inputs()[0].name
    output_name = gesture_sess.get_outputs()[0].name
    if DEBUG:
        print(f"🧠 ONNX gesture model loaded: {ONNX_MODEL_PATH}")
        print("   input name:", input_name, "| output name:", output_name)
except Exception as e:
    print(f"⚠️ ONNX load failed: {e}")
    gesture_sess = None
    input_name = None
    output_name = None


# runtime state
_avoid_state = False  # whether currently in avoidance (non-blocking)
_avoid_start = 0.0
_avoid_dir = None  # 'left' or 'right'
_last_avoid_at = 0.0

# Timed gesture action
_timed_action = None  # 'forward', 'left', 'right', etc.
_timed_action_start = 0.0
TIMED_ACTION_DURATION = 3.0  # seconds
_last_gesture = None
# Gesture stability tracking
_gesture_start_time = None
_gesture_stable = None
GESTURE_HOLD_TIME = 1.0  # seconds required for stable gesture

def mobilenet_v2_preprocess(img):
    """
    Keras MobileNetV2 preprocess_input:
      - expects RGB images
      - scales pixels from [0, 255] to [-1, 1]
    """
    img = img.astype("float32")
    img = img / 127.5 - 1.0
    return img

# ---------------- Helpers ----------------
def dbg(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def clamp99(x):
    return int(max(-99, min(99, x)))


def set_tank(bot, left, right):
    """Apply tank speeds (integers). Safe wrapper around AUPPBot."""
    try:
        l = clamp99(LEFT_SIGN * left)
        r = clamp99(RIGHT_SIGN * right)
        bot.motor1.speed(l);
        bot.motor2.speed(l)
        bot.motor3.speed(r);
        bot.motor4.speed(r)
    except Exception as e:
        dbg("❌ Motor error:", e)


def start_timed_action(action_name):
    global _timed_action, _timed_action_start
    _timed_action = action_name
    _timed_action_start = time.time()
    dbg(f"[ACTION] Started timed action: {action_name}")


# ---------------- Flask stream ----------------
def generate_frames():
    global current_frame, shutdown_flag
    while not shutdown_flag:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.01)
                continue
            try:
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                frame = buffer.tobytes()
            except Exception as e:
                dbg("❌ Frame encode error:", e)
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return '<html><body><img src="/video_feed" width="640" height="480"></body></html>'


# ---------------- Ultrasonic ----------------
def ultrasonic_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.output(TRIG, False)
    time.sleep(0.05)


def read_distance_cm():
    global _ultra_last_time
    now = time.time()
    if now - _ultra_last_time < ULTRA_MIN_INTERVAL:
        return None
    _ultra_last_time = now
    try:
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        t0 = time.time()
        while GPIO.input(ECHO) == 0:
            if time.time() - t0 > 0.02:
                return None
        pulse_start = time.time()
        while GPIO.input(ECHO) == 1:
            if time.time() - pulse_start > 0.02:
                return None
        pulse_end = time.time()
        dist_cm = (pulse_end - pulse_start) * 17150.0
        return round(dist_cm, 2)
    except Exception as e:
        dbg("ultra read err:", e)
        return None


# ---------------- Gesture detection (MobileNetV2) ----------------
# Example mapping of class IDs to gesture names (fallback)
GESTURE_CLASS_MAP = {
    0: "open_palm",
    1: "fist",
    2: "one_finger",
    3: "two_fingers",
    4: "thumbs_up",
}

def _get_gesture_name(cls_id, names_dict=None):
    """
    Normalize class id (from MobileNetV2 or YOLO) to our gesture keywords.
    """
    raw = ""

    # 1) Try YOLO-style names dict if provided
    if names_dict is not None:
        try:
            raw = names_dict[int(cls_id)].lower()
        except Exception:
            raw = ""

    # 2) If still empty, try CLASS_NAMES (MobileNetV2)
    if not raw:
        try:
            raw = CLASS_NAMES[int(cls_id)].lower()
        except Exception:
            raw = GESTURE_CLASS_MAP.get(int(cls_id), "").lower()

    # 3) Normalize formatting
    raw = raw.replace("-", "_").replace(" ", "_")

    # 4) Map model names → robot keywords
    if raw in ["fist"]:
        return "fist"

    if raw in ["one_finger_up", "one_finger", "onefingerup"]:
        return "one_finger"

    if raw in ["open_palm", "open_hand", "palm"]:
        return "open_palm"

    # Note: handle your label typo "thumps-up" → "thumps_up"
    if raw in ["thumbs_up", "thumb_up", "thumbsup", "thumps_up"]:
        return "thumbs_up"

    if raw in ["two_finger", "two_fingers", "two_finger_up"]:
        return "two_fingers"

    # fallback (rare)
    return raw

def detect_gesture(frame):
    """
    Run ONNX MobileNetV2 classifier on frame and return:
      gesture_name: 'open_palm' | 'fist' | 'one_finger' | 'two_fingers' | 'thumbs_up' | None
      vis_boxes:    [(x1, y1, x2, y2, gesture_name, conf), ...] for visualization

    We classify a center crop of the frame and draw one box over that area.
    """
    if gesture_sess is None:
        return None, []

    try:
        h, w = frame.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None, []

        # BGR -> RGB, resize, preprocess
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)

        img = mobilenet_v2_preprocess(img)   # [-1, 1] scaling
        img = np.expand_dims(img, axis=0)    # (1, 224, 224, 3)

        # ONNX inference
        preds = gesture_sess.run([output_name], {input_name: img})[0][0]

        conf = float(np.max(preds))
        cls_id = int(np.argmax(preds))
        gname = _get_gesture_name(cls_id)

        if DEBUG:
            try:
                raw_name = CLASS_NAMES[cls_id]
            except Exception:
                raw_name = "N/A"
            dbg(f"[ONNX] cls_id={cls_id}, raw='{raw_name}', mapped='{gname}', conf={conf:.3f}")

        vis_boxes = [(x1, y1, x2, y2, gname, conf)]

        return gname, vis_boxes

    except Exception as e:
        dbg("ONNX gesture detect error:", e)
        return None, []


# ---------------- Avoidance (non-blocking state machine) ----------------
def start_avoid(direction):
    global _avoid_state, _avoid_start, _avoid_dir, _last_avoid_at
    _avoid_dir = direction
    _avoid_start = time.time()
    _avoid_state = True
    _last_avoid_at = time.time()
    dbg(f"[AVOID] start dir={direction} at {_avoid_start:.2f}")


# ---------------- Control Loop (priority: ultrasonic check -> gesture -> avoid) ----------------
def robot_control_loop(bot, cap):
    """
    Control loop:
      - Read camera
      - Check ultrasonic for safety
      - Detect hand gesture with MobileNetV2 classifier
      - Drive based on gesture:
          'fist'       → forward (timed)
          'one_finger' → turn left (timed)
          'two_fingers'→ turn right (timed)
          'thumbs_up'  → reverse (timed)
          'open_palm' or no gesture → stop
    """
    global _last_gesture
    _last_gesture = None

    global current_frame, shutdown_flag, _ultra_buf, _timed_action, _timed_action_start

    last_print = 0
    frame_counter = 0
    _ultra_buf = []

    while not shutdown_flag:
        ok, frame = cap.read()
        if not ok:
            dbg("❌ Failed camera read")
            time.sleep(0.05)
            continue

        # same rotate + resize as before
        if ROTATE_90_CW:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, (W, H))
        h, w = frame.shape[:2]

        # ========== Ultrasonic safety ==========
        dist = read_distance_cm()
        if dist is not None:
            _ultra_buf.append(dist)
            if len(_ultra_buf) > ULTRA_CONSECUTIVE:
                _ultra_buf.pop(0)
        else:
            _ultra_buf.clear()

        too_close = (
                len(_ultra_buf) == ULTRA_CONSECUTIVE and
                all(d is not None and d < STOP_THRESHOLD_CM for d in _ultra_buf)
        )

        # If obstacle is too close → always stop, regardless of gesture
        if too_close:
            set_tank(bot, 0, 0)
            dbg(f"[SAFETY STOP] distance buffer={_ultra_buf}")
            # Visualization + stream
            vis = frame.copy()
            cv2.putText(vis, f"STOP (Obstacle {min(_ultra_buf):.1f} cm)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            with frame_lock:
                current_frame = vis
            time.sleep(0.02)
            frame_counter += 1
            continue

        # ========== Gesture detection ==========
        gesture, vis_boxes = detect_gesture(frame)
        # ========== EMERGENCY STOP (OPEN PALM) ==========
        if gesture == "open_palm":
            _timed_action = None  # cancel any running timed action
            set_tank(bot, 0, 0)  # stop motors immediately
            drive_status = "EMERGENCY STOP (open palm)"

            # update frame for stream
            vis = frame.copy()
            cv2.putText(vis, "EMERGENCY STOP",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            with frame_lock:
                current_frame = vis

            time.sleep(0.01)
            continue  # skip everything else in this loop

        # Default drive status each loop
        drive_status = "IDLE"

        # ======= TIMED ACTION CHECK (non-blocking) =======
        if _timed_action is not None:
            elapsed = time.time() - _timed_action_start
            if elapsed < TIMED_ACTION_DURATION:
                # execute the saved action
                if _timed_action == "forward":
                    set_tank(bot, BASE, BASE)
                elif _timed_action == "left":
                    set_tank(bot, BASE - DELTA, BASE + DELTA)
                elif _timed_action == "right":
                    set_tank(bot, BASE + DELTA, BASE - DELTA)
                elif _timed_action == "reverse":
                    set_tank(bot, -BASE, -BASE)

                drive_status = f"TIMED ACTION {_timed_action.upper()} ({elapsed:.1f}s)"

                # update frame and continue loop (skip gesture reading)
                vis = frame.copy()
                with frame_lock:
                    current_frame = vis
                time.sleep(0.01)
                continue  # skip gesture-driven control
            else:
                dbg("[ACTION] Timed action finished")
                _timed_action = None

        # ---------------- Stable Gesture Logic ----------------
        global _gesture_start_time, _gesture_stable

        now = time.time()

        if gesture != _gesture_stable:
            # Gesture changed -> reset stability timer
            _gesture_start_time = now
            _gesture_stable = gesture

        # Check if gesture has been held long enough
        gesture_ready = False
        if _gesture_stable is not None:
            if (now - _gesture_start_time) >= GESTURE_HOLD_TIME:
                gesture_ready = True

        # Only trigger action when gesture is stable & changed from last action
        if gesture_ready and _gesture_stable != _last_gesture:

            g = _gesture_stable

            if g == "open_palm":
                _timed_action = None
                set_tank(bot, 0, 0)
                drive_status = "STOP (open palm)"

            elif g == "fist":
                start_timed_action("forward")

            elif g == "one_finger":
                start_timed_action("left")

            elif g == "two_fingers":
                start_timed_action("right")

            elif g == "thumbs_up":
                start_timed_action("reverse")

            _last_gesture = g

        # ========= After timed action ends, STOP for safety =========
        if _timed_action is None:
            set_tank(bot, 0, 0)

        # ========== Visualization ==========
        vis = frame.copy()

        # Draw detected gesture box (full crop)
        for (x1, y1, x2, y2, gname, conf) in vis_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{gname} {conf:.2f}"
            cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Overlay status + ultrasonic info
        txt = f"{drive_status} | ULTRA_BUF={_ultra_buf}"
        cv2.putText(vis, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with frame_lock:
            current_frame = vis

        # Debug print (throttled)
        if time.time() - last_print > 0.4:
            dbg(f"[GESTURE] {drive_status} | ULTRA_BUF={_ultra_buf} | gesture={gesture}")
            last_print = time.time()

        frame_counter += 1
        time.sleep(0.005)


# ---------------- Main ----------------
def main():
    global shutdown_flag
    dbg("🤖 Starting robot")

    # connect robot
    try:
        bot = AUPPBot(PORT, BAUD, auto_safe=True)
        dbg("✅ Robot connected")
    except Exception as e:
        print("❌ Failed to connect to robot:", e)
        return

    # camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    dbg("✅ Camera opened")

    # ultrasonic setup
    ultrasonic_setup()
    dbg(f"📡 Ultrasonic ready (TRIG={TRIG}, ECHO={ECHO})")

    # optional servo
    try:
        bot.servo1.angle(40)
    except Exception:
        pass

    # start control thread
    ctl = threading.Thread(target=robot_control_loop, args=(bot, cap), daemon=True)
    ctl.start()

    # network info
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = "raspberrypi.local"

    print("\n" + "=" * 50)
    print(f"🎥 Stream: http://{ip_address}:5000")
    print("=" * 50)

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        dbg("Stopping (KeyboardInterrupt)")
    finally:
        shutdown_flag = True
        time.sleep(0.2)
        try:
            set_tank(bot, 0, 0)
            bot.stop_all()
            bot.close()
        except:
            pass
        try:
            cap.release()
        except:
            pass
        try:
            GPIO.cleanup()
        except:
            pass
        dbg("✅ Clean exit")


if __name__ == "__main__":
    main()
