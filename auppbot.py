# file: auppbot.py
import serial, atexit, signal, threading, time, sys

def _clamp(v, lo, hi): 
    return lo if v < lo else hi if v > hi else v

class _Writer:
    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self._lock = threading.Lock()
    def send(self, i: int, d: int, v: int):
        pkt = bytes([(i & 0xFF), (d & 0xFF), (v & 0xFF)])
        with self._lock:
            self.ser.write(pkt)
            self.ser.flush()

class Motor:
    def __init__(self, w, motor_id: int):
        self._w, self.id = w, motor_id
    def speed(self, value: int):
        val = int(_clamp(value, -99, 99))
        direction = 0 if val >= 0 else 1
        self._w.send(self.id, direction, abs(val))
    def forward(self, speed: int): self.speed(abs(speed))
    def backward(self, speed: int): self.speed(-abs(speed))
    def stop(self): self._w.send(self.id, 0, 0)

class Servo:
    def __init__(self, w, index: int):
        self._w, self._id = w, (5 if index == 1 else 6)
    def angle(self, degrees: int):
        self._w.send(self._id, 0, int(_clamp(degrees, 0, 180)))
    def center(self): self.angle(90)

class AUPPBot:
    def __init__(self, port="/dev/ttyUSB0", baud=115200, timeout=0.2,
                 auto_safe=True, use_signals=True):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self._w = _Writer(self.ser)
        self.auto_safe = auto_safe

        # expose devices
        self.motor1 = Motor(self._w, 1)
        self.motor2 = Motor(self._w, 2)
        self.motor3 = Motor(self._w, 3)
        self.motor4 = Motor(self._w, 4)
        self.servo1 = Servo(self._w, 1)
        self.servo2 = Servo(self._w, 2)

        # setup exit behavior
        if auto_safe:
            atexit.register(self.safe)
        if use_signals:
            def _handler(sig, frame):
                self.safe()
                self.close()
                sys.exit(0)
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)

    # ---------- actions ----------
    def stop_all(self): self._w.send(0xFF, 0, 0)
    def safe(self):
        """Center servos & stop all motors"""
        try:
            self.servo1.center(); self.servo2.center()
            self.stop_all()
        except Exception: pass

    # ---------- lifecycle ----------
    def close(self):
        if self.auto_safe:
            self.safe()
        try: self.ser.close()
        except Exception: pass

    # ---------- run loop ----------
    def run_forever(self, msg="Running. Press Ctrl+C to stop."):
        """Keep program alive until Ctrl+C"""
        print(msg)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # handled by signal already
            pass
