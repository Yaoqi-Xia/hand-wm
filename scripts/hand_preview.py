"""Webcam hand tracking preview (no simulation).

Shows the 21 hand landmarks and the values that would be sent to the finger.
  C    re-calibrate (current hand pose = finger home pose)
  ESC  quit
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from handwm.camera_source import CameraSource

WIN = "hand preview"


def main():
    src = CameraSource(cam_index=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(__doc__)
    try:
        while True:
            img = src.preview()
            if img is not None:
                ci = src.read(time.monotonic())
                cv2.putText(img, f"latency {ci.data_age * 1000:4.0f} ms  valid={ci.valid}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.imshow(WIN, img)
            key = cv2.waitKey(15) & 0xFF
            if key == 27:
                break
            if key in (ord("c"), ord("C")):
                src.calibrate()
            if img is not None and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
