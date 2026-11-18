#!/usr/bin/env python
"""Live camera preview of the input module: landmarks + streaming normalised features.

  python scripts/live_preview.py                 # webcam 0
  python scripts/live_preview.py --source clip.mp4 --dominant left
Press Q or ESC to quit.  This is the feed a trained LSTM/GRU would consume.
"""
import _bootstrap  # noqa: F401
import argparse
import time

import cv2

from signvoice.capture import FeatureStream, draw_landmarks
from signvoice.config import MASK_ORDER


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--dominant", choices=["left", "right"], default="right")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--window", type=int, default=64, help="sequence length handed to a model")
    args = ap.parse_args()

    n, t0 = 0, time.monotonic()
    with FeatureStream(args.source, dominant=args.dominant, width=args.width, height=args.height) as stream:
        for item in stream:
            n += 1
            shown = cv2.flip(item.frame, 1) if stream.source.is_camera else item.frame.copy()
            draw_landmarks(shown, item.landmarks, mirror=stream.source.is_camera)
            fps = n / max(time.monotonic() - t0, 1e-6)
            seen = ", ".join(k for k, v in zip(MASK_ORDER, item.landmarks.present()) if v) or "nothing"
            status = f"{fps:4.1f} fps | detected: {seen} | features: " + (
                f"{item.features.shape[0]}-d" if item.features is not None else "waiting for signer")
            cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 255, 200), 2, cv2.LINE_AA)
            cv2.imshow("Signvoice live preview", shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    print(f"window for a model would be shape {stream.window(args.window).shape}")


if __name__ == "__main__":
    main()
