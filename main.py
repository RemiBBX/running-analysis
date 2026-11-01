from pathlib import Path

import cv2

from src.video import get_frame

video_path = Path("data/raw/video1.mp4")

if __name__ == "__main__":
    frame = get_frame(video_path, 5.2)
    cv2.imshow("frame", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
