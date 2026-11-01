from pathlib import Path

import cv2

from src.pose_estimation import detect_person, load_person_detector
from src.video import get_frame

video_path = Path("data/raw/video1.mp4")

if __name__ == "__main__":
    frame = get_frame(video_path, 5)
    person_image_processor, person_model = load_person_detector()

    person_boxes = detect_person(
        frame, person_image_processor=person_image_processor, person_model=person_model
    )

    for bbox in person_boxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
