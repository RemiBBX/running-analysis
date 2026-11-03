from pathlib import Path

import cv2

from running_analysis.pose_estimation import detect_person, load_person_detector
from running_analysis.video import get_frame


def draw_box(frame, person_boxes):
    for bbox in person_boxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Person detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    video_path = Path("data/raw/video1.mp4")

    frame = get_frame(video_path, 5)

    person_image_processor, person_model = load_person_detector()

    person_boxes = detect_person(
        frame,
        person_image_processor=person_image_processor,
        person_model=person_model,
    )

    draw_box(frame, person_boxes)


if __name__ == "__main__":
    main()
