from pathlib import Path

import cv2

from running_analysis.pipeline import process_frame
from running_analysis.pose_estimation import (
    load_person_detector,
    load_pose_estimator,
)
from running_analysis.video import get_frame


def main():
    video_path = Path("data/raw/video1.mp4")
    frame = get_frame(video_path, 4.2)

    # 1. Détection de la personne
    person_detector = load_person_detector()
    pose_estimator = load_pose_estimator()

    process_frame(frame, person_detector, pose_estimator, threshold=0.3)
    cv2.imshow("keypoints", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
