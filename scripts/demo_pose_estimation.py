from pathlib import Path

import cv2
import torch

from running_analysis.models.detector.rtdetr import RTDetrDetector
from running_analysis.models.pose.vitpose import VitPoseEstimator
from running_analysis.video import get_frame
from running_analysis.visualization import draw_skeleton


def main():
    video_path = Path("data/raw/video1.mp4")

    frame = get_frame(video_path, 4.2)

    detector = RTDetrDetector()
    detector.load_model()
    person_boxes = detector.predict(frame)

    pose_estimator = VitPoseEstimator()
    pose_estimator.load_model()
    pose_results = pose_estimator.predict(frame, person_boxes)

    # 3. Inspection des résultats
    print(f"Nombre de personnes détectées : {len(pose_results)}")

    for i, person in enumerate(pose_results):
        print(f"\nPersonne {i}")

        for key, value in person.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} : shape = {value.shape}")
            else:
                print(f"{key} : {type(value)}")

    # frame = draw_keypoints(frame, pose_results[0])
    frame = draw_skeleton(frame, person)
    cv2.imshow("keypoints", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
