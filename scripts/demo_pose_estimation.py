from pathlib import Path

import cv2
import torch

from running_analysis.pose_estimation import (
    detect_person,
    estimate_pose,
    load_person_detector,
    load_pose_estimator,
)
from running_analysis.video import get_frame
from running_analysis.visualization import draw_keypoints, draw_skeleton


def main():
    video_path = Path("data/raw/video1.mp4")
    frame = get_frame(video_path, 4.2)

    # 1. Détection de la personne
    person_image_processor, person_model = load_person_detector()

    person_boxes = detect_person(
        frame,
        person_image_processor,
        person_model,
    )

    # 2. Estimation de pose
    pose_image_processor, pose_model = load_pose_estimator()

    pose_results = estimate_pose(
        frame,
        person_boxes,
        pose_image_processor,
        pose_model,
        threshold = 0.3
    )

    # 3. Inspection des résultats
    print(f"Nombre de personnes détectées : {len(pose_results)}")

    for i, person in enumerate(pose_results):
        print(f"\nPersonne {i}")

        for key, value in person.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} : shape = {value.shape}")
            else:
                print(f"{key} : {type(value)}")

    #frame = draw_keypoints(frame, pose_results[0])
    frame = draw_skeleton(frame, person)
    cv2.imshow("keypoints", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
