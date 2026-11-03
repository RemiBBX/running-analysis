import torch

from running_analysis.pose_estimation import detect_person, estimate_pose
from running_analysis.visualization import draw_keypoints, draw_skeleton

device = "mps" if torch.backends.mps.is_available() else "cpu"


def process_frame(frame, person_detector, pose_estimator, threshold=0.3):
    person_image_processor, person_model = person_detector
    image_processor, model = pose_estimator

    person_boxes = detect_person(
        frame, person_image_processor, person_model, threshold=threshold
    )
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    n = len(person_boxes)
    print(f"Nombre de personnes détectées : {n}")
    if n == 0:
        return frame

    pose_result = estimate_pose(
        frame, person_boxes, image_processor, model, threshold=threshold
    )

    for i, person in enumerate(pose_result):
        frame = draw_skeleton(frame, person)
        frame = draw_keypoints(frame, person)

    return frame
