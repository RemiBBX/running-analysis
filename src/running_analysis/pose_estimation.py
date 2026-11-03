import cv2
import torch
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_pose_estimator():
    image_processor = AutoProcessor.from_pretrained(
        "usyd-community/vitpose-base-simple"
    )
    model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base-simple"
    ).to(device)

    return image_processor, model


def load_person_detector():
    person_image_processor = AutoProcessor.from_pretrained(
        "PekingU/rtdetr_r50vd_coco_o365"
    )
    person_model = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd_coco_o365"
    ).to(device)

    return person_image_processor, person_model


def detect_person(img, person_image_processor, person_model, threshold=0.3):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    inputs = person_image_processor(images=img_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(height, width)]), threshold=threshold
    )
    result = results[0]  # take first image results

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    # person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    # person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    return person_boxes


def estimate_pose(img, person_boxes, image_processor, model, threshold=0.3):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = image_processor(img_rgb, boxes=[person_boxes], return_tensors="pt").to(
        device
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(
        outputs, boxes=[person_boxes]
    )
    image_pose_result = pose_results[0]  # results for first imaged

    return image_pose_result
