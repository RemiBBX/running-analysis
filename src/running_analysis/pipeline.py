from running_analysis.visualization import draw_keypoints, draw_skeleton


def process_frame(frame, detector, pose_estimator):
    person_boxes = detector.predict(frame)

    n = len(person_boxes)
    print(f"Nombre de personnes détectées : {n}")
    if n == 0:
        return frame

    pose_results = pose_estimator.predict(frame, person_boxes)

    for i, person in enumerate(pose_results):
        frame = draw_skeleton(frame, person)
        frame = draw_keypoints(frame, person)

    return frame
