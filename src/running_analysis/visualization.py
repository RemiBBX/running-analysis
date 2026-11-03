import cv2


def draw_keypoints(frame, person, threshold=0.5):
    keypoints = person["keypoints"].cpu().numpy()
    scores = person["scores"].cpu().numpy()

    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        if scores[i] >= threshold:
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    return frame


SKELETON = [
    (5, 6),  # Épaules
    (5, 11),  # Épaule gauche - hanche gauche
    (6, 12),  # Épaule droite - hanche droite
    (11, 12),  # Hanches
    (5, 7),  # Bras gauche
    (7, 9),
    (6, 8),  # Bras droit
    (8, 10),
    (11, 13),  # Jambe gauche
    (13, 15),
    (12, 14),  # Jambe droite
    (14, 16),
]


def draw_skeleton(frame, person, threshold=0.5):
    keypoints = person["keypoints"].cpu().numpy()
    scores = person["scores"].cpu().numpy()

    for k1, k2 in SKELETON:
        if scores[k1] >= threshold and scores[k2] >= threshold:
            k1 = keypoints[k1]
            k2 = keypoints[k2]
            x1, y1 = int(k1[0]), int(k1[1])
            x2, y2 = int(k2[0]), int(k2[1])
            cv2.line(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=5)

    return frame
