from pathlib import Path

import cv2

video_name = "video1.mp4"
path = Path("data/raw") / video_name


def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("Echec de l'ouverture de la video")

    info = {}
    info["fps"] = cap.get(cv2.CAP_PROP_FPS)
    info["frame_count"] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    info["width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    info["height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    info["duration"] = info["frame_count"] / info["fps"]

    cap.release()

    return info


def read_video(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError("Échec de l'ouverture de la vidéo")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo.")
            break
        cv2.imshow("Lecture video openCV", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_frame(path, timestamp):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("Echec de l'ouverture de la video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    frame_id = int(timestamp * fps)

    if timestamp < 0 or timestamp >= duration:
        cap.release()
        raise ValueError("Le timestamp est en dehors de la vidéo.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise ValueError("Impossible de lire la frame.")

    return frame


def process_video(path, process_frame = None) :
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
            raise ValueError("Échec de l'ouverture de la vidéo")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo.")
            break

        if process_frame is not None : 
                frame = process_frame(frame)
        cv2.imshow("Lecture video openCV", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()