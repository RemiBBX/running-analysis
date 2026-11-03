from pathlib import Path

from running_analysis.pipeline import process_frame
from running_analysis.pose_estimation import (
    load_person_detector,
    load_pose_estimator,
)
from running_analysis.video import process_video


def main():
    video_path = Path("data/raw/video2.mp4")

    person_detector = load_person_detector()
    pose_estimator = load_pose_estimator()

    process_video(
        video_path,
        process_frame=lambda frame: process_frame(
            frame, person_detector, pose_estimator
        ),
    )


if __name__ == "__main__":
    main()
