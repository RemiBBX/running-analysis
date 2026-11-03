from pathlib import Path

from running_analysis.models.detector.rtdetr import RTDetrDetector
from running_analysis.models.pose.vitpose import VitPoseEstimator
from running_analysis.pipeline import process_frame
from running_analysis.video import process_video


def main():
    video_path = Path("data/raw/video2.mp4")

    detector = RTDetrDetector()
    detector.load_model()
    pose_estimator = VitPoseEstimator()
    pose_estimator.load_model()

    process_video(
        video_path,
        process_frame=lambda frame: process_frame(frame, detector, pose_estimator),
    )


if __name__ == "__main__":
    main()
