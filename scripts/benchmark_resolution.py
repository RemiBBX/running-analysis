import time
from pathlib import Path

import cv2
import numpy as np
import torch

from running_analysis.models.detector.rtdetr import RTDetrDetector
from running_analysis.video import get_frame


RESOLUTIONS = {
    "3840x2160": (3840, 2160),
    "1920x1080": (1920, 1080),
    "1280x720": (1280, 720),
}

ITERATIONS = 30
WARMUP = 5
THRESHOLD = 0.3


def synchronize():
    """Synchronize GPU operations when using MPS or CUDA."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_time(function):
    """Measure execution time of a function."""
    synchronize()
    start = time.perf_counter()

    result = function()

    synchronize()
    elapsed = time.perf_counter() - start

    return result, elapsed


def print_results(times):
    """Display benchmark statistics."""

    print("\n" + "=" * 65)
    print(f"{'COMPONENT':<25} {'MEAN (ms)':>12} {'MEDIAN (ms)':>12} {'STD (ms)':>12}")
    print("-" * 65)

    total = 0

    for name, values in times.items():
        values_ms = np.array(values) * 1000

        mean = np.mean(values_ms)
        median = np.median(values_ms)
        std = np.std(values_ms)

        total += mean

        print(f"{name:<25} {mean:>12.2f} {median:>12.2f} {std:>12.2f}")

    print("-" * 65)
    print(f"{'TOTAL':<25} {total:>12.2f}")
    print(f"{'THEORETICAL FPS':<25} {1000 / total:>12.2f}")
    print("=" * 65)


def benchmark(detector, frame, width, height):

    resized_frame = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    model = detector.model
    processor = detector.processor

    times = {
        "Color conversion": [],
        "Preprocessing": [],
        "Backbone": [],
        "Encoder": [],
        "Decoder": [],
        "Postprocessing": [],
    }

    modules = {
        "Backbone": model.model.backbone,
        "Encoder": model.model.encoder,
        "Decoder": model.model.decoder,
    }

    for iteration in range(WARMUP + ITERATIONS):

        module_times = {}

        def register_hooks():

            handles = []

            for name, module in modules.items():

                def pre_hook(module, inputs, name=name):
                    synchronize()
                    module_times[name + "_start"] = time.perf_counter()

                def post_hook(module, inputs, outputs, name=name):
                    synchronize()
                    module_times[name + "_end"] = time.perf_counter()

                handles.append(
                    module.register_forward_pre_hook(pre_hook)
                )

                handles.append(
                    module.register_forward_hook(post_hook)
                )

            return handles

        hooks = register_hooks()

        # 1. Color conversion
        _, conversion_time = measure_time(
            lambda: cv2.cvtColor(
                resized_frame,
                cv2.COLOR_BGR2RGB
            )
        )

        # 2. Preprocessing
        inputs, preprocessing_time = measure_time(
            lambda: processor(
                images=img_rgb,
                return_tensors="pt"
            ).to(model.device)
        )

        # 3. Model inference
        with torch.no_grad():
            outputs, _ = measure_time(
                lambda: model(**inputs)
            )

        # 4. Postprocessing
        results, postprocessing_time = measure_time(
            lambda: processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor(
                    [(height, width)]
                ),
                threshold=THRESHOLD
            )
        )

        result = results[0]

        person_boxes = result["boxes"][
            result["labels"] == 0
        ].cpu().numpy()

        synchronize()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Ignore warmup iterations
        if iteration >= WARMUP:

            times["Color conversion"].append(conversion_time)
            times["Preprocessing"].append(preprocessing_time)
            times["Postprocessing"].append(postprocessing_time)

            for name in modules:

                duration = (
                    module_times[name + "_end"]
                    - module_times[name + "_start"]
                )

                times[name].append(duration)

    print_results(times)

    print(f"\nPersons detected: {len(person_boxes)}")

    return times


def main():

    video_path = Path("data/raw/video1.mp4")

    frame = get_frame(video_path, 4.2)

    detector = RTDetrDetector()
    detector.load_model()

    print("\nRT-DETR MODEL")
    print(f"Device: {next(detector.model.parameters()).device}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Warmup: {WARMUP}")

    for name, (width, height) in RESOLUTIONS.items():

        print(f"\n\nBENCHMARK RESOLUTION: {name}")

        benchmark(
            detector,
            frame,
            width,
            height
        )


if __name__ == "__main__":
    main()