from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUMBER_OF_SAMPLES = 256 * 100
NUMBER_OF_FEATURES = 32
NOISE_STANDARD_DEVIATION = 0.1
INPUT_FEATURE_SCALE = 1.0


class TargetFunction(Enum):
    CONVEX = "convex"


def convex_target_function(input_features: np.ndarray) -> np.ndarray:
    return np.sum(input_features**2, axis=1)


def compute_target(
    target_function: TargetFunction, input_features: np.ndarray
) -> np.ndarray:
    if target_function == TargetFunction.CONVEX:
        return convex_target_function(input_features)
    raise ValueError(f"Unsupported target function: {target_function}")


def main() -> None:
    np.random.seed(RANDOM_SEED)

    input_features = np.random.uniform(
        -INPUT_FEATURE_SCALE,
        INPUT_FEATURE_SCALE,
        size=(NUMBER_OF_SAMPLES, NUMBER_OF_FEATURES),
    )

    target_function = TargetFunction.CONVEX
    target_values = compute_target(target_function, input_features) + np.random.normal(
        0,
        NOISE_STANDARD_DEVIATION,
        size=NUMBER_OF_SAMPLES,
    )

    column_names = [
        *(f"input_feature_{index + 1}" for index in range(NUMBER_OF_FEATURES)),
        "target",
    ]
    dataset_values = np.hstack((input_features, target_values.reshape(-1, 1)))
    data_frame = pd.DataFrame(dataset_values, columns=column_names)

    output_directory = Path("reference") / "data"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file_path = output_directory / f"synthetic_{target_function.value}_large.csv"
    data_frame.to_csv(output_file_path, index=False, header=False)
    print(f"Generated dataset: {output_file_path}")


if __name__ == "__main__":
    main()
