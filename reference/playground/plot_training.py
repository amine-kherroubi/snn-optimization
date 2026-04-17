from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def parse_training_history(history_file_path: Path) -> tuple[list[int], list[float]]:
    epoch_numbers: list[int] = []
    mean_squared_error_values: list[float] = []

    with history_file_path.open("r", encoding="utf-8") as history_file:
        for line in history_file:
            if "Epoch" not in line or "MSE" not in line:
                continue

            parts = line.strip().split(",")
            epoch_number = int(parts[0].split()[1])
            mean_squared_error = float(parts[1].split()[1])
            epoch_numbers.append(epoch_number)
            mean_squared_error_values.append(mean_squared_error)

    return epoch_numbers, mean_squared_error_values


def main() -> None:
    history_file_path = Path("reference") / "log" / "history.txt"
    output_plot_path = Path("reference") / "log" / "training.png"

    try:
        epoch_numbers, mean_squared_error_values = parse_training_history(
            history_file_path
        )
    except FileNotFoundError:
        raise SystemExit(f"Training history file not found: {history_file_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(
        epoch_numbers,
        mean_squared_error_values,
        marker="o",
        label="Loss (Mean Squared Error)",
    )
    plt.title("Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_plot_path, dpi=300)
    print(f"Saved plot: {output_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
