"""Run the emotional drift experiment and print results.

Usage: poetry run python -m sophia.experiments.run_drift
"""

from sophia.experiments.emotional_drift import run_emotional_drift_experiment


def main() -> None:
    print("Running emotional drift experiment...")
    print("=" * 60)

    for emotion in ["angry", "curious"]:
        artifacts = run_emotional_drift_experiment(
            seed=42,
            matrix_std=0.01,
            alpha=0.01,
            emotion=emotion,
        )

        print(f"\nEmotion: {emotion}")
        print(f"  Baseline mean similarity:  {artifacts['baseline_mean']:.6f}")
        print(f"  Filtered mean similarity:  {artifacts['filtered_mean']:.6f}")
        print(f"  Drift:                     {artifacts['drift']:+.6f}")
        print(
            f"  Per-input baseline:  {[f'{s:.4f}' for s in artifacts['baseline_similarities']]}"
        )
        print(
            f"  Per-input filtered:  {[f'{s:.4f}' for s in artifacts['filtered_similarities']]}"
        )

    print("\n" + "=" * 60)
    print("Done. Positive drift = neutral embeddings moved toward emotional region.")


if __name__ == "__main__":
    main()
