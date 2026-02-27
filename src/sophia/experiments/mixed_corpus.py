"""Mixed corpus emotional state experiment.

Feed the matrix a stream of mixed emotional experiences where one emotion
dominates. Characterize what the resulting filter does to new input.

The matrix should naturally converge on the dominant emotional signal.
"""

import numpy as np
from sophia.experiments.agents.embedding import EmbeddingAgent
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import OuterProductUpdateAgent


# Diverse emotional texts — no labels used during training
TEXTS_BY_CATEGORY = {
    "angry": [
        "I am furious about this situation and cannot believe it happened",
        "This is completely unacceptable and makes me incredibly angry",
        "I am enraged by the incompetence and negligence on display",
        "This infuriating problem keeps getting worse and nobody cares",
        "I am livid and fed up with being ignored and dismissed",
        "The sheer arrogance of that decision makes my blood boil",
        "I cannot stand this level of disrespect any longer",
        "Every time I think about it I get more furious",
    ],
    "curious": [
        "I wonder how this mechanism actually works under the hood",
        "That is a fascinating pattern I have never noticed before",
        "What would happen if we approached this from a different angle",
        "I am intrigued by the unexpected connection between these ideas",
        "How does this phenomenon emerge from such simple rules",
        "There must be an underlying structure we are not seeing yet",
        "What if the conventional explanation is completely wrong",
        "I keep finding new layers every time I look at this problem",
    ],
    "joyful": [
        "This is wonderful news and I am so happy to hear it",
        "I feel a deep sense of gratitude for everything that happened",
        "What an amazing surprise that completely made my day",
        "I am overjoyed by how well everything turned out",
        "This moment of pure happiness is something I will treasure",
        "Everything came together perfectly and I could not be happier",
        "The warmth and kindness of that gesture moved me deeply",
        "I am celebrating this incredible achievement with so much pride",
    ],
    "sad": [
        "I feel an overwhelming sense of loss that will not go away",
        "The disappointment is crushing and I do not know how to cope",
        "Everything feels empty and meaningless right now",
        "I cannot stop thinking about what went wrong and why",
        "The loneliness is unbearable and nothing seems to help",
        "I feel broken inside and unable to see any hope",
        "This grief is something I carry with me every single day",
        "Nothing I do can fill the void that has been left behind",
    ],
    "neutral": [
        "The meeting is scheduled for three o'clock on Tuesday",
        "Please update the spreadsheet with the quarterly numbers",
        "The package arrived and has been placed on your desk",
        "The temperature today is expected to be around sixty degrees",
        "The report summarizes the findings from last month",
        "The train departs at seven fifteen from platform four",
        "The document has been filed in the shared drive folder",
        "Lunch will be served in the main conference room at noon",
    ],
}


def build_mixed_corpus(
    dominant: str, dominant_ratio: float, total: int, rng: np.random.Generator
) -> list[str]:
    """Build a corpus where one emotion dominates at the given ratio.

    The rest is split evenly among other categories.
    Texts are shuffled — no ordering information.
    """
    categories = list(TEXTS_BY_CATEGORY.keys())
    others = [c for c in categories if c != dominant]

    n_dominant = int(total * dominant_ratio)
    n_other_each = (total - n_dominant) // len(others)

    corpus = []
    # Sample with replacement from dominant
    dominant_texts = TEXTS_BY_CATEGORY[dominant]
    corpus.extend(rng.choice(dominant_texts, size=n_dominant, replace=True).tolist())

    # Sample from others
    for cat in others:
        cat_texts = TEXTS_BY_CATEGORY[cat]
        corpus.extend(rng.choice(cat_texts, size=n_other_each, replace=True).tolist())

    rng.shuffle(corpus)
    return corpus


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def run_mixed_corpus_experiment(
    dominant: str = "angry",
    dominant_ratio: float = 0.5,
    corpus_size: int = 40,
    alpha: float = 0.1,
    dim: int = 1536,
    seed: int = 42,
) -> dict:
    """Run the mixed corpus experiment.

    1. Build a mixed corpus with one emotion dominating
    2. Embed everything (corpus + category centroids for post-hoc analysis)
    3. Feed the shuffled corpus through the matrix updater — no labels
    4. Characterize: pass probe texts through the filter, measure what changed
    """
    rng = np.random.default_rng(seed)
    embedder = EmbeddingAgent(model="text-embedding-3-small", dim=dim)

    # Build the mixed corpus
    corpus = build_mixed_corpus(dominant, dominant_ratio, corpus_size, rng)

    # Embed category centroids (for post-hoc characterization only)
    category_centroids = {}
    for cat, texts in TEXTS_BY_CATEGORY.items():
        embs = [embedder.process(t) for t in texts]
        category_centroids[cat] = np.mean(embs, axis=0)

    # Embed the corpus
    corpus_embeddings = [embedder.process(t) for t in corpus]

    # Embed fresh probe texts (unseen during training)
    probe_texts = [
        "I need to review the budget proposal before Friday",  # neutral-ish
        "The results of the analysis were somewhat unexpected",  # mildly curious
        "We should discuss the implications of this decision",  # neutral-ish
        "That approach has some interesting properties worth exploring",  # curious-ish
        "The situation has become quite difficult to manage",  # mildly negative
    ]
    probe_embeddings = [embedder.process(t) for t in probe_texts]

    # Create matrix and updater
    matrix = MatrixAgent(dim=dim, std=0.0, seed=seed)
    updater = OuterProductUpdateAgent(matrix_agent=matrix, alpha=alpha)

    # Baseline: filter probes through identity-ish matrix
    baseline_probe_results = {}
    for cat, centroid in category_centroids.items():
        sims = [cosine(matrix.process(p), centroid) for p in probe_embeddings]
        baseline_probe_results[cat] = sims

    # Feed the mixed corpus — no labels, just embeddings in shuffled order
    matrix_snapshots = [matrix.snapshot()]
    for emb in corpus_embeddings:
        updater.process(emb)
        matrix_snapshots.append(matrix.snapshot())

    # Post-training: filter same probes through trained matrix
    trained_probe_results = {}
    for cat, centroid in category_centroids.items():
        sims = [cosine(matrix.process(p), centroid) for p in probe_embeddings]
        trained_probe_results[cat] = sims

    # Compute drift per category
    drift_per_category = {}
    for cat in category_centroids:
        baseline_mean = np.mean(baseline_probe_results[cat])
        trained_mean = np.mean(trained_probe_results[cat])
        drift_per_category[cat] = float(trained_mean - baseline_mean)

    # Which category had the most drift?
    max_drift_cat = max(drift_per_category, key=lambda k: drift_per_category[k])

    # Matrix characterization
    final_state = matrix.get_state()

    return {
        "dominant": dominant,
        "dominant_ratio": dominant_ratio,
        "corpus_size": corpus_size,
        "alpha": alpha,
        "seed": seed,
        "drift_per_category": drift_per_category,
        "max_drift_category": max_drift_cat,
        "correct": max_drift_cat == dominant,
        "baseline_probe_results": baseline_probe_results,
        "trained_probe_results": trained_probe_results,
        "matrix_norm": float(np.linalg.norm(final_state, "fro")),
        "matrix_max": float(np.max(np.abs(final_state))),
        "probe_texts": probe_texts,
    }


def main() -> None:
    print("Mixed Corpus Emotional State Experiment")
    print("=" * 70)

    # Test: does the matrix converge on the dominant emotion?
    print("\n=== DOMINANT EMOTION DETECTION ===")
    print("Train on mixed corpus, see which emotion the filter drifts toward.")
    print(
        f"{'dominant':>10} {'ratio':>6} | {'angry':>10} {'curious':>10} {'joyful':>10} {'sad':>10} {'neutral':>10} | {'detected':>10} {'correct':>8}"
    )

    for dominant in ["angry", "curious", "joyful", "sad", "neutral"]:
        r = run_mixed_corpus_experiment(
            dominant=dominant,
            dominant_ratio=0.5,
            corpus_size=40,
            alpha=0.1,
            seed=42,
        )
        d = r["drift_per_category"]
        print(
            f"{dominant:>10} {0.5:>6.1%} | {d['angry']:>+10.6f} {d['curious']:>+10.6f} {d['joyful']:>+10.6f} {d['sad']:>+10.6f} {d['neutral']:>+10.6f} | {r['max_drift_category']:>10} {'YES' if r['correct'] else 'NO':>8}"
        )

    # Test: does dominant ratio matter?
    print("\n=== RATIO SENSITIVITY (dominant=angry) ===")
    print(
        f"{'ratio':>6} | {'angry':>10} {'curious':>10} {'joyful':>10} {'sad':>10} {'neutral':>10} | {'detected':>10}"
    )
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = run_mixed_corpus_experiment(
            dominant="angry",
            dominant_ratio=ratio,
            corpus_size=40,
            alpha=0.1,
            seed=42,
        )
        d = r["drift_per_category"]
        print(
            f"{ratio:>6.0%} | {d['angry']:>+10.6f} {d['curious']:>+10.6f} {d['joyful']:>+10.6f} {d['sad']:>+10.6f} {d['neutral']:>+10.6f} | {r['max_drift_category']:>10}"
        )

    # Test: what about equal distribution?
    print("\n=== EQUAL DISTRIBUTION (no dominant emotion) ===")
    r = run_mixed_corpus_experiment(
        dominant="angry",
        dominant_ratio=0.2,
        corpus_size=50,
        alpha=0.1,
        seed=42,
    )
    d = r["drift_per_category"]
    print(f"{'angry':>10} {'curious':>10} {'joyful':>10} {'sad':>10} {'neutral':>10}")
    print(
        f"{d['angry']:>+10.6f} {d['curious']:>+10.6f} {d['joyful']:>+10.6f} {d['sad']:>+10.6f} {d['neutral']:>+10.6f}"
    )
    print(f"Detected: {r['max_drift_category']} (should be ambiguous/weak)")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
