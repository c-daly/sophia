#!/usr/bin/env python3
"""Demonstration of V-JEPA PoC backend.

This script demonstrates the Proof-of-Concept V-JEPA backend implementation,
showing how it can be used for:
1. Media processing with visual/physics embeddings
2. K-step simulation rollouts
3. Configuration via environment variables

Usage:
    # Use PoC backend
    JEPA_BACKEND=poc python examples/poc_jepa_demo.py

    # Use stub backend (default)
    python examples/poc_jepa_demo.py
"""

import os
import asyncio
import tempfile
from PIL import Image

from sophia.jepa import JEPARunner
from sophia.jepa.models import SimulationContext, Entity


def create_sample_image() -> str:
    """Create a sample image for demonstration."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Create a simple colored image
        img = Image.new("RGB", (256, 256))
        pixels = img.load()

        # Create a gradient pattern
        for i in range(256):
            for j in range(256):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)

        img.save(f.name)
        return f.name


async def demo_media_processing(runner: JEPARunner, image_path: str):
    """Demonstrate media sample processing."""
    print("\n" + "=" * 70)
    print("DEMO 1: Media Sample Processing")
    print("=" * 70)

    result = await runner.process_media_sample(
        sample_id="demo_sample_1",
        file_path=image_path,
        media_type="image",
        metadata={"width": 256, "height": 256, "demo": True},
        question="What physical properties can be inferred from this scene?",
    )

    print(f"\nSample ID: {result['sample_id']}")
    print(f"Model Version: {result['model_version']}")
    print(f"Embedding Dimension: {result['embedding_dim']}")
    print(f"Confidence: {result['confidence']:.3f}")

    visual_emb = result["embeddings"]["visual"]
    physics_emb = result["embeddings"]["physics"]

    print(
        f"\nVisual Embedding: [{visual_emb[0]:.4f}, {visual_emb[1]:.4f}, ..., {visual_emb[-1]:.4f}]"
    )
    print(
        f"Physics Embedding: [{physics_emb[0]:.4f}, {physics_emb[1]:.4f}, ..., {physics_emb[-1]:.4f}]"
    )

    # Compute embedding statistics
    import numpy as np

    visual_norm = np.linalg.norm(visual_emb)
    physics_norm = np.linalg.norm(physics_emb)

    print("\nEmbedding norms:")
    print(f"  Visual: {visual_norm:.3f}")
    print(f"  Physics: {physics_norm:.3f}")


def demo_simulation(runner: JEPARunner):
    """Demonstrate k-step simulation rollout."""
    print("\n" + "=" * 70)
    print("DEMO 2: K-Step Simulation Rollout")
    print("=" * 70)

    # Create a simple scene with objects
    entities = [
        Entity(
            id="red_block",
            type="object",
            properties={"mass": 0.5, "color": "red", "shape": "cube"},
            position={"x": 0.0, "y": 0.0, "z": 0.2},
        ),
        Entity(
            id="blue_block",
            type="object",
            properties={"mass": 0.3, "color": "blue", "shape": "cube"},
            position={"x": 0.1, "y": 0.0, "z": 0.4},
        ),
        Entity(
            id="robot_arm",
            type="agent",
            properties={"status": "idle", "gripper": "open"},
            position={"x": -0.5, "y": 0.0, "z": 0.5},
        ),
    ]

    # Define action sequence
    actions = [
        {
            "type": "MOVE",
            "target": "robot_arm",
            "target_position": {"x": 0.0, "y": 0.0, "z": 0.2},
        },
        {"type": "GRASP", "target": "red_block"},
        {
            "type": "MOVE",
            "target": "robot_arm",
            "target_position": {"x": 0.3, "y": 0.0, "z": 0.2},
        },
        {"type": "RELEASE", "target": "red_block"},
    ]

    context = SimulationContext(
        entities=entities,
        actions=actions,
    )

    # Run simulation
    k_steps = 5
    result = runner.simulate(
        context,
        k_steps=k_steps,
        assumptions=["objects are graspable", "robot is functional"],
    )

    print(f"\nSimulation ID: {result.simulation_id}")
    print(f"Model Version: {result.model_version}")
    print(f"K-Steps: {result.k_steps}")
    print(f"Overall Confidence: {result.overall_confidence:.3f}")

    print(f"\nImagined Processes: {len(result.imagined_processes)}")
    for i, process in enumerate(result.imagined_processes):
        print(f"  {i+1}. {process.description} (confidence={process.confidence:.3f})")

    print("\nImagined States:")
    for state in result.imagined_states:
        print(f"  Step {state.step}: confidence={state.confidence:.3f}")

        # Show entity positions
        for entity in state.entities:
            if entity.position:
                pos = entity.position
                print(
                    f"    {entity.id}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})"
                )


def demo_backend_comparison():
    """Compare stub vs PoC backend behavior."""
    print("\n" + "=" * 70)
    print("DEMO 3: Backend Comparison")
    print("=" * 70)

    # Create simple test scenario
    entities = [
        Entity(id="test_obj", type="object", position={"x": 0.0, "y": 0.0, "z": 0.1})
    ]
    context = SimulationContext(entities=entities)

    # Test with stub backend
    os.environ["JEPA_BACKEND"] = "stub"
    stub_runner = JEPARunner()
    stub_result = stub_runner.simulate(context, k_steps=3)

    # Test with PoC backend
    os.environ["JEPA_BACKEND"] = "poc"
    poc_runner = JEPARunner()
    poc_result = poc_runner.simulate(context, k_steps=3)

    print("\nBackend Comparison:")
    print(f"{'Metric':<30} {'Stub':<20} {'PoC':<20}")
    print("-" * 70)
    print(
        f"{'Backend Class':<30} {stub_runner._backend.__class__.__name__:<20} {poc_runner._backend.__class__.__name__:<20}"
    )
    print(
        f"{'Model Version':<30} {stub_result.model_version:<20} {poc_result.model_version:<20}"
    )
    print(
        f"{'Overall Confidence':<30} {stub_result.overall_confidence:.4f}{'':<16} {poc_result.overall_confidence:.4f}"
    )
    print(
        f"{'# Processes':<30} {len(stub_result.imagined_processes):<20} {len(poc_result.imagined_processes):<20}"
    )
    print(
        f"{'# States':<30} {len(stub_result.imagined_states):<20} {len(poc_result.imagined_states):<20}"
    )

    print("\nConfidence Decay Pattern:")
    print(f"{'Step':<10} {'Stub Confidence':<20} {'PoC Confidence':<20}")
    print("-" * 50)
    for i in range(len(stub_result.imagined_states)):
        stub_conf = stub_result.imagined_states[i].confidence
        poc_conf = poc_result.imagined_states[i].confidence
        print(f"{i:<10} {stub_conf:.4f}{'':<16} {poc_conf:.4f}")


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("V-JEPA PoC Backend Demonstration")
    print("=" * 70)

    # Determine which backend to use
    backend_choice = os.getenv("JEPA_BACKEND", "stub")
    print(f"\nBackend: {backend_choice.upper()}")

    # Initialize runner
    runner = JEPARunner()
    print(f"Runner initialized with: {runner._backend.__class__.__name__}")

    # Create sample image
    image_path = create_sample_image()
    print(f"Created sample image: {image_path}")

    try:
        # Run demonstrations
        await demo_media_processing(runner, image_path)
        demo_simulation(runner)

        # Only run comparison if not already comparing
        if backend_choice == "stub":
            demo_backend_comparison()

    finally:
        # Cleanup
        try:
            os.unlink(image_path)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
    print("\nNext steps:")
    print(
        "  1. Run with PoC backend: JEPA_BACKEND=poc python examples/poc_jepa_demo.py"
    )
    print("  2. Configure weights path: JEPA_WEIGHTS_PATH=/path/to/weights")
    print("  3. Set device: JEPA_DEVICE=cuda:0")
    print("  4. Set dtype: JEPA_DTYPE=fp16")
    print()


if __name__ == "__main__":
    asyncio.run(main())
