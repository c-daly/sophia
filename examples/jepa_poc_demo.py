#!/usr/bin/env python3
"""JEPA PoC Backend Demo Script.

This script demonstrates the PoC JEPA backend capabilities:
1. Backend initialization and configuration
2. Simulation with k-step rollouts
3. Media sample processing with embedding generation
4. Health status and metrics

Usage:
    # With stub backend (default)
    python examples/jepa_poc_demo.py

    # With PoC backend (requires PyTorch + checkpoint)
    JEPA_BACKEND=poc JEPA_WEIGHTS_PATH=/path/to/checkpoint.pth python examples/jepa_poc_demo.py

    # With PoC backend on specific GPU
    JEPA_BACKEND=poc JEPA_WEIGHTS_PATH=/path/to/checkpoint.pth JEPA_DEVICE=cuda:1 python examples/jepa_poc_demo.py
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Check for torch availability
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_status(label: str, value, indent: int = 2) -> None:
    """Print a status line."""
    print(f"{' '*indent}{label}: {value}")


def demo_environment() -> None:
    """Show environment and configuration."""
    print_header("Environment")
    
    print_status("Python", sys.version.split()[0])
    print_status("PyTorch available", TORCH_AVAILABLE)
    
    if TORCH_AVAILABLE:
        print_status("PyTorch version", torch.__version__)
        print_status("CUDA available", GPU_AVAILABLE)
        if GPU_AVAILABLE:
            print_status("GPU device", torch.cuda.get_device_name(0))
            print_status("GPU memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print()
    print_status("JEPA_BACKEND", os.getenv("JEPA_BACKEND", "stub (default)"))
    print_status("JEPA_WEIGHTS_PATH", os.getenv("JEPA_WEIGHTS_PATH", "not set"))
    print_status("JEPA_DEVICE", os.getenv("JEPA_DEVICE", "cuda:0 (default)"))
    print_status("JEPA_DTYPE", os.getenv("JEPA_DTYPE", "fp16 (default)"))


def demo_backend_initialization() -> None:
    """Demonstrate backend initialization."""
    print_header("Backend Initialization")
    
    from sophia.jepa.runner import JEPARunner
    
    runner = JEPARunner()
    
    print_status("Runner created", "✓")
    print_status("Backend", runner.backend_name)
    print_status("Model version", runner.model_version)
    
    # Get health status
    status = runner.get_health_status()
    print()
    print("  Health Status:")
    for key, value in status.items():
        print_status(key, value, indent=4)


def demo_simulation() -> None:
    """Demonstrate simulation with k-step rollouts."""
    print_header("Simulation Demo")
    
    from sophia.jepa.runner import JEPARunner
    from sophia.jepa.models import SimulationContext, Entity, SensorReference
    
    runner = JEPARunner()
    
    # Create simulation context
    entities = [
        Entity(
            id="red_block",
            type="object",
            properties={"mass": 0.5, "color": "red", "shape": "cube"},
            position={"x": 0.0, "y": 0.0, "z": 0.1},
        ),
        Entity(
            id="robot_arm",
            type="agent",
            properties={"status": "idle", "gripper": "open"},
        ),
    ]
    
    sensor_refs = [
        SensorReference(
            sensor_id="camera_1",
            sensor_type="camera",
            frame_id="base_link",
        ),
    ]
    
    actions = [
        {"type": "MOVE", "target": "red_block"},
        {"type": "GRASP", "target": "red_block"},
        {"type": "MOVE", "target": "bin", "target_position": {"x": 1.0, "y": 0.0, "z": 0.5}},
    ]
    
    context = SimulationContext(
        entities=entities,
        sensor_refs=sensor_refs,
        actions=actions,
    )
    
    print_status("Entities", len(entities))
    print_status("Sensors", len(sensor_refs))
    print_status("Actions", len(actions))
    print()
    
    # Run simulation
    print("  Running 5-step simulation...")
    result = runner.simulate(context, k_steps=5, assumptions=["robot is operational"])
    
    print()
    print_status("Simulation ID", result.simulation_id[:8] + "...")
    print_status("K-steps", result.k_steps)
    print_status("Model version", result.model_version)
    print_status("Overall confidence", f"{result.overall_confidence:.2%}")
    print_status("Imagined states", len(result.imagined_states))
    print_status("Imagined processes", len(result.imagined_processes))
    
    print()
    print("  State Rollout:")
    for state in result.imagined_states:
        print(f"    Step {state.step}: confidence={state.confidence:.2%}")


async def demo_media_processing() -> None:
    """Demonstrate media sample processing."""
    print_header("Media Processing Demo")
    
    from sophia.jepa.runner import JEPARunner
    from PIL import Image
    
    runner = JEPARunner()
    
    # Create a test image
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_image.jpg"
        
        # Create a simple test image
        img = Image.new("RGB", (224, 224), color="blue")
        # Add some features
        for i in range(50, 150):
            for j in range(50, 150):
                img.putpixel((i, j), (255, 0, 0))  # Red square
        img.save(img_path)
        
        print_status("Test image created", str(img_path))
        print_status("Image size", "224x224")
        print()
        
        # Process media sample
        print("  Processing media sample...")
        result = await runner.process_media_sample(
            sample_id="demo-sample-001",
            file_path=str(img_path),
            media_type="image",
            metadata={"source": "demo", "format": "JPEG"},
            question="What objects are visible?",
        )
        
        print()
        print_status("Sample ID", result["sample_id"])
        print_status("Media type", result["media_type"])
        print_status("Model version", result["model_version"])
        print_status("Confidence", f"{result['confidence']:.2%}")
        print_status("Backend", result.get("backend", "stub"))
        
        print()
        print("  Embeddings:")
        visual_emb = result["embeddings"]["visual"]
        physics_emb = result["embeddings"]["physics"]
        print_status("Visual embedding dim", len(visual_emb), indent=4)
        print_status("Physics embedding dim", len(physics_emb), indent=4)
        print_status("Visual embedding norm", f"{sum(x**2 for x in visual_emb)**0.5:.4f}", indent=4)
        print_status("Physics embedding norm", f"{sum(x**2 for x in physics_emb)**0.5:.4f}", indent=4)


def demo_metrics() -> None:
    """Demonstrate metrics and health status after inference."""
    print_header("Metrics After Inference")
    
    from sophia.jepa.runner import JEPARunner
    from sophia.jepa.models import SimulationContext, Entity
    
    runner = JEPARunner()
    
    # Run a few simulations
    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)
    
    print("  Running 3 simulations to accumulate metrics...")
    for i in range(3):
        runner.simulate(context, k_steps=2)
    
    status = runner.get_health_status()
    
    print()
    print("  Updated Health Status:")
    for key, value in status.items():
        if isinstance(value, float) and key.endswith("_ms"):
            print_status(key, f"{value:.2f} ms", indent=4)
        else:
            print_status(key, value, indent=4)


def main():
    """Run the JEPA PoC demo."""
    print("\n" + "="*60)
    print("       JEPA PoC Backend Demo")
    print("="*60)
    
    try:
        demo_environment()
        demo_backend_initialization()
        demo_simulation()
        asyncio.run(demo_media_processing())
        demo_metrics()
        
        print_header("Demo Complete")
        print("  All demos completed successfully! ✓")
        print()
        
    except Exception as e:
        print(f"\n  Error: {e}")
        print("\n  If using PoC backend, ensure:")
        print("    - PyTorch is installed: poetry install --with ml")
        print("    - JEPA_WEIGHTS_PATH points to a valid checkpoint")
        print("    - GPU is available (or set JEPA_DEVICE=cpu)")
        sys.exit(1)


if __name__ == "__main__":
    main()
