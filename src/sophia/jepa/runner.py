"""JEPA runner with pluggable backends (stub by default).

This keeps the existing stub behavior for tests/local runs while allowing
the backend to be swapped for a real V-JEPA implementation via injection or
environment configuration.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Protocol, runtime_checkable

from sophia.jepa.models import (
    SimulationContext,
    SimulationResult,
    ImaginedProcess,
    ImaginedState,
    Entity,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class JEPABackend(Protocol):
    """Backend interface for JEPA operations."""

    def simulate(
        self,
        context: SimulationContext,
        k_steps: int = 5,
        assumptions: List[str] | None = None,
    ) -> SimulationResult: ...

    async def process_media_sample(
        self,
        sample_id: str,
        file_path: str,
        media_type: str,
        metadata: Dict[str, Any],
        question: str | None = None,
    ) -> Dict[str, Any]: ...


class StubJEPABackend:
    """Existing CPU-friendly stub implementation."""

    def __init__(
        self, model_version: str = "jepa-stub-v1.0", confidence_decay: float = 0.05
    ):
        self.model_version = model_version
        self.confidence_decay = confidence_decay
        self._inference_count = 0
        logger.info(f"Initialized JEPA stub backend: {model_version}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for health probes."""
        return {
            "backend": "stub",
            "model_version": self.model_version,
            "model_loaded": True,  # Stub is always "loaded"
            "gpu_available": False,
            "device": "cpu",
            "inference_count": self._inference_count,
        }

    def simulate(
        self,
        context: SimulationContext,
        k_steps: int = 5,
        assumptions: List[str] | None = None,
    ) -> SimulationResult:
        simulation_id = str(uuid.uuid4())
        assumptions = assumptions or []
        self._inference_count += 1

        logger.info(
            f"Starting JEPA simulation {simulation_id} with {k_steps} steps (stub)"
        )

        imagined_processes = self._generate_processes(
            context, k_steps, assumptions, simulation_id
        )

        imagined_states = self._generate_state_rollout(
            context, k_steps, assumptions, simulation_id
        )

        overall_confidence = sum(s.confidence for s in imagined_states) / len(
            imagined_states
        )

        result = SimulationResult(
            simulation_id=simulation_id,
            context=context,
            imagined_processes=imagined_processes,
            imagined_states=imagined_states,
            k_steps=k_steps,
            model_version=self.model_version,
            overall_confidence=overall_confidence,
        )

        logger.info(
            f"Simulation {simulation_id} complete: "
            f"{len(imagined_states)} states, "
            f"confidence {overall_confidence:.2f}"
        )

        return result

    def _generate_processes(
        self,
        context: SimulationContext,
        k_steps: int,
        assumptions: List[str],
        simulation_id: str,
    ) -> List[ImaginedProcess]:
        processes: List[ImaginedProcess] = []

        main_process = ImaginedProcess(
            process_id=f"{simulation_id}_process_dynamics",
            description="Forward dynamics prediction process",
            confidence=0.85,
            model_version=self.model_version,
            horizon=k_steps,
            assumptions=assumptions,
            imagined=True,
            properties={
                "type": "dynamics",
                "context_entities": len(context.entities),
                "context_sensors": len(context.sensor_refs),
            },
        )
        processes.append(main_process)

        if context.actions:
            for i, action in enumerate(context.actions):
                action_process = ImaginedProcess(
                    process_id=f"{simulation_id}_process_action_{i}",
                    description=f"Action process: {action.get('type', 'unknown')}",
                    confidence=0.80,
                    model_version=self.model_version,
                    horizon=k_steps,
                    assumptions=assumptions,
                    imagined=True,
                    properties={
                        "type": "action",
                        "action_data": action,
                        "action_index": i,
                    },
                )
                processes.append(action_process)

        return processes

    def _generate_state_rollout(
        self,
        context: SimulationContext,
        k_steps: int,
        assumptions: List[str],
        simulation_id: str,
    ) -> List[ImaginedState]:
        imagined_states: List[ImaginedState] = []
        current_entities = [entity.model_copy(deep=True) for entity in context.entities]

        for step in range(k_steps):
            confidence = max(0.0, 0.95 - (step * self.confidence_decay))

            next_entities = self._evolve_entities(current_entities, context, step)

            state_data = self._create_state_data(next_entities, context, step)

            imagined_state = ImaginedState(
                state_id=f"{simulation_id}_state_{step}",
                step=step,
                description=f"Imagined state at step {step}",
                confidence=confidence,
                model_version=self.model_version,
                horizon=k_steps,
                assumptions=assumptions,
                imagined=True,
                state_data=state_data,
                entities=next_entities,
            )
            imagined_states.append(imagined_state)

            current_entities = next_entities

        return imagined_states

    def _evolve_entities(
        self,
        entities: List[Entity],
        context: SimulationContext,
        step: int,
    ) -> List[Entity]:
        evolved = []

        for entity in entities:
            new_entity = entity.model_copy(deep=True)

            if entity.type == "object":
                if new_entity.position:
                    new_entity.position["x"] = new_entity.position.get("x", 0.0) + (
                        step * 0.001
                    )
                    new_entity.position["y"] = new_entity.position.get("y", 0.0) + (
                        step * 0.001
                    )

            elif entity.type == "agent":
                if "status" in new_entity.properties:
                    if step > 2:
                        new_entity.properties["status"] = "active"

            if context.actions and step < len(context.actions):
                action = context.actions[step]
                if action.get("target") == entity.id:
                    self._apply_action_to_entity(new_entity, action)

            evolved.append(new_entity)

        return evolved

    def _apply_action_to_entity(self, entity: Entity, action: Dict[str, Any]) -> None:
        action_type = action.get("type", "")

        if action_type == "MOVE":
            target_pos = action.get("target_position", {})
            if entity.position and target_pos:
                entity.position.update(target_pos)

        elif action_type == "GRASP":
            entity.properties["grasped"] = True

        elif action_type == "RELEASE":
            entity.properties["grasped"] = False

    def _create_state_data(
        self,
        entities: List[Entity],
        context: SimulationContext,
        step: int,
    ) -> Dict[str, Any]:
        state_data: Dict[str, Any] = {
            "step": step,
            "entity_count": len(entities),
            "metadata": {
                "talos_metadata": context.talos_metadata.model_dump(),
                "sensor_count": len(context.sensor_refs),
            },
        }

        for entity in entities:
            state_data[entity.id] = {
                "type": entity.type,
                "properties": entity.properties,
                "position": entity.position,
            }

        return state_data

    async def process_media_sample(
        self,
        sample_id: str,
        file_path: str,
        media_type: str,
        metadata: Dict[str, Any],
        question: str | None = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"Processing media sample {sample_id} ({media_type}) for physical understanding (stub)"
        )
        if question:
            logger.info(f"Perception question: {question}")

        embedding_dim = 768

        visual_embedding = [
            float(hash(f"{sample_id}_visual_{i}") % 1000) / 1000.0
            for i in range(embedding_dim)
        ]

        physics_embedding = [
            float(hash(f"{sample_id}_physics_{i}") % 1000) / 1000.0
            for i in range(embedding_dim)
        ]

        result = {
            "sample_id": sample_id,
            "media_type": media_type,
            "embeddings": {
                "visual": visual_embedding,
                "physics": physics_embedding,
            },
            "embedding_dim": embedding_dim,
            "model_version": self.model_version,
            "confidence": 0.85,
            "metadata": {
                "file_path": file_path,
                "question": question,
                "media_metadata": metadata,
            },
        }

        embeddings: Dict[str, List[float]] = result.get("embeddings", {})  # type: ignore
        logger.info(f"Generated {len(embeddings)} embeddings for sample {sample_id}")

        return result


class JEPARunner:
    """JEPA-based dynamics runner with selectable backend.

    Default remains the stub backend; a real V-JEPA backend can be injected or
    selected via the `JEPA_BACKEND` environment variable.
    """

    def __init__(
        self,
        model_version: str = "jepa-stub-v1.0",
        confidence_decay: float = 0.05,
        backend: JEPABackend | None = None,
    ):
        self.model_version = model_version
        self.confidence_decay = confidence_decay
        self._backend = backend or self._select_backend(model_version, confidence_decay)
        logger.info(
            f"Initialized JEPARunner with backend: {self._backend.__class__.__name__}"
        )

    @property
    def backend_name(self) -> str:
        """Get the name of the current backend."""
        return self._backend.__class__.__name__

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for health probes.

        Returns backend-specific health information for use in /health endpoint.
        """
        if hasattr(self._backend, "get_health_status"):
            status: Dict[str, Any] = self._backend.get_health_status()
            return status

        # Fallback for backends without health status
        return {
            "backend": self.backend_name,
            "model_version": self.model_version,
        }

    def simulate(
        self,
        context: SimulationContext,
        k_steps: int = 5,
        assumptions: List[str] | None = None,
    ) -> SimulationResult:
        return self._backend.simulate(
            context=context, k_steps=k_steps, assumptions=assumptions
        )

    async def process_media_sample(
        self,
        sample_id: str,
        file_path: str,
        media_type: str,
        metadata: Dict[str, Any],
        question: str | None = None,
    ) -> Dict[str, Any]:
        return await self._backend.process_media_sample(
            sample_id=sample_id,
            file_path=file_path,
            media_type=media_type,
            metadata=metadata,
            question=question,
        )

    def _select_backend(
        self, model_version: str, confidence_decay: float
    ) -> JEPABackend:
        """Select backend based on JEPA_BACKEND environment variable.

        Supported values:
        - 'stub' (default): CPU-friendly stub for tests/CI
        - 'poc': PoC backend with real V-JEPA model support (requires GPU + weights)
        - 'real': Alias for 'poc' (future: production-optimized backend)
        """
        backend_choice = os.getenv("JEPA_BACKEND", "stub").lower()

        if backend_choice == "stub":
            return StubJEPABackend(
                model_version=model_version, confidence_decay=confidence_decay
            )

        if backend_choice in ("poc", "real"):
            try:
                from sophia.jepa.backends.poc import PoCJEPABackend

                return PoCJEPABackend(
                    model_version=model_version.replace("stub", "poc"),
                    confidence_decay=confidence_decay,
                )
            except ImportError as e:
                logger.error(
                    f"Failed to import PoCJEPABackend: {e}. "
                    "Ensure PyTorch is installed: pip install torch"
                )
                raise RuntimeError(
                    f"JEPA_BACKEND={backend_choice} requires PyTorch. "
                    "Install with: pip install torch"
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize PoCJEPABackend: {e}")
                raise

        logger.warning(
            f"Unknown JEPA_BACKEND value '{backend_choice}'. "
            "Valid options: stub, poc, real. Falling back to stub."
        )
        return StubJEPABackend(
            model_version=model_version, confidence_decay=confidence_decay
        )
