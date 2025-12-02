"""Main FastAPI application for Sophia service."""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware

from sophia.api.models import (
    PlanRequest,
    PlanResponse,
    PlanStep,
    ImagineRequest,
    ImagineResponse,
    ImaginedState,
    ExecuteRequest,
    ExecuteResponse,
    ExecutionResult,
    HealthResponse,
    StateResponse,
    StateUpdateRequest,
    StateUpdateResponse,
    SimulateRequest,
    SimulateResponse,
    HermesProposalRequest,
    HermesProposalResponse,
)
from sophia.jepa.models import SimulationResult
from sophia.models.media_models import (
    MediaType,
    MediaIngestResponse,
    MediaSampleResponse,
    MediaSamplesListResponse,
    MediaSampleQuery,
)
from sophia.api.auth import verify_token
from sophia.planner import Planner
from sophia.executor import Executor
from sophia.knowledge_graph import KnowledgeGraph, Node, Edge
from sophia.hcg_client import HCGClient
from sophia.hcg_client.seeder import seed_pick_and_place_data
from sophia.cwm_g import ContinuousWorkingMemoryGenerative
from sophia.cwm_a import ContinuousWorkingMemoryAssociative
from sophia.jepa import JEPARunner
from sophia.jepa.models import (
    SimulationContext,
    Entity as JEPAEntity,
    SensorReference,
    TalosMetadata,
)
from sophia.storage import MediaStorageService
from sophia.ingestion import MediaIngestionService


logger = logging.getLogger(__name__)


def load_kg_from_hcg(hcg_client: HCGClient) -> KnowledgeGraph:
    """Load knowledge graph from Neo4j HCG.

    Args:
        hcg_client: HCG client instance

    Returns:
        KnowledgeGraph loaded from Neo4j
    """
    kg = KnowledgeGraph()

    # Query all nodes from Neo4j (simplified approach)
    # In a real implementation, we'd query Neo4j for all nodes
    # For now, we'll use the seeded pick-and-place data structure
    node_ids = [
        "table",
        "bin",
        "red_block",
        "blue_block",
        "move_to_red_block",
        "grasp_red_block",
        "move_to_bin",
        "release_red_block",
        "goal_red_block_in_bin",
    ]

    for node_id in node_ids:
        try:
            node_data = hcg_client.get_node(node_id)
            if node_data:
                node = Node(
                    id=node_data["id"],
                    type=node_data["type"],
                    properties=node_data.get("properties", {}),
                )
                kg.add_node(node)
        except Exception as e:
            logger.debug(f"Could not load node {node_id}: {e}")

    # Query edges (using Neo4j adapter's query methods)
    edge_queries = [
        ("red_block", "table", "located_at"),
        ("blue_block", "table", "located_at"),
        ("move_to_red_block", "grasp_red_block", "enables"),
        ("grasp_red_block", "move_to_bin", "enables"),
        ("move_to_bin", "release_red_block", "enables"),
        ("release_red_block", "bin", "achieves"),
        ("goal_red_block_in_bin", "release_red_block", "requires"),
    ]

    for source, target, relation in edge_queries:
        try:
            edges = hcg_client.query_edges_from(source)
            for edge in edges:
                if edge["target"] == target and edge["relation"] == relation:
                    kg.add_edge(
                        Edge(
                            source=source,
                            target=target,
                            relation=relation,
                            properties=edge.get("properties", {}),
                        )
                    )
        except Exception as e:
            logger.debug(f"Could not load edge {source}->{target}: {e}")

    return kg


# Global state
_planner: Optional[Planner] = None
_executor: Optional[Executor] = None
_hcg_client: Optional[HCGClient] = None
_cwm_g: Optional[ContinuousWorkingMemoryGenerative] = None
_cwm_a: Optional[ContinuousWorkingMemoryAssociative] = None
_kg: Optional[KnowledgeGraph] = None
_jepa_runner: Optional[JEPARunner] = None
_media_storage: Optional[MediaStorageService] = None
_media_ingestion: Optional[MediaIngestionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    global _planner, _executor, _hcg_client, _cwm_g, _cwm_a, _kg, _jepa_runner, _media_storage, _media_ingestion

    # Startup
    logger.info("Starting Sophia API service...")

    # Initialize knowledge graph
    _kg = KnowledgeGraph()

    # Initialize HCG client
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jtest")
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    try:
        milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
    except ValueError:
        logger.warning("Invalid MILVUS_PORT value; falling back to 19530")
        milvus_port = 19530

    try:
        _hcg_client = HCGClient(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_user,
            neo4j_password=neo4j_password,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )
        logger.info("HCG client initialized")

        # Seed pick-and-place data if enabled
        seed_data = os.getenv("SEED_PICK_AND_PLACE_DATA", "true").lower() == "true"
        if seed_data:
            logger.info("Seeding pick-and-place data into Neo4j...")
            try:
                # Clear existing data first (optional, controlled by env var)
                clear_before_seed = (
                    os.getenv("CLEAR_BEFORE_SEED", "false").lower() == "true"
                )
                if clear_before_seed:
                    _hcg_client.clear_all()
                    logger.info("Cleared existing HCG data")

                seed_pick_and_place_data(_hcg_client)
                logger.info("Pick-and-place data seeded successfully")
            except Exception as e:
                logger.warning(f"Failed to seed pick-and-place data: {e}")

        # Load knowledge graph from Neo4j
        logger.info("Loading knowledge graph from Neo4j HCG...")
        _kg = load_kg_from_hcg(_hcg_client)
        logger.info(
            f"Knowledge graph loaded: {len(_kg._nodes)} nodes, {len(_kg._edges)} edges"
        )

    except Exception as e:
        logger.warning(f"Failed to initialize HCG client: {e}")
        _hcg_client = None
        # Fallback to empty knowledge graph if HCG fails
        if _kg is None:
            _kg = KnowledgeGraph()

    # Initialize cognitive components
    _planner = Planner(knowledge_graph=_kg)
    _executor = Executor()
    _cwm_g = ContinuousWorkingMemoryGenerative()
    _cwm_a = ContinuousWorkingMemoryAssociative()
    _jepa_runner = JEPARunner(model_version="jepa-stub-v1.0")

    # Initialize media ingestion services
    storage_root = os.getenv("MEDIA_STORAGE_ROOT", "./media_storage")
    _media_storage = MediaStorageService(storage_root=storage_root)
    if _hcg_client:  # Type guard for mypy
        _media_ingestion = MediaIngestionService(
            hcg_client=_hcg_client,
            storage_service=_media_storage,
            jepa_runner=_jepa_runner,
        )
        logger.info(
            f"Media ingestion service initialized with storage root: {storage_root}"
        )

    logger.info("Sophia API service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Sophia API service...")
    if _hcg_client:
        _hcg_client.close()
    logger.info("Sophia API service shut down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Sophia API",
        description="Sophia cognitive service API with planning, imagination, and execution",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint (no auth required)
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        components = {}

        if _hcg_client:
            components.update(_hcg_client.health_check())
        else:
            components = {"neo4j": False, "milvus": False}

        overall_status = "healthy" if all(components.values()) else "degraded"

        return HealthResponse(
            status=overall_status,
            components=components,
            version="0.1.0",
        )

    # State endpoint (GET) - Read current state from Neo4j
    @app.get(
        "/state",
        response_model=StateResponse,
        dependencies=[Depends(verify_token)],
        tags=["state"],
    )
    async def get_state() -> StateResponse:
        """Get the current world state from Neo4j HCG.

        This endpoint reads the current state node from Neo4j.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            # Get current state from Neo4j
            state_node = _hcg_client.get_node("current_state")

            if not state_node:
                # Return empty state if not found
                return StateResponse(
                    state={},
                    state_id="current_state",
                )

            # Extract state from node properties
            state_data = state_node.get("properties", {})

            return StateResponse(
                state=state_data,
                state_id=state_node.get("id", "current_state"),
            )

        except Exception as e:
            logger.error(f"Error reading state: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read state: {str(e)}",
            )

    # State endpoint (POST) - Update state in Neo4j with SHACL validation
    @app.post(
        "/state",
        response_model=StateUpdateResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_token)],
        tags=["state"],
    )
    async def update_state(request: StateUpdateRequest) -> StateUpdateResponse:
        """Update the world state in Neo4j HCG with SHACL validation.

        This endpoint updates the current state node in Neo4j.
        SHACL validation is applied automatically by the HCG client.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            # Check if current_state node exists
            existing_state = _hcg_client.get_node("current_state")

            if existing_state:
                # Update existing state node
                # Delete and recreate with new properties (Neo4j pattern)
                _hcg_client.delete_node("current_state")

            # Create new state node with SHACL validation
            _hcg_client.add_node(
                node_id="current_state",
                node_type="state",
                properties=request.state,
            )

            # Also update in-memory planner state if available
            if _planner:
                _planner.update_state(request.state)

            return StateUpdateResponse(
                state_id="current_state",
                validation_passed=True,
            )

        except ValueError as e:
            # SHACL validation failed
            logger.error(f"State validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"State validation failed: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update state: {str(e)}",
            )

    # Plan endpoint
    @app.post(
        "/plan",
        response_model=PlanResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(verify_token)],
        tags=["planning"],
    )
    async def plan(request: PlanRequest) -> PlanResponse:
        """Generate a plan to achieve a goal.

        This endpoint uses backward chaining to decompose a goal into
        actionable steps based on the knowledge graph stored in Neo4j HCG.
        The generated plan is written back to Neo4j with SHACL validation.

        Requires authentication via Bearer token.
        """
        if not _planner:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Planner service not available",
            )

        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            goal_payload = request.goal_dict()

            # Read current state from Neo4j HCG
            state_node = _hcg_client.get_node("current_state")
            if state_node:
                current_state = state_node.get("properties", {})
                _planner.update_state(current_state)

            # Generate plan using backward chaining
            plan_steps = _planner.plan(goal_payload)

            # Convert to response format
            plan_step_models = [
                PlanStep(
                    id=step["id"],
                    name=step["name"],
                    type=step["type"],
                    action_type=step["action_type"],
                    target=step.get("target", ""),
                )
                for step in plan_steps
            ]

            plan_id = str(uuid.uuid4())

            # Write plan back to Neo4j HCG with SHACL validation
            _hcg_client.add_node(
                node_id=plan_id,
                node_type="plan",
                properties={
                    "goal": goal_payload,
                    "steps": [
                        {
                            "id": step.id,
                            "name": step.name,
                            "action_type": step.action_type,
                            "target": step.target,
                        }
                        for step in plan_step_models
                    ],
                },
            )

            # Link plan to goal if it exists in HCG
            goal_id = goal_payload.get("target_state", "")
            goal_node_id = f"goal_{goal_id}" if goal_id else None
            if goal_node_id:
                try:
                    goal_node = _hcg_client.get_node(goal_node_id)
                    if goal_node:
                        _hcg_client.add_edge(
                            edge_id=f"e_{plan_id}_achieves_goal",
                            source_id=plan_id,
                            target_id=goal_node_id,
                            relation="achieves",
                        )
                except Exception as e:
                    logger.warning(f"Could not link plan to goal: {e}")

            return PlanResponse(
                plan=plan_step_models,
                goal=goal_payload,
                plan_id=plan_id,
            )

        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate plan: {str(e)}",
            )

    # Imagine endpoint
    @app.post(
        "/imagine",
        response_model=ImagineResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(verify_token)],
        tags=["imagination"],
    )
    async def imagine(request: ImagineRequest) -> ImagineResponse:
        """Generate imagined future states based on CWM-G imagery and CWM-E emotion tags.

        This endpoint processes generative memory content and emotion tags to
        produce imagined states, storing them in Neo4j with metadata.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            imagination_id = str(uuid.uuid4())

            # Process CWM-G imagery if provided
            if request.cwm_g_imagery and _cwm_g:
                for imagery_item in request.cwm_g_imagery:
                    _cwm_g.add(imagery_item)

            # Generate imagined states
            # For now, we create placeholder states based on the input
            # In a full implementation, this would use ML models or other reasoning
            imagined_states: List[ImaginedState] = []

            num_states = request.horizon
            for i in range(num_states):
                state_id = f"{imagination_id}_state_{i}"
                state = ImaginedState(
                    state_id=state_id,
                    description=f"Imagined state {i + 1} based on provided context",
                    confidence=0.8 - (i * 0.1),  # Decreasing confidence
                    properties={
                        "horizon_step": i,
                        "emotion_tags": request.cwm_e_emotion_tags or [],
                        "context": request.context or {},
                    },
                )
                imagined_states.append(state)

                # Store in Neo4j with metadata
                node_id = state_id
                _hcg_client.add_node(
                    node_id=node_id,
                    node_type="imagined_state",
                    properties={
                        "description": state.description,
                        "confidence": state.confidence,
                        "model_version": request.model_version,
                        "horizon": request.horizon,
                        "horizon_step": i,
                        "assumptions": request.assumptions or [],
                        "imagination_id": imagination_id,
                    },
                )

            return ImagineResponse(
                imagined_states=imagined_states,
                imagination_id=imagination_id,
                model_version=request.model_version,
                horizon=request.horizon,
                assumptions=request.assumptions or [],
            )

        except Exception as e:
            logger.error(f"Error generating imagined states: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate imagined states: {str(e)}",
            )

    # Simulate endpoint
    @app.post(
        "/simulate",
        response_model=SimulateResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(verify_token)],
        tags=["simulation"],
    )
    async def simulate(request: SimulateRequest) -> SimulateResponse:
        """Perform JEPA-based k-step simulation with dynamics rollout.

        This endpoint uses the JEPA runner to perform forward prediction
        of system dynamics over k steps, creating imagined processes and
        states with confidence scores. Results are persisted to Neo4j.

        Requires authentication via Bearer token.
        """
        if not _jepa_runner:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="JEPA runner not available",
            )

        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            # Check if media_sample_id is provided and retrieve embeddings
            media_embeddings: List[str] = []
            if request.media_sample_id:
                if not _media_ingestion:
                    logger.warning(
                        f"Media sample {request.media_sample_id} referenced but media ingestion service not available"
                    )
                else:
                    # Verify media sample exists
                    media_sample = _media_ingestion.get_media_sample(
                        request.media_sample_id
                    )
                    if not media_sample:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Media sample {request.media_sample_id} not found",
                        )

                    # Retrieve embedding IDs for this sample from Neo4j
                    # (embeddings themselves are in Milvus, we just track the IDs)
                    try:
                        # Query Neo4j for embedding nodes linked to this sample
                        with _hcg_client._neo4j._driver.session(
                            database=_hcg_client._neo4j._database
                        ) as session:  # type: ignore
                            neo4j_result = session.run(
                                """
                                MATCH (m {sample_id: $sample_id})-[:has_embedding]->(e)
                                RETURN e.id as embedding_id
                                """,
                                {"sample_id": request.media_sample_id},
                            )
                            media_embeddings = [
                                record["embedding_id"] for record in neo4j_result
                            ]

                        logger.info(
                            f"Found {len(media_embeddings)} embeddings for media sample {request.media_sample_id}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to retrieve embeddings for media sample: {e}"
                        )

            # Build simulation context from request
            entities = [JEPAEntity(**entity_data) for entity_data in request.entities]

            sensor_refs = [
                SensorReference(**sensor_data) for sensor_data in request.sensor_refs
            ]

            talos_metadata = TalosMetadata(**(request.talos_metadata or {}))

            context = SimulationContext(
                entities=entities,
                sensor_refs=sensor_refs,
                talos_metadata=talos_metadata,
                initial_state=request.initial_state,
                actions=request.actions,
            )

            # Run JEPA simulation
            result: SimulationResult = _jepa_runner.simulate(
                context=context,
                k_steps=request.k_steps,
                assumptions=request.assumptions,
            )

            # Store imagined processes in Neo4j
            for process in result.imagined_processes:
                _hcg_client.add_node(
                    node_id=process.process_id,
                    node_type="imagined_process",
                    properties={
                        "description": process.description,
                        "confidence": process.confidence,
                        "model_version": process.model_version,
                        "horizon": process.horizon,
                        "assumptions": process.assumptions,
                        "imagined": True,
                        "simulation_id": result.simulation_id,
                        **process.properties,
                    },
                )

            # Store imagined states in Neo4j
            for state in result.imagined_states:
                _hcg_client.add_node(
                    node_id=state.state_id,
                    node_type="imagined_state",
                    properties={
                        "step": state.step,
                        "description": state.description,
                        "confidence": state.confidence,
                        "model_version": state.model_version,
                        "horizon": state.horizon,
                        "assumptions": state.assumptions,
                        "imagined": True,
                        "simulation_id": result.simulation_id,
                        "state_data": state.state_data,
                    },
                )

                # Link state to simulation
                _hcg_client.add_edge(
                    edge_id=f"e_{result.simulation_id}_{state.state_id}",
                    source_id=result.simulation_id,
                    target_id=state.state_id,
                    relation="produces",
                )

            # Store simulation metadata node
            simulation_properties = {
                "k_steps": result.k_steps,
                "model_version": result.model_version,
                "overall_confidence": result.overall_confidence,
                "entity_count": len(context.entities),
                "sensor_count": len(context.sensor_refs),
                "talos_metadata": context.talos_metadata.model_dump(),
            }

            # Add media reference if provided
            if request.media_sample_id:
                simulation_properties["media_sample_id"] = request.media_sample_id

            _hcg_client.add_node(
                node_id=result.simulation_id,
                node_type="simulation",
                properties=simulation_properties,
            )

            # Link simulation to media sample if provided
            if request.media_sample_id:
                try:
                    _hcg_client.add_edge(
                        edge_id=f"e_{result.simulation_id}_uses_{request.media_sample_id}",
                        source_id=result.simulation_id,
                        target_id=request.media_sample_id,
                        relation="uses_media",
                        properties={"embedding_count": len(media_embeddings)},
                    )
                    logger.info(
                        f"Linked simulation {result.simulation_id} to media sample {request.media_sample_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create simulation->media edge: {e}")

            # Convert result to response format
            response = SimulateResponse(
                simulation_id=result.simulation_id,
                imagined_processes=[
                    process.model_dump() for process in result.imagined_processes
                ],
                imagined_states=[
                    state.model_dump() for state in result.imagined_states
                ],
                k_steps=result.k_steps,
                model_version=result.model_version,
                overall_confidence=result.overall_confidence,
                media_sample_id=request.media_sample_id,
                media_embeddings=media_embeddings if media_embeddings else None,
            )

            return response

        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to run simulation: {str(e)}",
            )

    # Hermes proposal ingestion endpoint
    @app.post(
        "/ingest/hermes_proposal",
        response_model=HermesProposalResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["ingestion"],
    )
    async def ingest_hermes_proposal(
        request: HermesProposalRequest,
    ) -> HermesProposalResponse:
        """Ingest an LLM proposal from Hermes with provenance tracking.

        This endpoint accepts structured proposals from Hermes (including plans,
        imagined states, diagnostics, and tool calls) and persists them to Neo4j
        with full provenance metadata. SHACL validation is applied automatically.

        Note: Authentication is disabled for local development.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            stored_node_ids = []

            # Store the main proposal node with provenance
            proposal_properties = {
                "source_service": request.source_service,
                "llm_provider": request.llm_provider,
                "model": request.model,
                "generated_at": request.generated_at,
                "confidence": request.confidence,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            if request.raw_text:
                proposal_properties["raw_text"] = request.raw_text
            if request.diagnostics:
                proposal_properties["diagnostics"] = request.diagnostics
            if request.metadata:
                proposal_properties.update(request.metadata)

            _hcg_client.add_node(
                node_id=request.proposal_id,
                node_type="hermes_proposal",
                properties=proposal_properties,
            )
            stored_node_ids.append(request.proposal_id)

            # Store plan steps if provided
            if request.plan_steps:
                for idx, step in enumerate(request.plan_steps):
                    step_id = f"{request.proposal_id}_plan_step_{idx}"
                    _hcg_client.add_node(
                        node_id=step_id,
                        node_type="proposed_plan_step",
                        properties={
                            "source_proposal": request.proposal_id,
                            "step_index": idx,
                            **step,
                        },
                    )
                    stored_node_ids.append(step_id)

                    # Create edge from proposal to plan step
                    _hcg_client.add_edge(
                        edge_id=f"e_{request.proposal_id}_{step_id}",
                        source_id=request.proposal_id,
                        target_id=step_id,
                        relation="contains_plan_step",
                    )

            # Store imagined states if provided
            if request.imagined_states:
                for idx, state in enumerate(request.imagined_states):
                    state_id = f"{request.proposal_id}_imagined_state_{idx}"
                    _hcg_client.add_node(
                        node_id=state_id,
                        node_type="proposed_imagined_state",
                        properties={
                            "source_proposal": request.proposal_id,
                            "state_index": idx,
                            **state,
                        },
                    )
                    stored_node_ids.append(state_id)

                    # Create edge from proposal to imagined state
                    _hcg_client.add_edge(
                        edge_id=f"e_{request.proposal_id}_{state_id}",
                        source_id=request.proposal_id,
                        target_id=state_id,
                        relation="contains_imagined_state",
                    )

            # Store tool calls if provided
            if request.tool_calls:
                for idx, tool_call in enumerate(request.tool_calls):
                    tool_id = f"{request.proposal_id}_tool_call_{idx}"
                    _hcg_client.add_node(
                        node_id=tool_id,
                        node_type="proposed_tool_call",
                        properties={
                            "source_proposal": request.proposal_id,
                            "call_index": idx,
                            **tool_call,
                        },
                    )
                    stored_node_ids.append(tool_id)

                    # Create edge from proposal to tool call
                    _hcg_client.add_edge(
                        edge_id=f"e_{request.proposal_id}_{tool_id}",
                        source_id=request.proposal_id,
                        target_id=tool_id,
                        relation="contains_tool_call",
                    )

            return HermesProposalResponse(
                proposal_id=request.proposal_id,
                stored_node_ids=stored_node_ids,
                status="accepted",
            )

        except ValueError as e:
            # SHACL validation failed
            logger.error(f"Proposal validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Proposal validation failed: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error ingesting proposal: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to ingest proposal: {str(e)}",
            )

    # Execute endpoint
    @app.post(
        "/execute",
        response_model=ExecuteResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(verify_token)],
        tags=["execution"],
    )
    async def execute(request: ExecuteRequest) -> ExecuteResponse:
        """Execute a plan or a specific step from a plan.

        This endpoint simulates or executes planned actions, tracking state changes.

        Requires authentication via Bearer token.
        """
        if not _executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Executor service not available",
            )

        try:
            execution_id = str(uuid.uuid4())
            results: List[ExecutionResult] = []

            # For now, we simulate execution
            # In a full implementation, this would actually execute actions
            # and update the knowledge graph state

            # Create a mock execution result
            mock_step = PlanStep(
                id="mock_action_1",
                name="Mock Action",
                type="action",
                action_type="SIMULATE",
                target="",
            )

            result = ExecutionResult(
                step=mock_step,
                status="success" if not request.dry_run else "simulated",
                message=(
                    "Action executed successfully"
                    if not request.dry_run
                    else "Dry run - no actual execution"
                ),
                state_changes=(
                    {"mock_state": "updated"} if not request.dry_run else {}
                ),
            )
            results.append(result)

            overall_status = (
                "success" if all(r.status == "success" for r in results) else "partial"
            )

            return ExecuteResponse(
                plan_id=request.plan_id,
                results=results,
                overall_status=overall_status,
                execution_id=execution_id,
            )

        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute plan: {str(e)}",
            )

    @app.post(
        "/ingest/media",
        response_model=MediaIngestResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(verify_token)],
    )
    async def ingest_media(
        file: UploadFile = File(...),
        media_type: MediaType = Form(...),
        question: Optional[str] = Form(None),
    ) -> MediaIngestResponse:
        """
        Ingest a media file (image, video, or audio) for perception workflows.

        The file is validated, stored to disk, and metadata is extracted and
        persisted to Neo4j. Returns details about the ingested media sample.

        Requires authentication via Bearer token.
        """
        if not _media_ingestion:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Media ingestion service not available",
            )

        try:
            result = await _media_ingestion.ingest_media(
                file=file,
                media_type=media_type,
                question=question,
            )
            return result

        except ValueError as e:
            logger.error(f"Media validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Error ingesting media: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to ingest media: {str(e)}",
            )

    @app.get(
        "/media/samples",
        response_model=MediaSamplesListResponse,
        dependencies=[Depends(verify_token)],
    )
    async def list_media_samples(
        media_type: Optional[MediaType] = Query(None),
        after_timestamp: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> MediaSamplesListResponse:
        """
        List media samples with optional filtering and pagination.

        Supports filtering by media type and timestamp, with pagination
        controls for large result sets.

        Requires authentication via Bearer token.
        """
        if not _media_ingestion:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Media ingestion service not available",
            )

        try:
            # Parse timestamp string to datetime if provided
            parsed_timestamp = None
            if after_timestamp:
                from datetime import datetime

                parsed_timestamp = datetime.fromisoformat(after_timestamp)

            query = MediaSampleQuery(
                media_type=media_type,
                after_timestamp=parsed_timestamp,
                limit=limit,
                offset=offset,
            )
            result = _media_ingestion.list_media_samples(query)
            return result

        except Exception as e:
            logger.error(f"Error listing media samples: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list media samples: {str(e)}",
            )

    @app.get(
        "/media/samples/{sample_id}",
        response_model=MediaSampleResponse,
        dependencies=[Depends(verify_token)],
    )
    async def get_media_sample(
        sample_id: str,
    ) -> MediaSampleResponse:
        """
        Retrieve a specific media sample by ID.

        Returns the sample metadata and usage count (number of simulations
        that have referenced this sample).

        Requires authentication via Bearer token.
        """
        if not _media_ingestion:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Media ingestion service not available",
            )

        try:
            result = _media_ingestion.get_media_sample(sample_id)
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Media sample not found: {sample_id}",
                )
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving media sample: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve media sample: {str(e)}",
            )

    return app


# Create the application instance
app = create_app()
