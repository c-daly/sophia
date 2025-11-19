"""Main FastAPI application for Sophia service."""

import os
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
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
)
from sophia.api.auth import verify_token
from sophia.planner import Planner
from sophia.executor import Executor
from sophia.knowledge_graph import KnowledgeGraph
from sophia.hcg_client import HCGClient
from sophia.cwm_g import ContinuousWorkingMemoryGenerative
from sophia.cwm_a import ContinuousWorkingMemoryAssociative


logger = logging.getLogger(__name__)


# Global state
_planner: Optional[Planner] = None
_executor: Optional[Executor] = None
_hcg_client: Optional[HCGClient] = None
_cwm_g: Optional[ContinuousWorkingMemoryGenerative] = None
_cwm_a: Optional[ContinuousWorkingMemoryAssociative] = None
_kg: Optional[KnowledgeGraph] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    global _planner, _executor, _hcg_client, _cwm_g, _cwm_a, _kg

    # Startup
    logger.info("Starting Sophia API service...")

    # Initialize knowledge graph
    _kg = KnowledgeGraph()

    # Initialize HCG client
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "sophiadev")
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    try:
        _hcg_client = HCGClient(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_user,
            neo4j_password=neo4j_password,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )
        logger.info("HCG client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize HCG client: {e}")
        _hcg_client = None

    # Initialize cognitive components
    _planner = Planner(knowledge_graph=_kg)
    _executor = Executor()
    _cwm_g = ContinuousWorkingMemoryGenerative()
    _cwm_a = ContinuousWorkingMemoryAssociative()

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
        actionable steps based on the knowledge graph.

        Requires authentication via Bearer token.
        """
        if not _planner:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Planner service not available",
            )

        try:
            # Generate plan
            plan_steps = _planner.plan(request.goal)

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

            return PlanResponse(
                plan=plan_step_models,
                goal=request.goal,
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
                    description=f"Imagined state {i+1} based on provided context",
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

    return app


# Create the application instance
app = create_app()
