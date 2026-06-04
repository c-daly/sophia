"""Main FastAPI application for Sophia service."""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional

from dotenv import load_dotenv

# Load .env file before any pydantic-settings models are instantiated
load_dotenv()

from logos_observability import setup_telemetry
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import StatusCode, get_current_span


from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from logos_config import Neo4jConfig, MilvusConfig, RedisConfig, get_env_value
from logos_events import EventBus
from logos_config.health import HealthResponse as LogosHealthResponse, DependencyStatus
from logos_test_utils import setup_logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceQueue
from sophia.maintenance.scheduler import MaintenanceScheduler

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
    StateResponse,
    StateUpdateRequest,
    StateUpdateResponse,
    SimulateRequest,
    SimulateResponse,
    HermesProposalRequest,
    HermesProposalResponse,
    CWMStateResponse,
    CWMStateListResponse,
    HCGEntityResponse,
    HCGEdgeResponse,
    HCGGraphSnapshotResponse,
    # Persona models
    PersonaEntryCreate,
    PersonaEntryResponse,
    PersonaEntryFull,
    PersonaEntryUpdate,
    PersonaListResponse,
    SentimentResponse,
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
from sophia.cwm_g import ContinuousWorkingMemoryGenerative
from sophia.cwm_a import ContinuousWorkingMemoryAssociative, CWMAStateService
from sophia.jepa import JEPARunner
from sophia.jepa.models import (
    SimulationContext,
    Entity as JEPAEntity,
    SensorReference,
    TalosMetadata,
)
from sophia.storage import MediaStorageService
from sophia.ingestion import MediaIngestionService
from sophia.ingestion.proposal_processor import (
    ALL_MILVUS_COLLECTIONS,
    ProposalProcessor,
)
from sophia.cwm import CWMPersistence
from sophia.feedback import (
    FeedbackConfig,
    FeedbackDispatcher,
    FeedbackPayload,
    FeedbackQueue,
    FeedbackWorker,
    StepResult,
)

# Configure structured logging for sophia
logger = setup_logging("sophia")

# Max uuids per Milvus `uuid in [...]` query in the snapshot embedding lookup.
# Batching keeps the filter expression well under Milvus's expression-length cap
# (a single unbatched list of 10k 36-char uuids is ~400 KB and gets rejected).
_SNAPSHOT_EMB_BATCH = 512


def _run_full_type_emergence_scan(
    hcg_client: Any, run_one: Callable[[str], None]
) -> None:
    """Run emergence over every type definition, isolating per-type failures.

    A transient HCG/Milvus error on one type must not abort the rest of the
    periodic full scan, and individual failures must be surfaced rather than
    silently dropped (greptile #149). A failure listing the type definitions
    aborts the scan (there is nothing to iterate).
    """
    try:
        all_types = hcg_client.get_all_type_definitions()
    except Exception:
        logger.exception("Full type emergence scan: listing type definitions failed")
        return
    for td in all_types:
        type_uuid = td.get("uuid", "")
        if not type_uuid:
            continue
        try:
            run_one(type_uuid)
        except Exception:
            logger.exception("type_emergence failed during full scan for %r", type_uuid)


def sanitize_neo4j_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """Convert neo4j types to JSON-serializable Python types.

    Neo4j returns custom types like neo4j.time.DateTime that Pydantic
    cannot serialize. This function recursively converts them.
    """
    if not props:
        return props

    result = {}
    for key, value in props.items():
        if hasattr(value, "isoformat"):
            # neo4j.time.DateTime, datetime, date, time
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = sanitize_neo4j_properties(value)
        elif isinstance(value, list):
            result[key] = [
                (
                    sanitize_neo4j_properties(item)
                    if isinstance(item, dict)
                    else (item.isoformat() if hasattr(item, "isoformat") else item)
                )
                for item in value
            ]
        else:
            result[key] = value
    return result


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests for tracing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Store request_id in request state for access in handlers
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


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
                    uuid=node_data["uuid"],
                    name=node_data["name"],
                    type=node_data["type"],
                    properties=node_data.get("properties", {}),
                )
                kg.add_node(node)
        except Exception as e:
            logger.debug(f"Could not load node {node_id}: {e}")

    # Query edges (using Neo4j adapter's query methods)
    edge_queries = [
        ("red_block", "table", "LOCATED_AT"),
        ("blue_block", "table", "LOCATED_AT"),
        ("move_to_red_block", "grasp_red_block", "ENABLES"),
        ("grasp_red_block", "move_to_bin", "ENABLES"),
        ("move_to_bin", "release_red_block", "ENABLES"),
        ("release_red_block", "bin", "ACHIEVES"),
        ("goal_red_block_in_bin", "release_red_block", "REQUIRES"),
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
_cwm_a_state: Optional[CWMAStateService] = None
_kg: Optional[KnowledgeGraph] = None
_jepa_runner: Optional[JEPARunner] = None
_media_storage: Optional[MediaStorageService] = None
_media_ingestion: Optional[MediaIngestionService] = None
_cwm_persistence: Optional[CWMPersistence] = None
_feedback_dispatcher: Optional[FeedbackDispatcher] = None
_feedback_worker: Optional[FeedbackWorker] = None
_feedback_worker_task: Optional[Any] = None
_proposal_processor: Optional[ProposalProcessor] = None
_proposal_worker: Optional[Any] = None
_proposal_worker_task: Optional[Any] = None
_maintenance_scheduler: Optional[MaintenanceScheduler] = None
_maintenance_task: Optional[Any] = None
_maint_redis: Optional[Any] = None
_maint_event_bus: Optional[Any] = None
_event_bus: Optional[Any] = None
_redis_direct: Optional[Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    global _planner, _executor, _hcg_client, _cwm_g, _cwm_a, _cwm_a_state, _kg
    global _jepa_runner, _media_storage, _media_ingestion, _cwm_persistence
    global _feedback_dispatcher, _feedback_worker, _feedback_worker_task
    global _proposal_processor, _proposal_worker, _proposal_worker_task
    global _maintenance_scheduler, _maintenance_task, _maint_redis, _maint_event_bus
    global _event_bus, _redis_direct

    # Startup
    logger.info("Starting Sophia API service...")

    # Initialize OpenTelemetry
    otlp_endpoint = get_env_value("OTEL_EXPORTER_OTLP_ENDPOINT")
    setup_telemetry(
        service_name=get_env_value("OTEL_SERVICE_NAME", default="sophia") or "sophia",
        export_to_console=(
            get_env_value("OTEL_CONSOLE_EXPORT", default="false") or "false"
        ).lower()
        == "true",
        otlp_endpoint=otlp_endpoint,
    )
    logger.info(
        "OpenTelemetry initialized",
        extra={"otlp_endpoint": otlp_endpoint or "none"},
    )

    # Initialize feedback system (non-critical, graceful degradation)
    feedback_config = FeedbackConfig()
    if feedback_config.enabled:
        try:
            from redis.exceptions import ConnectionError as RedisConnectionError

            feedback_queue = FeedbackQueue(feedback_config.redis)
            # Test connection
            feedback_queue.pending_count()
            _feedback_dispatcher = FeedbackDispatcher(feedback_queue, enabled=True)
            _feedback_worker = FeedbackWorker(
                queue=feedback_queue,
                hermes_url=feedback_config.hermes_url,
                timeout=feedback_config.worker_timeout,
            )
            # Start worker as background task
            _feedback_worker_task = asyncio.create_task(_feedback_worker.start())
            logger.info("Feedback emission system initialized")
        except (RedisConnectionError, Exception) as e:
            logger.warning(f"Redis unavailable, feedback disabled: {e}")
            _feedback_dispatcher = FeedbackDispatcher(None, enabled=False)
    else:
        logger.info("Feedback emission disabled by configuration")
        _feedback_dispatcher = FeedbackDispatcher(None, enabled=False)

    # Initialize knowledge graph
    _kg = KnowledgeGraph()

    # Initialize HCG client - let env vars (NEO4J_*, MILVUS_*) take precedence
    # In containers, these are set via docker-compose environment section
    neo4j_config = Neo4jConfig()  # type: ignore[call-arg]  # password provided via NEO4J_PASSWORD env var
    milvus_config = MilvusConfig()

    try:
        _hcg_client = HCGClient(
            neo4j_uri=neo4j_config.uri,
            neo4j_username=neo4j_config.user,
            neo4j_password=neo4j_config.password,
            milvus_host=milvus_config.host,
            milvus_port=milvus_config.port,
        )
        logger.info("HCG client initialized")

        # Load knowledge graph from Neo4j
        logger.info("Loading knowledge graph from Neo4j HCG...")
        _kg = load_kg_from_hcg(_hcg_client)
        logger.info(
            f"Knowledge graph loaded: {len(_kg._nodes)} nodes, {len(_kg._edges)} edges"
        )

        # Seed pick-and-place data if enabled
        if (
            get_env_value("SEED_PICK_AND_PLACE_DATA", default="") or ""
        ).lower() == "true":
            from sophia.hcg_client.seeder import (
                seed_pick_and_place_data,
                seed_plan_data,
                seed_persona_entries,
            )

            logger.info("Seeding test data into Neo4j...")
            try:
                seed_pick_and_place_data(_hcg_client)
                seed_plan_data(_hcg_client)
                seed_persona_entries(_hcg_client)
                # Reload knowledge graph after seeding
                _kg = load_kg_from_hcg(_hcg_client)
                logger.info(
                    f"Knowledge graph reloaded after seeding: {len(_kg._nodes)} nodes"
                )
            except Exception as e:
                logger.warning(f"Failed to seed pick-and-place data: {e}")

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
    _cwm_a_state = CWMAStateService(source="sophia_api")
    _jepa_runner = JEPARunner(model_version="jepa-stub-v1.0")

    # Initialize CWM persistence (requires HCG client)
    if _hcg_client:
        _cwm_persistence = CWMPersistence(
            neo4j_driver=_hcg_client.driver,
            database=_hcg_client.database,
        )
        logger.info("CWM persistence service initialized")

    # Initialize media ingestion services
    storage_root = (
        get_env_value("MEDIA_STORAGE_ROOT", default="./media_storage")
        or "./media_storage"
    )
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

    # Initialize ProposalProcessor (requires HCG client + Milvus)
    _milvus_sync = None
    if _hcg_client:
        try:
            from logos_hcg.sync import HCGMilvusSync

            _milvus_sync = HCGMilvusSync(
                milvus_host=milvus_config.host,
                milvus_port=str(milvus_config.port),
            )
            _milvus_sync.connect()
            # Ensure HCG collections exist for proposal processing. Post-logos#542
            # ensure_collection() requires the embedding dim: pre-create at an
            # explicit LOGOS_EMBEDDING_DIM when set, otherwise defer to lazy
            # creation at the *measured* dim on first write (HCGMilvusSync
            # self-corrects), mirroring infra/init_milvus_collections.py.
            from logos_config import get_embedding_dim_override

            _dim_override = get_embedding_dim_override()
            if _dim_override is not None:
                for _nt in ALL_MILVUS_COLLECTIONS:
                    _milvus_sync.ensure_collection(_nt, _dim_override)
            else:
                logger.info(
                    "LOGOS_EMBEDDING_DIM unset — HCG embedding collections will be "
                    "created lazily at the measured dimension on first write "
                    "(logos#542)."
                )
            # Initialize EventBus for pub/sub
            try:
                import redis

                _redis_config = RedisConfig()
                _event_bus = EventBus(_redis_config)
                try:
                    _redis_direct = redis.from_url(_redis_config.url)
                except Exception:
                    _event_bus.close()
                    raise
                logger.info("EventBus initialized for pub/sub")
            except Exception as e:
                logger.warning(f"EventBus unavailable: {e}")
                _event_bus = None
                _redis_direct = None

            _proposal_processor = ProposalProcessor(
                hcg_client=_hcg_client,
                milvus_sync=_milvus_sync,
                event_bus=_event_bus,
                redis_client=_redis_direct,
            )
            logger.info("ProposalProcessor initialized")
        except Exception as e:
            logger.warning(f"ProposalProcessor unavailable (Milvus not ready): {e}")
            # Processor not available without Milvus — endpoint will skip cognitive processing
            _proposal_processor = None

    # Initialize proposal processing worker (requires feedback + processor)
    _proposal_worker = None
    _proposal_worker_task = None
    if feedback_config.enabled and _proposal_processor:
        try:
            from sophia.feedback.proposal_queue import ProposalQueue
            from sophia.feedback.proposal_worker import ProposalWorker

            proposal_queue = ProposalQueue(feedback_config.redis)
            proposal_queue.pending_count()  # Test connection
            _proposal_worker = ProposalWorker(
                queue=proposal_queue,
                processor=_proposal_processor,
            )
            _proposal_worker_task = asyncio.create_task(_proposal_worker.start())
            logger.info("Proposal processing worker initialized")
        except Exception as e:
            logger.warning(f"Proposal worker unavailable: {e}")

    # Initialize Maintenance Scheduler
    _maintenance_scheduler = None
    _maintenance_task = None
    try:
        _maintenance_config = MaintenanceConfig()
        if _maintenance_config.enabled:
            import redis

            _maint_redis_config = RedisConfig()
            _maint_event_bus = EventBus(_maint_redis_config)
            _maint_redis = redis.from_url(_maint_redis_config.url)
            _maint_queue = MaintenanceQueue(_maint_redis)

            # Register available handlers — adapters bridge scheduler params
            # to the signatures expected by the underlying detectors.
            from sophia.maintenance.emergence_handler import build_emergence_handler
            from sophia.ingestion.relationship_discoverer import RelationshipDiscoverer

            _handlers: dict = {}
            if _milvus_sync and _hcg_client:
                _emergence_run = build_emergence_handler(
                    config=_maintenance_config,
                    hcg=_hcg_client,
                    milvus=_milvus_sync,
                    event_bus=_event_bus,
                    hermes_url=feedback_config.hermes_url,
                    token=get_env_value("SOPHIA_API_TOKEN") or "",
                )
                _relationship_discoverer = RelationshipDiscoverer(
                    milvus=_milvus_sync, hcg=_hcg_client
                )

                def _handle_type_emergence(
                    type_uuid: str = "", scan: str = "", **kwargs: object
                ) -> None:
                    """Adapter: scheduler params -> TypeEmergenceDetector.check_type.

                    The scheduler enqueues jobs with ``type_uuid`` (from event
                    payloads) or ``scan='full'`` (periodic). The underlying
                    ``check_type`` accepts a type UUID directly.
                    """
                    assert _hcg_client is not None  # guarded by outer if
                    if scan == "full":
                        # Full scan: check every type definition, isolating
                        # per-type failures so one bad type can't abort the
                        # rest of the scan (greptile #149).
                        _run_full_type_emergence_scan(_hcg_client, _emergence_run)
                        return
                    if not type_uuid:
                        logger.warning("type_emergence job missing type_uuid param")
                        return
                    try:
                        _emergence_run(type_uuid)
                    except Exception:
                        logger.exception("type_emergence failed for %r", type_uuid)

                def _handle_relationship_discovery(
                    node_uuids: list | None = None, **kwargs: object
                ) -> None:
                    """Adapter: scheduler params -> RelationshipDiscoverer.find_candidates.

                    The underlying ``find_candidates`` requires per-node embedding
                    and type info. Embedding lookup infrastructure (e.g.
                    ``HCGMilvusSync.get_node_embedding``) does not exist yet, so
                    this adapter logs and skips until that API is available.
                    """
                    if not node_uuids:
                        logger.warning("relationship_discovery job missing node_uuids")
                        return
                    # TODO: Implement when HCGMilvusSync exposes a
                    # get_node_embedding(uuid) -> {embedding, node_type} method.
                    logger.info(
                        "relationship_discovery: skipping %d node(s) — "
                        "per-node embedding lookup not yet available",
                        len(node_uuids),
                    )

                from sophia.maintenance.type_rollup_handler import (
                    build_type_rollup_handler,
                )

                _rollup_run = build_type_rollup_handler(
                    config=_maintenance_config,
                    hcg=_hcg_client,
                    milvus=_milvus_sync,
                    event_bus=_event_bus,
                    hermes_url=feedback_config.hermes_url,
                    token=get_env_value("SOPHIA_API_TOKEN") or "",
                )

                _handlers = {
                    "type_emergence": _handle_type_emergence,
                    "relationship_discovery": _handle_relationship_discovery,
                    "type_rollup": _rollup_run,
                }

            if not _handlers:
                logger.warning(
                    "Maintenance scheduler started with no handlers "
                    "(Milvus/HCG unavailable?); all jobs will be skipped"
                )

            _maintenance_scheduler = MaintenanceScheduler(
                queue=_maint_queue,
                event_bus=_maint_event_bus,
                config=_maintenance_config,
                handlers=_handlers,
                hcg_client=_hcg_client,
            )
            _maintenance_task = asyncio.create_task(_maintenance_scheduler.start())
            logger.info("Maintenance scheduler started")
    except Exception:
        logger.exception("Failed to start maintenance scheduler")

    logger.info("Sophia API service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Sophia API service...")

    # Stop proposal worker
    if _proposal_worker:
        _proposal_worker.stop()
    if _proposal_worker_task:
        _proposal_worker_task.cancel()
        try:
            await _proposal_worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Proposal worker stopped")

    # Stop feedback worker
    if _feedback_worker:
        _feedback_worker.stop()
    if _feedback_worker_task:
        _feedback_worker_task.cancel()
        try:
            await _feedback_worker_task
        except asyncio.CancelledError:
            pass
        logger.info("Feedback worker stopped")

    # Stop Maintenance Scheduler
    if _maintenance_scheduler is not None:
        await _maintenance_scheduler.stop()
    if _maintenance_task is not None:
        _maintenance_task.cancel()
        try:
            await _maintenance_task
        except asyncio.CancelledError:
            pass
        logger.info("Maintenance scheduler stopped")
    if _maint_event_bus is not None:
        # safe after scheduler.stop() — stop() halts the listen loop,
        # close() releases the underlying Redis connection.
        _maint_event_bus.close()
    if _maint_redis is not None:
        _maint_redis.close()
    if _event_bus is not None:
        try:
            _event_bus.close()
            logger.info("EventBus closed")
        except Exception:
            logger.exception("Error closing EventBus")
    if _redis_direct is not None:
        try:
            _redis_direct.close()
            logger.info("Redis direct connection closed")
        except Exception:
            logger.exception("Error closing Redis direct connection")

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

    FastAPIInstrumentor.instrument_app(app)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(
            get_env_value(
                "CORS_ORIGINS", default="http://localhost:3000,http://localhost:3001"
            )
            or ""
        ).split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware for tracing
    app.add_middleware(RequestIDMiddleware)

    # Health check endpoint (no auth required)
    @app.get("/health", response_model=LogosHealthResponse, tags=["health"])
    async def health_check() -> LogosHealthResponse:
        """Health check endpoint using standardized logos_config schema."""
        dependencies = {}

        if _hcg_client:
            hcg_health = _hcg_client.health_check()
            dependencies["neo4j"] = DependencyStatus(
                status="healthy" if hcg_health.get("neo4j") else "unavailable",
                connected=hcg_health.get("neo4j", False),
            )
            dependencies["milvus"] = DependencyStatus(
                status="healthy" if hcg_health.get("milvus") else "unavailable",
                connected=hcg_health.get("milvus", False),
            )
        else:
            dependencies["neo4j"] = DependencyStatus(
                status="unavailable", connected=False
            )
            dependencies["milvus"] = DependencyStatus(
                status="unavailable", connected=False
            )

        all_healthy = all(d.status == "healthy" for d in dependencies.values())
        overall_status: Literal["healthy", "degraded"] = (
            "healthy" if all_healthy else "degraded"
        )

        return LogosHealthResponse(
            status=overall_status,
            service="sophia",
            version="0.1.0",
            dependencies=dependencies,
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
        span = get_current_span()
        span.update_name("sophia.state.get")
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

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
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

        This endpoint updates the current state node in Neo4j and emits
        a CWM-A state envelope for the change. The emitted CWMState follows
        the unified contract defined in PHASE2_SPEC.

        Requires authentication via Bearer token.
        """
        span = get_current_span()
        span.update_name("sophia.state.update")
        span.set_attribute("state.id", "current_state")
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            # Get existing state for diff computation
            existing_state = _hcg_client.get_node("current_state")
            before_properties = (
                existing_state.get("properties", {}) if existing_state else {}
            )

            if existing_state:
                # Update existing state node
                # Delete and recreate with new properties (Neo4j pattern)
                _hcg_client.delete_node("current_state")

            # Create new state node with SHACL validation
            _hcg_client.add_node(
                uuid="current_state",
                name="Current State",
                node_type="state",
                properties=request.state,
                source="orchestrator",
                derivation="observed",
            )

            # Emit CWM-A state envelope
            cwm_state = None
            entity_diffs = None
            if _cwm_a_state:
                # Compute entity diffs from the state update
                from sophia.cwm_a import EntityDiff, ValidationResult

                # Treat each top-level key in state as an entity
                diffs = []
                all_keys = set(
                    list(before_properties.keys()) + list(request.state.keys())
                )

                for key in all_keys:
                    before_val = before_properties.get(key)
                    after_val = request.state.get(key)

                    if before_val != after_val:
                        if before_val is None:
                            operation = "create"
                        elif after_val is None:
                            operation = "delete"
                        else:
                            operation = "update"

                        changed_props = []
                        if isinstance(before_val, dict) and isinstance(after_val, dict):
                            changed_props = [
                                k
                                for k in set(
                                    list(before_val.keys()) + list(after_val.keys())
                                )
                                if before_val.get(k) != after_val.get(k)
                            ]

                        diffs.append(
                            EntityDiff(
                                entity_id=key,
                                entity_type="state_entity",
                                operation=operation,
                                before=(
                                    before_val
                                    if isinstance(before_val, dict)
                                    else {"value": before_val} if before_val else None
                                ),
                                after=(
                                    after_val
                                    if isinstance(after_val, dict)
                                    else {"value": after_val} if after_val else None
                                ),
                                changed_properties=(
                                    changed_props if changed_props else None
                                ),
                            )
                        )

                if diffs:
                    cwm_state = _cwm_a_state.emit_state_update(
                        entity_diffs=diffs,
                        validation=ValidationResult(passed=True),
                        confidence=1.0,
                        derivation="observed",
                        tags=["source:api", "endpoint:/state"],
                    )
                    entity_diffs = [d.model_dump() for d in diffs]

            # Also update in-memory planner state if available
            if _planner:
                _planner.update_state(request.state)

            return StateUpdateResponse(
                state_id="current_state",
                cwm_state_id=cwm_state.state_id if cwm_state else None,
                validation_passed=True,
                entity_diffs=entity_diffs,
            )

        except ValueError as e:
            # SHACL validation failed
            logger.error(f"State validation failed: {e}")

            # Emit failed validation state
            if _cwm_a_state:
                from sophia.cwm_a import ValidationResult

                _cwm_a_state.emit_state_update(
                    entity_diffs=[],
                    validation=ValidationResult(passed=False, violations=[str(e)]),
                    confidence=0.0,
                    derivation="observed",
                    tags=["source:api", "endpoint:/state", "validation:failed"],
                )

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"State validation failed: {str(e)}",
            )
        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"Error updating state: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update state: {str(e)}",
            )

    # CWM State history endpoint
    @app.get(
        "/state/cwm",
        response_model=CWMStateListResponse,
        dependencies=[Depends(verify_token)],
        tags=["state"],
    )
    async def get_cwm_states(
        model_type: Optional[str] = Query(
            default=None,
            description="Filter by model type (CWM_A, CWM_G, CWM_E)",
        ),
        limit: int = Query(
            default=100,
            ge=1,
            le=1000,
            description="Maximum number of states to return",
        ),
    ) -> CWMStateListResponse:
        """Get CWM state history.

        Returns recent CWMState emissions for diagnostics and auditing.
        Currently returns CWM-A states; future versions will aggregate
        across all CWM model types.

        Requires authentication via Bearer token.
        """
        states = []

        # Get CWM-A states
        if _cwm_a_state and (model_type is None or model_type == "CWM_A"):
            cwm_a_states = _cwm_a_state.get_state_history(limit=limit)
            for s in cwm_a_states:
                # Extract provenance fields from data (now on node, not envelope)
                data = s.data
                states.append(
                    CWMStateResponse(
                        state_id=s.state_id,
                        model_type=s.model_type,
                        source=data.get("source", "unknown"),
                        timestamp=s.timestamp.isoformat(),
                        confidence=data.get("confidence", 0.0),
                        status=data.get("derivation", "observed"),
                        links=data.get("links", {}),
                        tags=data.get("tags", []),
                        data=data,
                    )
                )

        # TODO: Add CWM-G and CWM-E state retrieval when available

        # Sort by timestamp descending
        states.sort(key=lambda x: x.timestamp, reverse=True)

        return CWMStateListResponse(
            states=states[:limit],
            total=len(states),
            model_type=model_type,
        )

    # CWM persistence endpoint - reads from Neo4j
    @app.get(
        "/cwm",
        response_model=CWMStateListResponse,
        dependencies=[Depends(verify_token)],
        tags=["cwm"],
    )
    async def get_cwm_persisted(
        types: Optional[str] = Query(
            default=None,
            description="Comma-separated CWM types to filter (cwm_a, cwm_g, cwm_e)",
        ),
        after_timestamp: Optional[str] = Query(
            default=None,
            description="Only return states after this ISO timestamp",
        ),
        limit: int = Query(
            default=20,
            ge=1,
            le=100,
            description="Maximum number of states to return",
        ),
    ) -> CWMStateListResponse:
        """Get persisted CWM states from Neo4j.

        Returns CWMState envelopes persisted to Neo4j via the CWMPersistence
        service. This endpoint is intended for Hermes to query historical
        cognitive states for context.

        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            from typing import cast
            from sophia.cwm.persistence import CWMType

            # Parse types filter
            type_list: list[CWMType] | None = None
            if types:
                type_list = cast(list[CWMType], [t.strip() for t in types.split(",")])

            # Parse timestamp filter with RFC3339 support
            parsed_timestamp = None
            if after_timestamp:
                try:
                    # Handle RFC3339 "Z" suffix
                    ts = after_timestamp.replace("Z", "+00:00")
                    parsed_timestamp = datetime.fromisoformat(ts)
                except ValueError as e:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid timestamp format: {after_timestamp}. Expected ISO 8601.",
                    ) from e

            # Query Neo4j
            cwm_states = _cwm_persistence.find_states(
                types=type_list,
                after_timestamp=parsed_timestamp,
                limit=limit,
            )

            # Convert to response format
            response_states = []
            for s in cwm_states:
                # Extract provenance fields from data (now on node, not envelope)
                data = s.data if isinstance(s.data, dict) else {}
                response_states.append(
                    CWMStateResponse(
                        state_id=s.state_id,
                        model_type=s.model_type,
                        source=data.get("source", "unknown"),
                        timestamp=s.timestamp.isoformat() if s.timestamp else "",
                        confidence=data.get("confidence", 0.0),
                        status=data.get("derivation", "observed"),
                        links=data.get("links", {}),
                        tags=data.get("tags", []),
                        data=data,
                    )
                )

            return CWMStateListResponse(
                states=response_states,
                total=len(response_states),
                model_type=types,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving persisted CWM states: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve CWM states: {str(e)}",
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
        span = get_current_span()
        span.update_name("sophia.plan")
        span.set_attribute("plan.goal", str(request.goal)[:200])
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
                uuid=plan_id,
                name=f"Plan {plan_id[:8]}",
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
                source="planner",
                derivation="imagined",
            )

            # Link plan to goal if it exists in HCG
            goal_id = goal_payload.get("target_state", "")
            goal_node_id = f"goal_{goal_id}" if goal_id else None
            if goal_node_id:
                try:
                    goal_node = _hcg_client.get_node(goal_node_id)
                    if goal_node:
                        _hcg_client.add_edge(
                            edge_uuid=f"e_{plan_id}_achieves_goal",
                            source_uuid=plan_id,
                            target_uuid=goal_node_id,
                            relation="ACHIEVES",
                        )
                except Exception as e:
                    logger.warning(f"Could not link plan to goal: {e}")

            # Emit feedback to Hermes
            if _feedback_dispatcher:
                try:
                    step_summary = "→".join(s.action_type for s in plan_step_models)
                    _feedback_dispatcher.emit(
                        FeedbackPayload(
                            correlation_id=request.correlation_id,
                            plan_id=plan_id,
                            feedback_type="plan",
                            outcome="created",
                            reason=f"Generated {len(plan_step_models)}-step plan: {step_summary}",
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit plan feedback: {e}")

            return PlanResponse(
                plan=plan_step_models,
                goal=goal_payload,
                plan_id=plan_id,
            )

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
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
        span = get_current_span()
        span.update_name("sophia.imagine")
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
                _hcg_client.add_node(
                    uuid=state_id,
                    name=f"Imagined State {i + 1}",
                    node_type="state",
                    properties={
                        "description": state.description,
                        "model_version": request.model_version,
                        "horizon": request.horizon,
                        "horizon_step": i,
                        "assumptions": request.assumptions or [],
                        "imagination_id": imagination_id,
                    },
                    source="jepa_runner",
                    derivation="imagined",
                    confidence=state.confidence,
                    links={"imagination_id": imagination_id},
                )

            return ImagineResponse(
                imagined_states=imagined_states,
                imagination_id=imagination_id,
                model_version=request.model_version,
                horizon=request.horizon,
                assumptions=request.assumptions or [],
            )

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
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
        span = get_current_span()
        span.update_name("sophia.simulate")
        span.set_attribute("simulate.horizon", request.k_steps)
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
                        with _hcg_client.driver.session(
                            database=_hcg_client.database
                        ) as session:
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
                    uuid=process.process_id,
                    name=(
                        process.description[:50]
                        if process.description
                        else f"Process {process.process_id[:8]}"
                    ),
                    node_type="process",
                    properties={
                        "description": process.description,
                        "model_version": process.model_version,
                        "horizon": process.horizon,
                        "assumptions": process.assumptions,
                        "imagined": True,
                        "simulation_id": result.simulation_id,
                        **process.properties,
                    },
                    source="jepa_runner",
                    derivation="imagined",
                    confidence=process.confidence,
                    links={"simulation_id": result.simulation_id},
                )

            # Store imagined states in Neo4j
            for state in result.imagined_states:
                _hcg_client.add_node(
                    uuid=state.state_id,
                    name=(
                        state.description[:50]
                        if state.description
                        else f"State {state.state_id[:8]}"
                    ),
                    node_type="state",
                    properties={
                        "step": state.step,
                        "description": state.description,
                        "model_version": state.model_version,
                        "horizon": state.horizon,
                        "assumptions": state.assumptions,
                        "imagined": True,
                        "simulation_id": result.simulation_id,
                        "state_data": state.state_data,
                    },
                    source="jepa_runner",
                    derivation="imagined",
                    confidence=state.confidence,
                    links={"simulation_id": result.simulation_id},
                )

                # Link state to simulation
                _hcg_client.add_edge(
                    edge_uuid=f"e_{result.simulation_id}_{state.state_id}",
                    source_uuid=result.simulation_id,
                    target_uuid=state.state_id,
                    relation="PRODUCES",
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
                uuid=result.simulation_id,
                name=f"Simulation {result.simulation_id[:8]}",
                node_type="simulation",
                properties=simulation_properties,
                source="jepa_runner",
                derivation="imagined",
            )

            # Link simulation to media sample if provided
            if request.media_sample_id:
                try:
                    _hcg_client.add_edge(
                        edge_uuid=f"e_{result.simulation_id}_uses_{request.media_sample_id}",
                        source_uuid=result.simulation_id,
                        target_uuid=request.media_sample_id,
                        relation="USES_MEDIA",
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

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"Error running simulation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to run simulation: {str(e)}",
            )

    # Hermes proposal ingestion endpoint
    # NOTE: Intentionally unauthenticated for local development.
    # In production, set SOPHIA_ENABLE_DEV_ENDPOINTS=0 (or unset it) to disable.
    @app.post(
        "/ingest/hermes_proposal",
        response_model=HermesProposalResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["ingestion"],
    )
    async def ingest_hermes_proposal(
        request: HermesProposalRequest,
    ) -> HermesProposalResponse:
        """Receive a proposal from Hermes for cognitive processing.

        This endpoint accepts structured proposals from Hermes and logs them
        for observability. Sophia will process the proposal and decide what
        semantic nodes to create based on her cognitive evaluation.

        This endpoint is intentionally unauthenticated. It is gated behind
        the ``SOPHIA_ENABLE_DEV_ENDPOINTS`` environment variable (must be
        set to ``1``) for safety outside of local development.
        """
        if os.environ.get("SOPHIA_ENABLE_DEV_ENDPOINTS", "1") != "1":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Dev endpoints are disabled (set SOPHIA_ENABLE_DEV_ENDPOINTS=1 to enable)",
            )
        span = get_current_span()
        span.update_name("sophia.ingest.hermes_proposal")
        span.set_attribute("ingest.proposal_id", str(request.proposal_id))
        span.set_attribute(
            "ingest.node_count", len(request.plan_steps) if request.plan_steps else 0
        )
        # Log the proposal for observability
        logger.info(
            f"Received proposal {request.proposal_id} from {request.source_service} "
            f"(provider: {request.llm_provider}, model: {request.model}, "
            f"confidence: {request.confidence})"
        )
        if request.plan_steps:
            logger.debug(
                f"Proposal {request.proposal_id} contains {len(request.plan_steps)} plan steps"
            )
        if request.imagined_states:
            logger.debug(
                f"Proposal {request.proposal_id} contains {len(request.imagined_states)} imagined states"
            )
        if request.raw_text:
            logger.debug(
                f"Proposal {request.proposal_id} raw text: {request.raw_text[:100]}..."
            )

        # Process through ProposalProcessor if available
        stored_node_ids: List[str] = []
        stored_edge_ids: List[str] = []
        relevant_context: List[Dict[str, Any]] = []

        if _proposal_processor:
            try:
                result = _proposal_processor.process(request.model_dump())
                stored_node_ids = result.get("stored_node_ids", [])
                stored_edge_ids = result.get("stored_edge_ids", [])
                relevant_context = result.get("relevant_context", [])
                span.set_attribute("ingest.stored_count", len(stored_node_ids))
                span.set_attribute("ingest.stored_edge_count", len(stored_edge_ids))
                span.set_attribute("ingest.context_count", len(relevant_context))
                logger.info(
                    f"Proposal {request.proposal_id}: stored {len(stored_node_ids)} nodes, "
                    f"{len(stored_edge_ids)} edges, "
                    f"found {len(relevant_context)} context items"
                )
            except Exception as e:
                logger.error(f"ProposalProcessor failed for {request.proposal_id}: {e}")
                span.record_exception(e)
        else:
            logger.debug(
                "ProposalProcessor not initialized, skipping cognitive processing"
            )

        # Emit feedback to Hermes
        if _feedback_dispatcher:
            try:
                _feedback_dispatcher.emit(
                    FeedbackPayload(
                        correlation_id=request.correlation_id,
                        feedback_type="observation",
                        outcome="accepted",
                        reason=f"Processed proposal {request.proposal_id}: "
                        f"{len(stored_node_ids)} nodes stored, "
                        f"{len(stored_edge_ids)} edges stored, "
                        f"{len(relevant_context)} context items",
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to emit proposal feedback: {e}")

        return HermesProposalResponse(
            proposal_id=request.proposal_id,
            stored_node_ids=stored_node_ids,
            stored_edge_ids=stored_edge_ids,
            relevant_context=relevant_context,
            status="accepted",
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
        span = get_current_span()
        span.update_name("sophia.execute")
        span.set_attribute("execute.plan_id", str(request.plan_id))
        span.set_attribute(
            "execute.step", request.step_index if request.step_index is not None else -1
        )
        if not _executor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Executor service not available",
            )

        try:
            # Validate that the plan exists in Neo4j
            if _hcg_client:
                plan_node = _hcg_client.get_node(request.plan_id)
                if not plan_node:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Plan not found: {request.plan_id}",
                    )
            else:
                # If HCG client is not available, we can't validate the plan
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="HCG client not available to validate plan",
                )

            execution_id = str(uuid.uuid4())
            results: List[ExecutionResult] = []

            # Create execution container node (parallel to simulation for imagined)
            _hcg_client.add_node(
                uuid=execution_id,
                name=f"Execution {execution_id[:8]}",
                node_type="execution",
                properties={
                    "plan_id": request.plan_id,
                    "dry_run": request.dry_run,
                    "step_index": request.step_index,
                },
                source="executor",
                derivation="observed",
                links={"plan_id": request.plan_id},
            )

            # Get plan details from Neo4j
            plan_props = plan_node.get("properties", {})
            plan_steps = plan_props.get("steps", [])
            goal = plan_props.get("goal", {})
            target_state = goal.get("target_state", "")

            # Validate step_index if provided
            if request.step_index is not None:
                if request.step_index >= len(plan_steps):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid step_index {request.step_index}: plan has {len(plan_steps)} steps",
                    )

            # Execute each step in the plan
            steps_to_execute = (
                [plan_steps[request.step_index]]
                if request.step_index is not None
                else plan_steps
            )

            state_changes: Dict[str, Any] = {}

            for step in steps_to_execute:
                step_model = PlanStep(
                    id=step.get("id", "unknown"),
                    name=step.get("name", "Unknown"),
                    type=step.get("type", "action"),
                    action_type=step.get("action_type", "UNKNOWN"),
                    target=step.get("target", ""),
                )

                if request.dry_run:
                    result = ExecutionResult(
                        step=step_model,
                        status="simulated",
                        message="Dry run - no actual execution",
                        state_changes={},
                    )
                else:
                    # Execute the action and track state changes
                    result = ExecutionResult(
                        step=step_model,
                        status="success",
                        message=f"Executed {step_model.action_type} on {step_model.target}",
                        state_changes={},
                    )

                    # Create process node for the executed step (parallel to imagined_process)
                    process_id = f"{execution_id}_process_{step_model.id}"
                    _hcg_client.add_node(
                        uuid=process_id,
                        name=step_model.name,
                        node_type="process",
                        properties={
                            "action_type": step_model.action_type,
                            "target": step_model.target,
                            "step_id": step_model.id,
                        },
                        source="executor",
                        derivation="observed",
                        links={"execution_id": execution_id},
                    )

                    # Link process to execution container
                    _hcg_client.add_edge(
                        edge_uuid=f"e_{execution_id}_{process_id}",
                        source_uuid=execution_id,
                        target_uuid=process_id,
                        relation="PRODUCES",
                    )

                results.append(result)

            # Apply state changes based on the goal's target_state
            if not request.dry_run and target_state:
                # Get current state
                current_state_node = _hcg_client.get_node("current_state")
                current_state = (
                    current_state_node.get("properties", {})
                    if current_state_node
                    else {}
                )

                # Apply goal-based state transitions
                # This handles known goal patterns like "red_block_in_bin"
                if target_state == "red_block_in_bin":
                    if "red_block" in current_state:
                        current_state["red_block"]["location"] = "bin"
                        current_state["red_block"]["grasped"] = False
                        state_changes["red_block"] = current_state["red_block"]

                # Update state in Neo4j if there are changes
                if state_changes:
                    _hcg_client.delete_node("current_state")
                    _hcg_client.add_node(
                        uuid="current_state",
                        name="Current State",
                        node_type="state",
                        properties=current_state,
                        source="executor",
                        derivation="observed",
                        links={"execution_id": execution_id},
                    )

                    # Link execution to resulting state
                    _hcg_client.add_edge(
                        edge_uuid=f"e_{execution_id}_results_in_state",
                        source_uuid=execution_id,
                        target_uuid="current_state",
                        relation="RESULTS_IN",
                    )

                    # Update planner state
                    if _planner:
                        _planner.update_state(current_state)

                    # Update the last result with state changes
                    if results:
                        results[-1] = ExecutionResult(
                            step=results[-1].step,
                            status=results[-1].status,
                            message=results[-1].message,
                            state_changes=state_changes,
                        )

            overall_status = (
                "success" if all(r.status == "success" for r in results) else "partial"
            )

            # Emit feedback to Hermes
            if _feedback_dispatcher:
                try:
                    step_results = [
                        StepResult(
                            step_index=i,
                            action=r.step.action_type,
                            outcome="success" if r.status == "success" else "failure",
                            error=r.message if r.status != "success" else None,
                        )
                        for i, r in enumerate(results)
                    ]
                    feedback_outcome: Literal["success", "partial"] = (
                        "success" if overall_status == "success" else "partial"
                    )
                    _feedback_dispatcher.emit(
                        FeedbackPayload(
                            plan_id=request.plan_id,
                            execution_id=execution_id,
                            feedback_type="execution",
                            outcome=feedback_outcome,
                            reason=f"Executed {len(results)} steps: {overall_status}",
                            step_results=step_results,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit execution feedback: {e}")

            return ExecuteResponse(
                plan_id=request.plan_id,
                results=results,
                overall_status=overall_status,
                execution_id=execution_id,
            )

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"Error executing plan: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute plan: {str(e)}",
            )

    # =========================================================================
    # HCG Graph API Endpoints (read-only, for Apollo's Neo4j removal)
    # =========================================================================

    @app.get(
        "/hcg/snapshot",
        response_model=HCGGraphSnapshotResponse,
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def get_hcg_snapshot(
        entity_type: Optional[str] = Query(
            default=None,
            description="Filter entities by type",
        ),
        limit: int = Query(
            default=1000,
            ge=1,
            le=100000,
            description="Maximum number of entities/edges to return. This "
            "endpoint returns node/edge metadata only (no embedding vectors), "
            "so the payload stays modest even at the cap.",
        ),
        include_embeddings: bool = Query(
            default=False,
            description="Attach each entity's stored vector (for semantic layout)",
        ),
    ) -> HCGGraphSnapshotResponse:
        """Get a snapshot of the entire HCG graph for visualization.

        Returns all entities and edges in the graph, suitable for rendering
        in Apollo's graph visualization component.

        Requires authentication via Bearer token.
        """
        span = get_current_span()
        span.update_name("sophia.hcg.snapshot")
        span.set_attribute("hcg.limit", limit)
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            # Fetch all nodes
            nodes = _hcg_client.list_all_nodes(node_type=entity_type, limit=limit)

            # Optionally attach stored entity vectors so Apollo's explorer can
            # lay nodes out by embedding (semantic layout). Two things matter:
            #   1. Embeddings are sharded across per-type Milvus collections
            #      (Entity/Concept/State/Process) and a snapshot returns nodes of
            #      every type -- so each uuid must be looked up in the collection
            #      that actually holds its vector, not just hcg_entity_embeddings,
            #      or every Concept/State/Process node silently comes back null.
            #   2. The `uuid in [...]` filter is batched: a single list of up to
            #      10k 36-char uuids is ~400 KB and exceeds Milvus's expression
            #      cap, which would throw and (via the except) null *everything*.
            # uuid is a VARCHAR primary key, so each value is double-quoted.
            emb_by_uuid: Dict[str, Any] = {}
            if include_embeddings and nodes:
                try:
                    # Default Milvus connection is established at startup.
                    from pymilvus import Collection

                    from logos_hcg.sync import COLLECTION_NAMES
                    from sophia.ingestion.proposal_processor import _collection_for

                    uuids_by_collection: Dict[str, list[str]] = {}
                    for _n in nodes:
                        _cname = COLLECTION_NAMES[_collection_for(_n["type"])]
                        uuids_by_collection.setdefault(_cname, []).append(
                            str(_n["uuid"])
                        )

                    for _cname, _uuids in uuids_by_collection.items():
                        try:
                            _col = Collection(_cname)
                            _col.load()
                            for _i in range(0, len(_uuids), _SNAPSHOT_EMB_BATCH):
                                _batch = _uuids[_i : _i + _SNAPSHOT_EMB_BATCH]
                                _uuid_list = ", ".join(f'"{u}"' for u in _batch)
                                for _row in _col.query(
                                    expr=f"uuid in [{_uuid_list}]",
                                    output_fields=["uuid", "embedding"],
                                    limit=len(_batch),
                                ):
                                    emb_by_uuid[_row["uuid"]] = _row.get("embedding")
                        except Exception as _e:
                            # Log per-collection so a genuine query failure is
                            # visible, not indistinguishable from nodes that
                            # simply have no stored vector.
                            logger.warning(
                                "snapshot embeddings unavailable for %s: %s",
                                _cname,
                                _e,
                            )
                except Exception as _e:
                    logger.warning(f"snapshot embeddings unavailable: {_e}")

            # Convert to response format
            entities: List[HCGEntityResponse] = []
            for node in nodes:
                props = sanitize_neo4j_properties(node.get("properties", {}))
                entities.append(
                    HCGEntityResponse(
                        id=node["uuid"],
                        type=node["type"],
                        name=node["name"],
                        properties=props,
                        labels=[],
                        created_at=props.get("created"),
                        embedding=emb_by_uuid.get(node["uuid"]),
                    )
                )

            # Fetch all edges
            raw_edges = _hcg_client.list_all_edges(limit=limit)

            # Build set of entity IDs for filtering orphan edges
            entity_ids = {e.id for e in entities}

            # Convert to response format, filtering out edges with missing nodes
            edges: List[HCGEdgeResponse] = []
            for edge in raw_edges:
                # Skip edges where source or target isn't in the returned entities
                if edge["source"] not in entity_ids or edge["target"] not in entity_ids:
                    continue
                edges.append(
                    HCGEdgeResponse(
                        id=edge["id"],
                        source_id=edge["source"],
                        target_id=edge["target"],
                        edge_type=edge["relation"],
                        properties=sanitize_neo4j_properties(
                            edge.get("properties", {})
                        ),
                    )
                )

            return HCGGraphSnapshotResponse(
                entities=entities,
                edges=edges,
                entity_count=len(entities),
                edge_count=len(edges),
            )

        except HTTPException:
            raise
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            logger.error(f"Error fetching HCG snapshot: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch HCG snapshot: {str(e)}",
            )

    @app.get(
        "/hcg/entities/{entity_id}",
        response_model=HCGEntityResponse,
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def get_hcg_entity(entity_id: str) -> HCGEntityResponse:
        """Get a single entity by ID.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            node = _hcg_client.get_node(entity_id)

            if not node:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Entity not found: {entity_id}",
                )

            props = sanitize_neo4j_properties(node.get("properties", {}))
            return HCGEntityResponse(
                id=node["uuid"],
                type=node["type"],
                name=node["name"],
                properties=props,
                labels=[],
                created_at=props.get("created"),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching entity {entity_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch entity: {str(e)}",
            )

    @app.get(
        "/hcg/edges",
        response_model=List[HCGEdgeResponse],
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def list_hcg_edges(
        edge_type: Optional[str] = Query(
            default=None,
            description="Filter by edge/relation type (e.g., 'ENABLES', 'ACHIEVES')",
        ),
        source_id: Optional[str] = Query(
            default=None,
            description="Filter by source entity ID",
        ),
        target_id: Optional[str] = Query(
            default=None,
            description="Filter by target entity ID",
        ),
        limit: int = Query(
            default=1000,
            ge=1,
            le=10000,
            description="Maximum number of edges to return",
        ),
    ) -> List[HCGEdgeResponse]:
        """List causal edges with optional filters.

        Supports filtering by edge type, source entity, or target entity.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            raw_edges = _hcg_client.list_all_edges(
                relation_type=edge_type,
                source_uuid=source_id,
                target_uuid=target_id,
                limit=limit,
            )

            edges: List[HCGEdgeResponse] = []
            for edge in raw_edges:
                edges.append(
                    HCGEdgeResponse(
                        id=edge["id"],
                        source_id=edge["source"],
                        target_id=edge["target"],
                        edge_type=edge["relation"],
                        properties=sanitize_neo4j_properties(
                            edge.get("properties", {})
                        ),
                    )
                )

            return edges

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing HCG edges: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list edges: {str(e)}",
            )

    @app.get(
        "/hcg/entities",
        response_model=List[HCGEntityResponse],
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def list_hcg_entities(
        entity_type: Optional[str] = Query(
            default=None,
            alias="type",
            description="Filter by entity type",
        ),
        limit: int = Query(
            default=100,
            ge=1,
            le=500,
            description="Maximum number of entities to return",
        ),
        offset: int = Query(
            default=0,
            ge=0,
            description="Number of entities to skip",
        ),
    ) -> List[HCGEntityResponse]:
        """List entities from HCG with optional type filter.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            raw_nodes = _hcg_client.list_all_nodes(
                node_type=entity_type,
                limit=limit,
            )

            # Apply offset manually since list_all_nodes doesn't support it
            raw_nodes = raw_nodes[offset : offset + limit]

            entities: List[HCGEntityResponse] = []
            for node in raw_nodes:
                props = sanitize_neo4j_properties(node.get("properties", {}))
                entities.append(
                    HCGEntityResponse(
                        id=node.get("uuid", node.get("id", "")),
                        type=node.get("type", "unknown"),
                        name=node.get("name", ""),
                        properties=props,
                        labels=[],
                        created_at=props.get("created"),
                    )
                )

            return entities

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing HCG entities: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list entities: {str(e)}",
            )

    @app.get(
        "/hcg/states",
        response_model=List[HCGEntityResponse],
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def list_hcg_states(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> List[HCGEntityResponse]:
        """List state entities from HCG.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            raw_nodes = _hcg_client.list_all_nodes(node_type="state", limit=limit)
            raw_nodes = raw_nodes[offset : offset + limit]

            states: List[HCGEntityResponse] = []
            for node in raw_nodes:
                props = sanitize_neo4j_properties(node.get("properties", {}))
                states.append(
                    HCGEntityResponse(
                        id=node.get("uuid", node.get("id", "")),
                        type="state",
                        name=node.get("name", ""),
                        properties=props,
                        labels=[],
                        created_at=props.get("created"),
                    )
                )

            return states

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing HCG states: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list states: {str(e)}",
            )

    @app.get(
        "/hcg/processes",
        response_model=List[HCGEntityResponse],
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def list_hcg_processes(
        process_status: Optional[str] = Query(
            default=None,
            alias="status",
            description="Filter by process status",
        ),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> List[HCGEntityResponse]:
        """List process entities from HCG.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            raw_nodes = _hcg_client.list_all_nodes(node_type="process", limit=limit)
            raw_nodes = raw_nodes[offset : offset + limit]

            # Filter by status if provided
            if process_status:
                raw_nodes = [
                    n
                    for n in raw_nodes
                    if n.get("properties", {}).get("status") == process_status
                ]

            processes: List[HCGEntityResponse] = []
            for node in raw_nodes:
                props = sanitize_neo4j_properties(node.get("properties", {}))
                processes.append(
                    HCGEntityResponse(
                        id=node.get("uuid", node.get("id", "")),
                        type="process",
                        name=node.get("name", ""),
                        properties=props,
                        labels=[],
                        created_at=props.get("created"),
                    )
                )

            return processes

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing HCG processes: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list processes: {str(e)}",
            )

    @app.get(
        "/hcg/plans",
        response_model=List[HCGEntityResponse],
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def list_hcg_plans(
        goal_id: Optional[str] = Query(default=None, description="Filter by goal ID"),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> List[HCGEntityResponse]:
        """List plan entities from HCG.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            raw_nodes = _hcg_client.list_all_nodes(node_type="plan", limit=limit)

            # Filter by goal_id if provided
            if goal_id:
                raw_nodes = [
                    n
                    for n in raw_nodes
                    if n.get("properties", {}).get("goal_id") == goal_id
                ]

            plans: List[HCGEntityResponse] = []
            for node in raw_nodes:
                props = sanitize_neo4j_properties(node.get("properties", {}))
                plans.append(
                    HCGEntityResponse(
                        id=node.get("uuid", node.get("id", "")),
                        type="plan",
                        name=node.get("name", ""),
                        properties=props,
                        labels=[],
                        created_at=props.get("created"),
                    )
                )

            return plans

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing HCG plans: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list plans: {str(e)}",
            )

    @app.get(
        "/hcg/history",
        response_model=List[HCGEntityResponse],
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def list_hcg_history(
        state_id: Optional[str] = Query(default=None, description="Filter by state ID"),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> List[HCGEntityResponse]:
        """List state history from HCG.

        Requires authentication via Bearer token.
        """
        if not _hcg_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HCG client not available",
            )

        try:
            # Query state_history type nodes
            raw_nodes = _hcg_client.list_all_nodes(
                node_type="state_history", limit=limit
            )

            # Filter by state_id if provided
            if state_id:
                raw_nodes = [
                    n
                    for n in raw_nodes
                    if n.get("properties", {}).get("state_id") == state_id
                ]

            history: List[HCGEntityResponse] = []
            for node in raw_nodes:
                props = sanitize_neo4j_properties(node.get("properties", {}))
                history.append(
                    HCGEntityResponse(
                        id=node.get("uuid", node.get("id", "")),
                        type="state_history",
                        name=node.get("name", ""),
                        properties=props,
                        labels=[],
                        created_at=props.get("created"),
                    )
                )

            return history

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing HCG history: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list history: {str(e)}",
            )

    @app.get(
        "/hcg/health",
        dependencies=[Depends(verify_token)],
        tags=["hcg"],
    )
    async def hcg_health_check() -> dict:
        """Check HCG connection health.

        Requires authentication via Bearer token.
        """
        neo4j_connected = False
        if _hcg_client:
            try:
                # Simple connectivity check
                _hcg_client.list_all_nodes(limit=1)
                neo4j_connected = True
            except Exception:
                neo4j_connected = False

        return {
            "status": "healthy" if neo4j_connected else "degraded",
            "timestamp": datetime.now().isoformat(),
            "neo4j_connected": neo4j_connected,
        }

    # =========================================================================
    # Persona API Endpoints (CWM-E)
    # =========================================================================

    def _cwmstate_to_persona_entry(state: Dict[str, Any]) -> Optional[PersonaEntryFull]:
        """Convert a CWMState dict to PersonaEntryFull, or None if not a persona entry."""
        data = state.get("data", {})
        entry = data.get("entry", {})
        if not entry or entry.get("deleted"):
            return None
        try:
            return PersonaEntryFull(
                entry_id=entry.get("entry_id", ""),
                timestamp=state.get("timestamp", datetime.now()),
                entry_type=entry.get("entry_type", "observation"),
                content=entry.get("content", ""),
                summary=entry.get("summary"),
                trigger=entry.get("trigger"),
                sentiment=entry.get("sentiment"),
                confidence=entry.get("confidence"),
                related_process_ids=entry.get("related_process_ids", []),
                related_goal_ids=entry.get("related_goal_ids", []),
                emotion_tags=entry.get("emotion_tags", []),
                metadata=entry.get("metadata", {}),
            )
        except Exception:
            return None

    def _get_persona_entries(
        entry_type: Optional[str] = None,
        sentiment: Optional[str] = None,
        related_process_id: Optional[str] = None,
        related_goal_id: Optional[str] = None,
        after_timestamp: Optional[datetime] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[PersonaEntryFull], int]:
        """Fetch and filter persona entries from CWM-E states."""
        if not _cwm_persistence:
            return [], 0

        # Fetch more than needed to account for filtering
        raw_states = _cwm_persistence.find_states(
            types=["cwm_e"],
            after_timestamp=after_timestamp,
            limit=limit * 3 + offset,
        )

        # Convert and filter
        entries: List[PersonaEntryFull] = []
        seen_ids: set[str] = set()
        deleted_ids: set[str] = set()

        for state in raw_states:
            # Check if this is a deletion tombstone first
            state_dict = (
                state.__dict__
                if hasattr(state, "__dict__")
                else {"data": state.data, "timestamp": state.timestamp}
            )
            data = state_dict.get("data", {})
            entry_data = data.get("entry", {})
            if entry_data and entry_data.get("deleted"):
                entry_id = entry_data.get("entry_id")
                if entry_id:
                    deleted_ids.add(entry_id)
                continue

            entry = _cwmstate_to_persona_entry(
                state.__dict__
                if hasattr(state, "__dict__")
                else {"data": state.data, "timestamp": state.timestamp}
            )
            if not entry:
                continue

            # Dedupe by entry_id (keep latest by timestamp)
            if entry.entry_id in seen_ids:
                continue
            # Skip if this entry has been deleted
            if entry.entry_id in deleted_ids:
                continue

            seen_ids.add(entry.entry_id)

            # Apply filters
            if entry_type and entry.entry_type != entry_type:
                continue
            if sentiment and entry.sentiment != sentiment:
                continue
            if (
                related_process_id
                and related_process_id not in entry.related_process_ids
            ):
                continue
            if related_goal_id and related_goal_id not in entry.related_goal_ids:
                continue

            entries.append(entry)

        total = len(entries)
        # Apply pagination
        entries = entries[offset : offset + limit]
        return entries, total

    @app.post(
        "/persona/entries",
        response_model=PersonaEntryResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(verify_token)],
        tags=["persona"],
    )
    async def create_persona_entry(
        request: PersonaEntryCreate,
    ) -> PersonaEntryResponse:
        """Create a new persona diary entry.

        Stores the entry as a CWM-E state in Neo4j.
        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            from datetime import timezone
            from uuid import uuid4
            from sophia.cwm_a.state_service import CWMState

            entry_id = f"persona_{uuid4().hex[:12]}"
            state_id = f"cwm_e_{uuid4().hex[:12]}"
            timestamp = datetime.now(timezone.utc)

            # Build CWMState envelope
            cwm_state = CWMState(
                state_id=state_id,
                model_type="CWM_E",
                timestamp=timestamp,
                data={
                    "entry": {
                        "entry_id": entry_id,
                        "entry_type": request.entry_type,
                        "content": request.content,
                        "summary": request.summary,
                        "trigger": request.trigger,
                        "sentiment": request.sentiment,
                        "confidence": request.confidence,
                        "related_process_ids": request.related_process_ids,
                        "related_goal_ids": request.related_goal_ids,
                        "emotion_tags": request.emotion_tags,
                        "metadata": request.metadata,
                    },
                    "source": "persona_api",
                    "derivation": "observed",
                    "confidence": request.confidence or 1.0,
                    "tags": [f"entry_type:{request.entry_type}"],
                    "links": {},
                },
            )

            _cwm_persistence.persist(cwm_state)

            logger.info(f"Created persona entry: {entry_id}")
            return PersonaEntryResponse(
                entry_id=entry_id,
                cwm_state_id=state_id,
                timestamp=timestamp,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating persona entry: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create persona entry: {str(e)}",
            )

    @app.get(
        "/persona/entries",
        response_model=PersonaListResponse,
        dependencies=[Depends(verify_token)],
        tags=["persona"],
    )
    async def list_persona_entries(
        entry_type: Optional[str] = Query(None, description="Filter by entry type"),
        sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
        related_process_id: Optional[str] = Query(
            None, description="Filter by process ID"
        ),
        related_goal_id: Optional[str] = Query(None, description="Filter by goal ID"),
        after_timestamp: Optional[str] = Query(
            None, description="ISO timestamp filter"
        ),
        limit: int = Query(20, ge=1, le=150, description="Max entries"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
    ) -> PersonaListResponse:
        """List persona diary entries with optional filters.

        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            # Parse timestamp
            parsed_ts = None
            if after_timestamp:
                parsed_ts = datetime.fromisoformat(after_timestamp)

            entries, total = _get_persona_entries(
                entry_type=entry_type,
                sentiment=sentiment,
                related_process_id=related_process_id,
                related_goal_id=related_goal_id,
                after_timestamp=parsed_ts,
                limit=limit,
                offset=offset,
            )

            return PersonaListResponse(
                entries=entries,
                total=total,
                limit=limit,
                offset=offset,
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameter: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error listing persona entries: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list persona entries: {str(e)}",
            )

    @app.get(
        "/persona/entries/{entry_id}",
        response_model=PersonaEntryFull,
        dependencies=[Depends(verify_token)],
        tags=["persona"],
    )
    async def get_persona_entry(entry_id: str) -> PersonaEntryFull:
        """Get a specific persona entry by ID.

        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            entries, _ = _get_persona_entries(limit=1000)
            for entry in entries:
                if entry.entry_id == entry_id:
                    return entry

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Persona entry not found: {entry_id}",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting persona entry: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get persona entry: {str(e)}",
            )

    @app.patch(
        "/persona/entries/{entry_id}",
        response_model=PersonaEntryFull,
        dependencies=[Depends(verify_token)],
        tags=["persona"],
    )
    async def update_persona_entry(
        entry_id: str,
        request: PersonaEntryUpdate,
    ) -> PersonaEntryFull:
        """Update a persona entry (creates new CWM state, preserves history).

        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            from datetime import timezone
            from uuid import uuid4
            from sophia.cwm_a.state_service import CWMState

            # Find existing entry
            entries, _ = _get_persona_entries(limit=1000)
            existing = None
            for entry in entries:
                if entry.entry_id == entry_id:
                    existing = entry
                    break

            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Persona entry not found: {entry_id}",
                )

            # Merge updates
            updated_entry = existing.model_copy(
                update={
                    k: v
                    for k, v in request.model_dump(exclude_unset=True).items()
                    if v is not None
                }
            )

            # Create new CWM state with updated data
            state_id = f"cwm_e_{uuid4().hex[:12]}"
            timestamp = datetime.now(timezone.utc)

            cwm_state = CWMState(
                state_id=state_id,
                model_type="CWM_E",
                timestamp=timestamp,
                data={
                    "entry": {
                        "entry_id": entry_id,
                        "entry_type": updated_entry.entry_type,
                        "content": updated_entry.content,
                        "summary": updated_entry.summary,
                        "trigger": updated_entry.trigger,
                        "sentiment": updated_entry.sentiment,
                        "confidence": updated_entry.confidence,
                        "related_process_ids": updated_entry.related_process_ids,
                        "related_goal_ids": updated_entry.related_goal_ids,
                        "emotion_tags": updated_entry.emotion_tags,
                        "metadata": updated_entry.metadata,
                    },
                    "source": "persona_api",
                    "derivation": "observed",
                    "confidence": updated_entry.confidence or 1.0,
                    "tags": [f"entry_type:{updated_entry.entry_type}"],
                    "links": {},
                },
            )

            _cwm_persistence.persist(cwm_state)

            # Update timestamp on returned entry
            updated_entry.timestamp = timestamp

            logger.info(f"Updated persona entry: {entry_id}")
            return updated_entry

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating persona entry: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update persona entry: {str(e)}",
            )

    @app.delete(
        "/persona/entries/{entry_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(verify_token)],
        tags=["persona"],
    )
    async def delete_persona_entry(entry_id: str) -> None:
        """Delete a persona entry (soft delete via tombstone).

        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            from datetime import timezone
            from uuid import uuid4
            from sophia.cwm_a.state_service import CWMState

            # Find existing entry
            entries, _ = _get_persona_entries(limit=1000)
            existing = None
            for entry in entries:
                if entry.entry_id == entry_id:
                    existing = entry
                    break

            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Persona entry not found: {entry_id}",
                )

            # Create tombstone CWM state
            state_id = f"cwm_e_{uuid4().hex[:12]}"
            timestamp = datetime.now(timezone.utc)

            cwm_state = CWMState(
                state_id=state_id,
                model_type="CWM_E",
                timestamp=timestamp,
                data={
                    "entry": {
                        "entry_id": entry_id,
                        "deleted": True,
                    },
                    "source": "persona_api",
                    "derivation": "observed",
                    "confidence": 1.0,
                    "tags": ["deleted"],
                    "links": {},
                },
            )

            _cwm_persistence.persist(cwm_state)

            logger.info(f"Deleted persona entry: {entry_id}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting persona entry: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete persona entry: {str(e)}",
            )

    @app.get(
        "/persona/sentiment",
        response_model=SentimentResponse,
        dependencies=[Depends(verify_token)],
        tags=["persona"],
    )
    async def get_persona_sentiment(
        limit: int = Query(
            20, ge=1, le=100, description="Number of entries to aggregate"
        ),
        after_timestamp: Optional[str] = Query(
            None, description="ISO timestamp filter"
        ),
    ) -> SentimentResponse:
        """Get aggregated sentiment from recent persona entries.

        Requires authentication via Bearer token.
        """
        if not _cwm_persistence:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CWM persistence service not available",
            )

        try:
            # Parse timestamp
            parsed_ts = None
            if after_timestamp:
                parsed_ts = datetime.fromisoformat(after_timestamp)

            entries, _ = _get_persona_entries(
                after_timestamp=parsed_ts,
                limit=limit,
            )

            if not entries:
                return SentimentResponse(
                    sentiment=None,
                    confidence_avg=None,
                    recent_sentiment_trend=None,
                    emotion_distribution={},
                    entry_count=0,
                    last_updated=None,
                )

            # Aggregate sentiment
            sentiment_counts: Dict[str, int] = {}
            confidences: List[float] = []
            emotion_dist: Dict[str, int] = {}

            for entry in entries:
                if entry.sentiment:
                    sentiment_counts[entry.sentiment] = (
                        sentiment_counts.get(entry.sentiment, 0) + 1
                    )
                if entry.confidence is not None:
                    confidences.append(entry.confidence)
                for tag in entry.emotion_tags:
                    emotion_dist[tag] = emotion_dist.get(tag, 0) + 1

            # Most common sentiment
            most_common = (
                max(sentiment_counts, key=lambda k: sentiment_counts[k])
                if sentiment_counts
                else None
            )

            # Average confidence
            confidence_avg = (
                sum(confidences) / len(confidences) if confidences else None
            )

            # Trend: compare first half vs second half
            trend: Optional[Literal["rising", "falling", "stable"]] = None
            if len(entries) >= 4:
                mid = len(entries) // 2
                first_half = entries[
                    mid:
                ]  # Older entries (list is sorted newest first)
                second_half = entries[:mid]  # Newer entries

                def sentiment_score(e: PersonaEntryFull) -> int:
                    if e.sentiment == "positive":
                        return 1
                    elif e.sentiment == "negative":
                        return -1
                    return 0

                first_score = sum(sentiment_score(e) for e in first_half)
                second_score = sum(sentiment_score(e) for e in second_half)

                if second_score > first_score:
                    trend = "rising"
                elif second_score < first_score:
                    trend = "falling"
                else:
                    trend = "stable"

            return SentimentResponse(
                sentiment=most_common,
                confidence_avg=round(confidence_avg, 3) if confidence_avg else None,
                recent_sentiment_trend=trend,
                emotion_distribution=emotion_dist,
                entry_count=len(entries),
                last_updated=entries[0].timestamp if entries else None,
            )

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameter: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error getting persona sentiment: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get persona sentiment: {str(e)}",
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

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
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

        except HTTPException:
            # Let HTTP exceptions pass through with their status codes
            raise
        except ValueError as e:
            # Invalid timestamp format
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid parameter: {str(e)}",
            )
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
            # Let HTTP exceptions pass through with their status codes
            raise
        except Exception as e:
            logger.error(f"Error retrieving media sample: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve media sample: {str(e)}",
            )

    # Add /api/v1/ aliases for all routes (except health which stays at root)
    # This allows clients to use versioned endpoints while maintaining backwards compatibility
    for route in list(app.routes):
        if hasattr(route, "path") and route.path not in [
            "/health",
            "/openapi.json",
            "/docs",
            "/redoc",
        ]:
            # Create versioned route alias
            if hasattr(route, "endpoint"):
                app.add_api_route(
                    f"/api/v1{route.path}",
                    route.endpoint,
                    methods=route.methods if hasattr(route, "methods") else ["GET"],
                    response_model=(
                        route.response_model
                        if hasattr(route, "response_model")
                        else None
                    ),
                    status_code=(
                        route.status_code if hasattr(route, "status_code") else None
                    ),
                    tags=route.tags if hasattr(route, "tags") else None,
                    dependencies=(
                        route.dependencies if hasattr(route, "dependencies") else None
                    ),
                    include_in_schema=False,  # Don't duplicate in OpenAPI docs
                )

    return app


# Create the application instance
app = create_app()
