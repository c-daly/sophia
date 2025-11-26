"""
Common Cypher queries for HCG graph operations.

This module provides parameterized Cypher queries for:
- Finding entities, concepts, states, and processes
- Traversing causal relationships
- Querying graph structure (IS_A, HAS_STATE, PART_OF, etc.)

All queries use parameters to prevent injection attacks.
See Project LOGOS spec: Section 4.1 for relationship types.
"""


class HCGQueries:
    """Collection of common Cypher queries for HCG operations."""

    # ========== Entity Queries ==========

    @staticmethod
    def find_entity_by_uuid() -> str:
        """
        Find an entity by its UUID.

        Parameters:
        - uuid: Entity UUID (string format)

        Returns: Entity node properties
        """
        return """
        MATCH (e:Entity {uuid: $uuid})
        RETURN e
        """

    @staticmethod
    def find_entity_by_name() -> str:
        """
        Find entities by name (case-insensitive partial match).

        Parameters:
        - name: Entity name pattern

        Returns: List of matching Entity nodes
        """
        return """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e
        ORDER BY e.name
        """

    @staticmethod
    def find_all_entities() -> str:
        """
        Find all entities with optional pagination.

        Parameters:
        - skip: Number of records to skip (default 0)
        - limit: Maximum records to return (default 100)

        Returns: List of Entity nodes
        """
        return """
        MATCH (e:Entity)
        RETURN e
        ORDER BY e.name
        SKIP $skip
        LIMIT $limit
        """

    # ========== Concept Queries ==========

    @staticmethod
    def find_concept_by_uuid() -> str:
        """
        Find a concept by its UUID.

        Parameters:
        - uuid: Concept UUID (string format)

        Returns: Concept node properties
        """
        return """
        MATCH (c:Concept {uuid: $uuid})
        RETURN c
        """

    @staticmethod
    def find_concept_by_name() -> str:
        """
        Find a concept by exact name.

        Parameters:
        - name: Concept name (exact match)

        Returns: Concept node properties
        """
        return """
        MATCH (c:Concept {name: $name})
        RETURN c
        """

    @staticmethod
    def find_all_concepts() -> str:
        """
        Find all concepts.

        Returns: List of Concept nodes
        """
        return """
        MATCH (c:Concept)
        RETURN c
        ORDER BY c.name
        """

    # ========== State Queries ==========

    @staticmethod
    def find_state_by_uuid() -> str:
        """
        Find a state by its UUID.

        Parameters:
        - uuid: State UUID (string format)

        Returns: State node properties
        """
        return """
        MATCH (s:State {uuid: $uuid})
        RETURN s
        """

    @staticmethod
    def find_states_by_timestamp_range() -> str:
        """
        Find states within a timestamp range.

        Parameters:
        - start_time: Start timestamp (ISO format string)
        - end_time: End timestamp (ISO format string)

        Returns: List of State nodes ordered by timestamp
        """
        return """
        MATCH (s:State)
        WHERE s.timestamp >= datetime($start_time)
          AND s.timestamp <= datetime($end_time)
        RETURN s
        ORDER BY s.timestamp
        """

    # ========== Process Queries ==========

    @staticmethod
    def find_process_by_uuid() -> str:
        """
        Find a process by its UUID.

        Parameters:
        - uuid: Process UUID (string format)

        Returns: Process node properties
        """
        return """
        MATCH (p:Process {uuid: $uuid})
        RETURN p
        """

    @staticmethod
    def find_processes_by_time_range() -> str:
        """
        Find processes within a time range.

        Parameters:
        - start_time: Start timestamp (ISO format string)
        - end_time: End timestamp (ISO format string)

        Returns: List of Process nodes ordered by start_time
        """
        return """
        MATCH (p:Process)
        WHERE p.start_time >= datetime($start_time)
          AND p.start_time <= datetime($end_time)
        RETURN p
        ORDER BY p.start_time
        """

    # ========== Relationship Queries ==========

    @staticmethod
    def get_entity_type() -> str:
        """
        Get the concept/type of an entity via IS_A relationship.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: Concept node that the entity is an instance of
        """
        return """
        MATCH (e:Entity {uuid: $entity_uuid})-[:IS_A]->(c:Concept)
        RETURN c
        """

    @staticmethod
    def get_entity_states() -> str:
        """
        Get all states of an entity via HAS_STATE relationship.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: List of State nodes ordered by timestamp (most recent first)
        """
        return """
        MATCH (e:Entity {uuid: $entity_uuid})-[:HAS_STATE]->(s:State)
        RETURN s
        ORDER BY s.timestamp DESC
        """

    @staticmethod
    def get_entity_current_state() -> str:
        """
        Get the most recent state of an entity.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: Most recent State node
        """
        return """
        MATCH (e:Entity {uuid: $entity_uuid})-[:HAS_STATE]->(s:State)
        RETURN s
        ORDER BY s.timestamp DESC
        LIMIT 1
        """

    @staticmethod
    def get_entity_parts() -> str:
        """
        Get all parts of an entity via PART_OF relationship.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: List of Entity nodes that are parts of the given entity
        """
        return """
        MATCH (part:Entity)-[:PART_OF]->(whole:Entity {uuid: $entity_uuid})
        RETURN part
        ORDER BY part.name
        """

    @staticmethod
    def get_entity_parent() -> str:
        """
        Get the parent entity via PART_OF relationship.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: Parent Entity node
        """
        return """
        MATCH (part:Entity {uuid: $entity_uuid})-[:PART_OF]->(whole:Entity)
        RETURN whole
        """

    # ========== Causal Traversal Queries ==========

    @staticmethod
    def traverse_causality_forward(max_depth: int = 10) -> str:
        """
        Traverse causality chain forward from a state.
        Find all states that are caused (directly or indirectly) by processes
        that are caused by the given state.

        Parameters:
        - state_uuid: Starting State UUID (string format)
        - max_depth: Maximum traversal depth (default 10)

        Returns: List of (Process, State) pairs showing causal chain
        """
        return f"""
        MATCH path = (s:State {{uuid: $state_uuid}})<-[:REQUIRES*0..1]-
                     (p:Process)-[:CAUSES*1..{max_depth}]->(result:State)
        RETURN p, result, length(path) as depth
        ORDER BY depth, result.timestamp
        """

    @staticmethod
    def traverse_causality_backward(max_depth: int = 10) -> str:
        """
        Traverse causality chain backward from a state.
        Find all states that caused (directly or indirectly) the given state.

        Parameters:
        - state_uuid: Target State UUID (string format)
        - max_depth: Maximum traversal depth (default 10)

        Returns: List of (State, Process) pairs showing causal history
        """
        return f"""
        MATCH path = (cause:State)<-[:CAUSES*1..{max_depth}]-(p:Process)
                     -[:REQUIRES*0..1]->(s:State {{uuid: $state_uuid}})
        RETURN cause, p, length(path) as depth
        ORDER BY depth DESC, cause.timestamp DESC
        """

    @staticmethod
    def get_process_causes() -> str:
        """
        Get all states caused by a process.

        Parameters:
        - process_uuid: Process UUID (string format)

        Returns: List of State nodes caused by the process
        """
        return """
        MATCH (p:Process {uuid: $process_uuid})-[:CAUSES]->(s:State)
        RETURN s
        ORDER BY s.timestamp
        """

    @staticmethod
    def get_process_requirements() -> str:
        """
        Get all states required by a process (preconditions).

        Parameters:
        - process_uuid: Process UUID (string format)

        Returns: List of State nodes required by the process
        """
        return """
        MATCH (p:Process {uuid: $process_uuid})-[:REQUIRES]->(s:State)
        RETURN s
        ORDER BY s.timestamp
        """

    @staticmethod
    def get_state_effects() -> str:
        """
        Get all states that follow from this state via temporal ordering.

        Parameters:
        - state_uuid: State UUID (string format)

        Returns: List of State nodes that follow this state
        """
        return """
        MATCH (s:State {uuid: $state_uuid})-[:PRECEDES]->(next:State)
        RETURN next
        ORDER BY next.timestamp
        """

    # ========== Spatial/Physical Queries ==========

    @staticmethod
    def get_entity_location() -> str:
        """
        Get the location of an entity via LOCATED_AT relationship.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: Entity node where the given entity is located
        """
        return """
        MATCH (e:Entity {uuid: $entity_uuid})-[:LOCATED_AT]->(loc:Entity)
        RETURN loc
        """

    @staticmethod
    def get_entities_at_location() -> str:
        """
        Get all entities at a specific location.

        Parameters:
        - location_uuid: Location Entity UUID (string format)

        Returns: List of Entity nodes at the location
        """
        return """
        MATCH (e:Entity)-[:LOCATED_AT]->(loc:Entity {uuid: $location_uuid})
        RETURN e
        ORDER BY e.name
        """

    @staticmethod
    def get_entity_attachments() -> str:
        """
        Get all entities attached to this entity.

        Parameters:
        - entity_uuid: Entity UUID (string format)

        Returns: List of Entity nodes attached to the given entity
        """
        return """
        MATCH (attached:Entity)-[:ATTACHED_TO]->(e:Entity {uuid: $entity_uuid})
        RETURN attached
        ORDER BY attached.name
        """

    # ========== Utility Queries ==========

    @staticmethod
    def get_node_relationships() -> str:
        """
        Get all relationships for a node (any type).

        Parameters:
        - uuid: Node UUID (string format)

        Returns: List of relationship types, target nodes, and directions
        """
        return """
        MATCH (n {uuid: $uuid})
        OPTIONAL MATCH (n)-[r]->(target)
        WITH n, collect({type: type(r), direction: 'outgoing', node: target}) as outgoing
        OPTIONAL MATCH (source)-[r2]->(n)
        WITH n, outgoing, collect({type: type(r2), direction: 'incoming', node: source}) as incoming
        RETURN n, outgoing + incoming as relationships
        """

    @staticmethod
    def count_nodes_by_type() -> str:
        """
        Count nodes by type.

        Returns: Counts for each node type
        """
        return """
        OPTIONAL MATCH (e:Entity)
        WITH count(e) as entity_count
        OPTIONAL MATCH (c:Concept)
        WITH entity_count, count(c) as concept_count
        OPTIONAL MATCH (s:State)
        WITH entity_count, concept_count, count(s) as state_count
        OPTIONAL MATCH (p:Process)
        RETURN entity_count, concept_count, state_count, count(p) as process_count
        """
