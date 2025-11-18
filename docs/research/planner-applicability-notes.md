# Causal Reasoning Applicability Notes for Planner (#157)

**Quick Reference Guide**  
**Related**: [Full Survey](causal-reasoning-methods.md)

---

## TL;DR

Implement causal reasoning in Sophia Planner using a **hybrid approach**:
1. **Start**: Backward chaining for goal decomposition
2. **Enhance**: Add causal graph annotations to knowledge graph
3. **Extend**: Include counterfactual evaluation for robust planning

---

## Method Quick Comparison

| When to Use | Method | Complexity | Best For |
|-------------|--------|------------|----------|
| **Phase 1** | Backward Chaining | Low | Goal decomposition, task planning |
| **Phase 1** | Forward Chaining | Low-Medium | Execution monitoring, reactive planning |
| **Phase 2** | Causal Graphs | Medium | Effect prediction, explainable decisions |
| **Phase 3** | Counterfactuals | Medium-High | Alternative evaluation, learning |

---

## Implementation Priority

### ✅ Phase 1: Core Planning (Immediate)

**Backward Chaining Implementation**:
```python
class Planner:
    def decompose_goal(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down high-level goal into actionable steps.
        
        Algorithm:
        1. Query knowledge graph for actions that achieve goal
        2. For each action, identify prerequisites
        3. Recursively decompose prerequisites
        4. Return ordered action sequence
        """
        pass
```

**Forward Chaining Implementation**:
```python
class Planner:
    def execute_step(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one planning step from current state.
        
        Algorithm:
        1. Query knowledge graph for applicable actions
        2. Evaluate which actions move toward goal
        3. Apply selected action
        4. Return new state
        """
        pass
```

**Knowledge Graph Integration**:
- Use existing `KnowledgeGraph` class
- Add action/state nodes
- Use edges to represent action effects and prerequisites

---

### ⚡ Phase 2: Causal Enhancement (Near-term)

**Extend Edge Properties**:
```python
# Add causal semantics to edges
Edge(
    source="action_id",
    target="state_id", 
    relation="causes",  # or "enables", "requires", "prevents"
    properties={
        "strength": 0.9,      # Causal strength [0-1]
        "delay": "immediate", # Temporal aspect
        "certainty": 0.95     # Probabilistic
    }
)
```

**Causal Path Queries**:
```python
class Planner:
    def find_causal_path(self, source: str, target: str) -> List[Edge]:
        """Find causal path from source to target in knowledge graph."""
        pass
    
    def predict_effects(self, action: Dict[str, Any], 
                       state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict effects of action using causal graph."""
        pass
```

---

### 🚀 Phase 3: Counterfactual Reasoning (Medium-term)

**Alternative Evaluation**:
```python
class Planner:
    def evaluate_alternatives(self, 
                             goal: Dict[str, Any],
                             actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate multiple action alternatives without execution.
        
        Returns: Ranked list of (action, predicted_outcome, confidence)
        """
        pass
```

**Robust Planning**:
```python
class Planner:
    def add_contingencies(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add fallback actions for potential failures.
        
        Process:
        1. Generate failure scenarios for each step
        2. Identify critical failure points
        3. Add conditional branches with alternatives
        """
        pass
```

---

## Integration with Existing Code

### Current Planner Structure
```python
class Planner:
    def __init__(self):
        self._goals: List[Dict[str, Any]] = []
    
    # Existing methods
    def add_goal(self, goal) -> None: ...
    def get_goals(self) -> List: ...
    def clear_goals(self) -> None: ...
    def has_goals(self) -> bool: ...
```

### Recommended Extensions

**Add Knowledge Graph Reference**:
```python
class Planner:
    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None):
        self._goals: List[Dict[str, Any]] = []
        self._kg = knowledge_graph or KnowledgeGraph()
        self._current_plans: Dict[str, List[Dict[str, Any]]] = {}
```

**Add Planning Methods**:
```python
    def plan(self, goal: Dict[str, Any], 
             strategy: str = "backward") -> List[Dict[str, Any]]:
        """Create plan to achieve goal.
        
        Args:
            goal: Goal specification
            strategy: "backward", "forward", or "hybrid"
        
        Returns:
            Ordered list of actions
        """
        if strategy == "backward":
            return self._backward_chain(goal)
        elif strategy == "forward":
            return self._forward_chain(goal)
        else:  # hybrid
            return self._hybrid_plan(goal)
```

---

## Code Examples

### Example 1: Simple Backward Chaining

```python
# Define goal
goal = {
    "type": "achieve_state",
    "state": "document_written",
    "properties": {"topic": "causal_reasoning"}
}

# Create planner with knowledge graph
kg = KnowledgeGraph()

# Add domain knowledge
kg.add_node(Node(id="write_outline", type="action"))
kg.add_node(Node(id="research_topic", type="action"))
kg.add_node(Node(id="document_written", type="state"))
kg.add_node(Node(id="outline_exists", type="state"))

kg.add_edge(Edge(
    source="write_outline",
    target="outline_exists",
    relation="achieves"
))
kg.add_edge(Edge(
    source="research_topic",
    target="write_outline",
    relation="enables"
))

# Plan
planner = Planner(knowledge_graph=kg)
plan = planner.plan(goal, strategy="backward")
# Result: [research_topic, write_outline, write_document]
```

### Example 2: Causal Graph with Strength

```python
# Add causal strengths for better planning
kg.add_edge(Edge(
    source="action_A",
    target="goal_state",
    relation="causes",
    properties={"strength": 0.9}  # High likelihood
))
kg.add_edge(Edge(
    source="action_B",
    target="goal_state",
    relation="causes",
    properties={"strength": 0.3}  # Low likelihood
))

# Planner can now select action_A as more reliable
```

### Example 3: Counterfactual Evaluation

```python
# Evaluate alternatives before committing
goal = {"state": "task_complete"}
current_state = {"resources": ["A", "B"], "time": 100}

alternatives = [
    {"action": "approach_1", "cost": 10},
    {"action": "approach_2", "cost": 5},
    {"action": "approach_3", "cost": 15}
]

# Evaluate without executing
ranked = planner.evaluate_alternatives(goal, alternatives)
# Returns: [(approach_2, outcome_pred, 0.85), ...]

# Select best approach
best_action = ranked[0]
```

---

## Key Design Decisions

### 1. Graph Representation
- **Decision**: Use existing `KnowledgeGraph` class for all causal models
- **Rationale**: Reuse infrastructure, maintain consistency
- **Trade-off**: May need to extend edge/node properties

### 2. Planning Strategy
- **Decision**: Support multiple strategies (backward, forward, hybrid)
- **Rationale**: Different tasks benefit from different approaches
- **Trade-off**: More complex API, but more flexible

### 3. Causal Strength Encoding
- **Decision**: Store as edge property (0-1 float)
- **Rationale**: Simple, intuitive, compatible with probabilities
- **Trade-off**: May need richer uncertainty representation later

### 4. Counterfactual Evaluation
- **Decision**: Implement as separate evaluation method, not automatic
- **Rationale**: Computationally expensive, user should control when used
- **Trade-off**: More manual control needed

---

## Testing Strategy

### Unit Tests for Phase 1
```python
def test_backward_chaining():
    """Test basic goal decomposition."""
    planner = Planner()
    goal = {"state": "goal_achieved"}
    plan = planner.plan(goal, strategy="backward")
    assert len(plan) > 0
    assert plan[-1]["achieves"] == "goal_achieved"

def test_forward_chaining():
    """Test forward progression from initial state."""
    planner = Planner()
    state = {"current": "initial"}
    goal = {"state": "goal"}
    plan = planner.plan(goal, strategy="forward")
    # Verify plan reaches goal from initial state
```

### Integration Tests for Phase 2
```python
def test_causal_path_finding():
    """Test finding causal paths in knowledge graph."""
    kg = create_test_knowledge_graph()
    planner = Planner(knowledge_graph=kg)
    path = planner.find_causal_path("action_A", "goal_state")
    assert len(path) > 0
    assert path[0].source == "action_A"
    assert path[-1].target == "goal_state"

def test_effect_prediction():
    """Test predicting action effects."""
    planner = Planner(knowledge_graph=kg)
    action = {"id": "test_action"}
    state = {"value": 1}
    predicted = planner.predict_effects(action, state)
    assert "value" in predicted
```

### Performance Tests for Phase 3
```python
def test_counterfactual_efficiency():
    """Ensure counterfactual evaluation scales reasonably."""
    planner = Planner()
    alternatives = [{"action": f"a_{i}"} for i in range(100)]
    
    import time
    start = time.time()
    results = planner.evaluate_alternatives(goal, alternatives)
    duration = time.time() - start
    
    # Should complete in reasonable time (e.g., < 1 second for 100 alternatives)
    assert duration < 1.0
```

---

## Performance Considerations

### Computational Complexity
- **Backward Chaining**: O(b^d) - exponential in branching factor and depth
- **Forward Chaining**: O(a * s) - actions × state space size  
- **Causal Path Query**: O(V + E) - graph traversal
- **Counterfactual Evaluation**: O(k * c) - k alternatives × cost per evaluation

### Optimization Strategies
1. **Memoization**: Cache plan fragments for reuse
2. **Pruning**: Use heuristics to cut search space
3. **Lazy Evaluation**: Compute causal paths on-demand
4. **Parallel Evaluation**: Evaluate counterfactuals concurrently
5. **Hierarchical Abstraction**: Plan at multiple levels of detail

---

## Common Pitfalls & Solutions

### Pitfall 1: Infinite Recursion in Backward Chaining
**Problem**: Circular dependencies cause infinite loop  
**Solution**: Maintain visited set, detect cycles, break at max depth

### Pitfall 2: Causal Graphs with Cycles
**Problem**: Feedback loops violate DAG assumption  
**Solution**: Use temporal ordering or separate cycles into components

### Pitfall 3: Counterfactual Explosion
**Problem**: Too many alternatives to evaluate  
**Solution**: Use pruning, sample strategies, or heuristic filtering

### Pitfall 4: Stale Causal Models
**Problem**: Causal beliefs become outdated  
**Solution**: Implement learning mechanisms, periodic updates

---

## Next Steps After R1

1. **Design Detailed API** for Planner extensions
2. **Create Prototype** of backward chaining in sandbox
3. **Gather Domain Knowledge** for initial causal graph
4. **Define Test Scenarios** for planning tasks
5. **Implement Phase 1** (backward/forward chaining)
6. **Iterate Based on Usage** and performance metrics

---

## Questions for Discussion

1. Should we support probabilistic planning from the start, or add later?
2. What's the right balance between planning time and plan quality?
3. Should counterfactual evaluation be automatic or manual?
4. How do we handle planning under uncertainty?
5. What's the interface between Planner and Executor components?

---

## Resources

- **Full Survey**: [causal-reasoning-methods.md](causal-reasoning-methods.md)
- **Current Planner Code**: `src/sophia/planner/planner.py`
- **Knowledge Graph Code**: `src/sophia/knowledge_graph/`
- **Related Issue**: #157 (Planner implementation)

---

**Last Updated**: November 2025  
**Status**: Ready for implementation planning  
**Owner**: Planner development team
