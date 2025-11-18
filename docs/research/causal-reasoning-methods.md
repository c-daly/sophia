# Survey of Causal Reasoning Methods for HCG Planning

**Document**: R1 Research Survey  
**Target Component**: Planner (#157)  
**Date**: November 2025  
**Purpose**: Survey causal reasoning methods applicable to Hierarchical Cognitive Graph (HCG) planning

---

## Executive Summary

This document surveys three primary causal reasoning methods applicable to cognitive planning systems:
1. **Backward & Forward Chaining** - Rule-based inference strategies
2. **Causal Graphs** - Structural representation of causal relationships
3. **Counterfactual Reasoning** - What-if analysis and alternative scenarios

Each method is analyzed for its applicability to the Sophia Planner component, with practical recommendations for implementation in HCG planning contexts.

---

## 1. Backward and Forward Chaining

### 1.1 Overview

Chaining methods are fundamental inference strategies in rule-based systems that navigate relationships between goals and facts.

#### Backward Chaining (Goal-Driven)
- **Strategy**: Start from desired goals and work backwards to determine what conditions/actions are needed
- **Process**: 
  1. Begin with target goal state
  2. Identify rules/actions that can achieve this goal
  3. Recursively determine prerequisites for those actions
  4. Continue until reaching known facts or executable actions
- **Also known as**: Goal-directed reasoning, regression planning

#### Forward Chaining (Data-Driven)
- **Strategy**: Start from current state/facts and apply rules forward to reach goals
- **Process**:
  1. Begin with current state and available facts
  2. Identify applicable rules/actions given current state
  3. Apply actions and update state
  4. Continue until goal is achieved or no more rules apply
- **Also known as**: Data-driven reasoning, progression planning

### 1.2 Comparison

| Aspect | Backward Chaining | Forward Chaining |
|--------|------------------|------------------|
| Direction | Goal → Prerequisites | Facts → Consequences |
| Efficiency | More focused on relevant paths | May explore irrelevant branches |
| Use Case | Well-defined goals | Exploratory/discovery tasks |
| Completeness | May miss alternative solutions | Explores more possibilities |
| Cognitive Load | Lower (goal-focused) | Higher (considers all facts) |

### 1.3 Applicability to Sophia Planner

**Strengths for HCG Planning:**
- Natural fit for task decomposition in hierarchical graphs
- Backward chaining aligns with goal-based cognitive architectures
- Forward chaining useful for reactive planning and state updates
- Both can leverage knowledge graph structure for rule representation

**Implementation Considerations:**
1. **Graph Structure**: Represent actions/rules as edges in knowledge graph
   - Nodes: States, goals, preconditions
   - Edges: Actions, implications, causal links
   
2. **Backward Chaining for Planner**:
   ```
   Goal: Achieve state G
   → Query graph for edges: [precondition] --action--> [G]
   → Recursively plan for preconditions
   → Build action sequence bottom-up
   ```

3. **Forward Chaining for Planner**:
   ```
   Current state: S
   → Query graph for applicable actions from S
   → Apply action, update state
   → Check if goal reached
   → Repeat
   ```

4. **Hybrid Approach** (Recommended):
   - Use backward chaining for initial goal decomposition
   - Use forward chaining for execution monitoring and replanning
   - Combine both for bidirectional search in complex spaces

**Integration with Planner (#157):**
- Extend `Planner` class with `decompose_goal()` method (backward)
- Add `execute_step()` method for forward simulation
- Implement rule/action representation in knowledge graph
- Add planning algorithm selection based on task type

**Challenges:**
- Cycle detection in hierarchical goal decomposition
- Handling non-deterministic actions
- Managing computational complexity with large state spaces
- Balancing exploration vs. exploitation

---

## 2. Causal Graphs

### 2.1 Overview

Causal graphs are directed acyclic graphs (DAGs) that explicitly represent causal relationships between variables or events.

**Key Concepts:**
- **Nodes**: Variables, events, or states
- **Directed Edges**: Direct causal influence (X → Y means "X causes Y")
- **Path**: Sequence of causal links showing indirect influence
- **Structural Causal Models (SCM)**: Mathematical framework combining graphs with equations

### 2.2 Types of Causal Relationships

1. **Direct Causation**: X → Y (X directly causes Y)
2. **Indirect Causation**: X → Z → Y (X causes Y through Z)
3. **Common Cause**: X ← Z → Y (Z causes both X and Y)
4. **Common Effect**: X → Z ← Y (Both X and Y cause Z)
5. **Confounding**: Hidden common causes affecting relationships

### 2.3 Causal Graph Operations

**d-separation**: Determining conditional independence
- Crucial for identifying which variables influence planning outcomes
- Helps identify relevant factors for decision-making

**Intervention (do-calculus)**:
- Modeling actions: do(X=x) sets X to value x
- Different from observation: seeing X=x vs. making X=x
- Critical for action planning and prediction

**Markov Blanket**:
- Minimal set of variables needed to predict a target
- Useful for state abstraction in planning

### 2.4 Applicability to Sophia Planner

**Strengths for HCG Planning:**
- Natural representation in knowledge graph infrastructure
- Explicit causal relationships improve explainability
- Supports reasoning about action effects
- Enables prediction of consequences before execution
- Facilitates credit assignment and learning

**Implementation Considerations:**

1. **Graph Encoding**:
   ```python
   # Causal edges in knowledge graph
   Node("action_A")
   Node("state_X")
   Node("state_Y")
   Edge(source="action_A", target="state_X", relation="causes", 
        properties={"strength": 0.8, "delay": "immediate"})
   Edge(source="state_X", target="state_Y", relation="enables",
        properties={"strength": 0.9})
   ```

2. **Causal Inference for Planning**:
   - Use graph structure to predict action outcomes
   - Identify necessary preconditions via parent nodes
   - Detect side effects via descendant nodes
   - Avoid unintended consequences through path analysis

3. **Action Effect Modeling**:
   ```
   Action selection:
   → Identify desired goal state G
   → Query causal graph for actions leading to G
   → Evaluate causal paths for feasibility
   → Select action with strongest causal influence
   → Update graph beliefs after execution
   ```

4. **Causal Discovery** (Future Enhancement):
   - Learn causal structure from execution traces
   - Update graph weights based on observed outcomes
   - Refine causal model over time

**Integration with Planner (#157):**
- Add `CausalModel` class wrapping knowledge graph with causal semantics
- Implement `predict_effects(action, current_state)` method
- Add `find_causal_path(source, target)` for reasoning
- Extend edge properties to include causal strength and type
- Implement intervention queries: `simulate_do(action)`

**Challenges:**
- Distinguishing causation from correlation
- Handling cyclic causal relationships (feedback loops)
- Managing temporal aspects (delayed effects)
- Dealing with stochastic causation
- Computational complexity of large causal graphs

**Advantages Over Pure Rule-Based Systems:**
- More expressive than simple if-then rules
- Better handles indirect effects and side effects
- Supports quantitative reasoning (causal strength)
- Enables what-if analysis (counterfactuals)
- Facilitates transfer learning across similar domains

---

## 3. Counterfactual Reasoning

### 3.1 Overview

Counterfactual reasoning involves reasoning about alternative scenarios: "What would have happened if...?" 

**Core Question**: Given that event E occurred, what would have been different had action A been taken instead of action B?

**Key Components:**
1. **Actual world**: What actually happened
2. **Counterfactual world**: What would have happened under different conditions
3. **Minimal change principle**: Change only what's necessary to evaluate the counterfactual

### 3.2 Types of Counterfactuals

1. **Retrospective**: "If I had taken action A instead of B, would the outcome have been better?"
   - Used for learning and credit assignment
   - Crucial for regret minimization

2. **Prospective**: "If I take action A vs. action B, what will happen?"
   - Used for planning and decision making
   - Evaluates alternatives before committing

3. **Diagnostic**: "What action would have prevented this outcome?"
   - Used for debugging and failure analysis
   - Identifies critical decision points

### 3.3 Counterfactual Inference Process

1. **Abduction**: Determine what must have been true given observations
   - Update beliefs about unobserved variables
   - Infer latent state from observed outcomes

2. **Action/Intervention**: Modify the model to reflect the counterfactual condition
   - Apply do-operator: do(X=x')
   - Change only the intervened variable, not its causes

3. **Prediction**: Compute outcomes in the counterfactual world
   - Propagate changes through causal graph
   - Maintain consistency with abducted values

### 3.4 Applicability to Sophia Planner

**Strengths for HCG Planning:**
- Enables evaluation of alternative plans without execution
- Supports learning from hypothetical scenarios
- Improves decision quality through "mental simulation"
- Facilitates robust planning (considering what could go wrong)
- Enables explanation and justification of decisions

**Implementation Considerations:**

1. **Counterfactual Query Structure**:
   ```python
   # "What if I had chosen action A instead of action B?"
   counterfactual = CounterfactualQuery(
       actual_action="B",
       actual_outcome="failure",
       alternative_action="A",
       context=current_state
   )
   predicted_outcome = planner.evaluate_counterfactual(counterfactual)
   ```

2. **Integration with Causal Graphs**:
   - Use causal graph structure for counterfactual inference
   - Apply Pearl's three-step process: abduction, intervention, prediction
   - Maintain separate graphs for actual vs. counterfactual worlds

3. **Planning Applications**:
   
   **Alternative Plan Evaluation**:
   ```
   Given: Goal G, current state S, candidate actions [A1, A2, A3]
   For each action Ai:
     → Create counterfactual world with do(Ai)
     → Predict resulting state Si'
     → Evaluate proximity to goal G
     → Compute expected value/utility
   Select: Action with best counterfactual outcome
   ```

   **Failure Recovery**:
   ```
   Given: Failed plan P, undesired outcome O
   → Identify decision points in P
   → For each decision point:
       Generate counterfactual: "What if I chose differently?"
       Evaluate alternative outcome
   → Identify critical decisions that led to failure
   → Replan with lessons learned
   ```

   **Robust Planning**:
   ```
   Given: Plan P for goal G
   → Generate adverse counterfactuals: "What if X goes wrong?"
   → Identify vulnerabilities in plan P
   → Add contingency actions for high-risk branches
   → Create robust plan P' that handles failures
   ```

4. **Learning and Adaptation**:
   - Store execution traces with actual outcomes
   - Generate counterfactual alternatives for each decision
   - Update causal model based on counterfactual accuracy
   - Improve future planning through counterfactual learning

**Integration with Planner (#157):**

Add the following capabilities to Planner:

1. **Counterfactual Evaluation**:
   ```python
   def evaluate_alternatives(self, goal: Dict[str, Any], 
                            actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       """Evaluate counterfactual outcomes for alternative actions."""
       # For each action, simulate outcome without executing
       # Return ranked list with predicted outcomes
   ```

2. **Retrospective Analysis**:
   ```python
   def analyze_failure(self, failed_plan: List[Dict[str, Any]], 
                       actual_outcome: Dict[str, Any]) -> Dict[str, Any]:
       """Analyze what went wrong and what could have worked."""
       # Generate counterfactuals for key decision points
       # Identify critical decisions that led to failure
   ```

3. **Contingency Planning**:
   ```python
   def add_contingencies(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       """Add fallback actions for potential failure scenarios."""
       # Generate failure counterfactuals
       # Add conditional branches for robustness
   ```

**Challenges:**
- Computational cost of evaluating multiple counterfactuals
- Maintaining consistency in counterfactual worlds
- Handling uncertainty in counterfactual predictions
- Determining which counterfactuals are worth evaluating
- Balancing exploration of alternatives vs. exploitation of known good plans

**Synergy with Other Methods:**
- **With Chaining**: Use counterfactuals to evaluate alternative decomposition strategies
- **With Causal Graphs**: Leverage causal structure for efficient counterfactual inference
- **Combined**: Use causal graphs for structure, counterfactuals for evaluation, chaining for search

---

## 4. Comparative Analysis

### 4.1 Method Comparison Matrix

| Method | Expressiveness | Computational Cost | Explainability | Learning Support | Uncertainty Handling |
|--------|---------------|-------------------|----------------|------------------|---------------------|
| Backward Chaining | Medium | Low | High | Low | Low |
| Forward Chaining | Medium | Medium | High | Medium | Low |
| Causal Graphs | High | Medium | Very High | High | Medium |
| Counterfactuals | Very High | High | Very High | Very High | High |

### 4.2 Use Case Recommendations

**Use Backward Chaining When:**
- Goals are well-defined and specific
- The domain has clear goal-action relationships
- Computational efficiency is critical
- Explanation of reasoning is important

**Use Forward Chaining When:**
- Starting state is known but goal is exploratory
- Reactive behavior is needed
- Incremental state updates are important
- Discovery of emergent behaviors is desired

**Use Causal Graphs When:**
- Explicit causal relationships exist in the domain
- Understanding action effects is crucial
- Predicting consequences before execution
- Building explainable AI systems
- Domain knowledge can be encoded as causal structures

**Use Counterfactual Reasoning When:**
- Evaluating multiple alternatives is important
- Learning from mistakes is a priority
- Robust planning is required (handling failures)
- Decision justification is needed
- No direct execution is possible (simulation-based planning)

### 4.3 Integration Strategy for Sophia

**Recommended Hybrid Architecture:**

```
Planner Component Architecture:
│
├── Causal Knowledge Base (Foundation)
│   ├── Knowledge Graph with causal edges
│   ├── Causal strength annotations
│   └── Temporal/probabilistic information
│
├── Planning Engine (Core)
│   ├── Backward Chaining Module
│   │   └── Goal decomposition
│   ├── Forward Chaining Module
│   │   └── State progression
│   └── Hybrid Search Strategy
│       └── Bidirectional planning
│
├── Counterfactual Reasoning Module (Enhancement)
│   ├── Alternative evaluation
│   ├── Failure analysis
│   └── Robust plan generation
│
└── Learning & Adaptation (Future)
    ├── Causal structure learning
    ├── Counterfactual accuracy tracking
    └── Model refinement
```

---

## 5. Implementation Roadmap for Planner (#157)

### Phase 1: Foundation (Immediate)
1. **Extend Knowledge Graph**:
   - Add causal edge types to existing graph structure
   - Implement causal strength properties
   - Add temporal annotations

2. **Basic Backward Chaining**:
   - Implement goal decomposition algorithm
   - Add prerequisite resolution
   - Build action sequence generation

3. **Basic Forward Chaining**:
   - Implement state progression
   - Add applicable action detection
   - Build execution monitoring

### Phase 2: Causal Enhancement (Near-term)
1. **Causal Graph Operations**:
   - Implement causal path finding
   - Add intervention (do-calculus) queries
   - Build effect prediction

2. **Causal Planning**:
   - Integrate causal inference into action selection
   - Add side-effect detection
   - Implement precondition inference from causal structure

### Phase 3: Counterfactual Reasoning (Medium-term)
1. **Counterfactual Evaluation**:
   - Implement alternative action simulation
   - Add outcome prediction without execution
   - Build alternative ranking system

2. **Robust Planning**:
   - Add failure scenario generation
   - Implement contingency planning
   - Build robust plan synthesis

### Phase 4: Learning & Adaptation (Long-term)
1. **Causal Discovery**:
   - Learn causal structure from execution traces
   - Update causal weights from outcomes
   - Refine model over time

2. **Counterfactual Learning**:
   - Evaluate counterfactual accuracy
   - Update planning strategies based on errors
   - Implement meta-learning for planning

---

## 6. Recommendations

### 6.1 Immediate Actions for Planner (#157)

1. **Start with Backward Chaining**:
   - Most natural fit for goal-based planning
   - Relatively simple to implement
   - Provides immediate value for task decomposition

2. **Enhance with Causal Annotations**:
   - Extend existing knowledge graph with causal semantics
   - Start simple: just mark edges as "causes" or "enables"
   - Add strength weights as needed

3. **Design for Future Extensions**:
   - Structure code to support counterfactual queries later
   - Keep causal model separate from planning algorithm
   - Use abstraction layers for flexibility

### 6.2 Design Principles

1. **Modularity**: Each reasoning method should be a pluggable component
2. **Composability**: Methods should work together (hybrid approaches)
3. **Explainability**: All reasoning steps should be traceable
4. **Efficiency**: Use heuristics and pruning for large state spaces
5. **Adaptability**: Support learning and model refinement over time

### 6.3 Key Considerations

**Performance**:
- Backward chaining: O(b^d) where b=branching factor, d=depth
- Causal graphs: O(n^2) for path queries in n-node graph
- Counterfactuals: O(k * n^2) for k alternatives
- Use memoization, caching, and pruning extensively

**Scalability**:
- Hierarchical abstraction to manage large state spaces
- Focus search using heuristics and goal-relevance
- Incremental planning rather than full replanning
- Lazy evaluation of causal paths and counterfactuals

**Robustness**:
- Handle incomplete causal knowledge gracefully
- Provide default reasoning when causal info unavailable
- Support probabilistic/uncertain causal relationships
- Include failure recovery mechanisms

---

## 7. Conclusion

All three causal reasoning methods—chaining, causal graphs, and counterfactuals—offer valuable capabilities for HCG planning in Sophia:

- **Backward/Forward Chaining** provides efficient, goal-directed reasoning with clear explainability
- **Causal Graphs** offer rich representation of domain knowledge and enable sophisticated causal inference
- **Counterfactual Reasoning** enables evaluation of alternatives, learning from hypotheticals, and robust planning

**Recommended Approach**: Start with backward chaining as the core planning algorithm, enhance it with causal graph annotations for richer reasoning, and add counterfactual capabilities for robust decision-making and learning.

This hybrid approach leverages the strengths of each method while maintaining computational tractability and explainability—both crucial for cognitive architectures.

---

## References & Further Reading

### Academic Foundations
1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. [Chapters on Planning and Logic]
3. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

### Planning Systems
4. Ghallab, M., Nau, D., & Traverso, P. (2004). *Automated Planning: Theory & Practice*. Morgan Kaufmann.
5. Nau, D. et al. (2003). "SHOP2: An HTN Planning System". *Journal of Artificial Intelligence Research*, 20, 379-404.

### Causal Reasoning in AI
6. Schölkopf, B. et al. (2021). "Toward Causal Representation Learning". *Proceedings of the IEEE*, 109(5), 612-634.
7. Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.

### Counterfactual Reasoning
8. Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
9. Halpern, J. Y. (2016). *Actual Causality*. MIT Press.

### Cognitive Architecture Applications
10. Laird, J. E. (2012). *The Soar Cognitive Architecture*. MIT Press.
11. Anderson, J. R. (2007). *How Can the Human Mind Occur in the Physical Universe?*. Oxford University Press.

---

**Document Status**: Complete  
**Next Steps**: Implementation planning for Planner component (#157)  
**Review Date**: To be scheduled with team
