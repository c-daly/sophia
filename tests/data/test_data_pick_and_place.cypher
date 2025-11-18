// Test data for pick-and-place scenario
// This cypher file defines the Hierarchical Cognitive Graph (HCG) for a simple pick-and-place task

// Create spatial entities
CREATE (table:Location {id: 'table', name: 'Table', type: 'surface'})
CREATE (bin:Location {id: 'bin', name: 'Bin', type: 'container'})

// Create objects
CREATE (red_block:Object {id: 'red_block', name: 'Red Block', color: 'red', type: 'block'})
CREATE (blue_block:Object {id: 'blue_block', name: 'Blue Block', color: 'blue', type: 'block'})

// Initial state: blocks on table
CREATE (red_block)-[:LOCATED_AT]->(table)
CREATE (blue_block)-[:LOCATED_AT]->(table)

// Create action primitives
CREATE (move1:Action {id: 'move_to_red_block', name: 'Move to Red Block', type: 'MOVE', target: 'red_block'})
CREATE (grasp1:Action {id: 'grasp_red_block', name: 'Grasp Red Block', type: 'GRASP', target: 'red_block'})
CREATE (move2:Action {id: 'move_to_bin', name: 'Move to Bin', type: 'MOVE', target: 'bin'})
CREATE (release1:Action {id: 'release_red_block', name: 'Release Red Block', type: 'RELEASE', target: 'red_block'})

// Action preconditions and effects
CREATE (move1)-[:ENABLES]->(grasp1)
CREATE (grasp1)-[:ENABLES]->(move2)
CREATE (move2)-[:ENABLES]->(release1)
CREATE (release1)-[:ACHIEVES {state: 'red_block_in_bin'}]->(bin)

// Goal definition
CREATE (goal:Goal {id: 'goal_red_block_in_bin', description: 'red block in bin', target_state: 'red_block_in_bin'})
CREATE (goal)-[:REQUIRES]->(release1)
