// Test data for pick-and-place scenario
// This cypher file defines the Hierarchical Cognitive Graph (HCG) for a simple pick-and-place task
// Uses logos flexible ontology: :Node label with uuid, name, type, is_type_definition, ancestors
// Flat hierarchy: all types are direct children of root

// Create spatial entities (locations)
CREATE (table:Node {
    uuid: 'table',
    name: 'Table',
    type: 'location',
    is_type_definition: false,
    ancestors: ['root'],
    location_type: 'surface'
})
CREATE (bin:Node {
    uuid: 'bin',
    name: 'Bin',
    type: 'location',
    is_type_definition: false,
    ancestors: ['root'],
    location_type: 'container'
})

// Create objects
CREATE (red_block:Node {
    uuid: 'red_block',
    name: 'Red Block',
    type: 'object',
    is_type_definition: false,
    ancestors: ['root'],
    color: 'red',
    object_type: 'block'
})
CREATE (blue_block:Node {
    uuid: 'blue_block',
    name: 'Blue Block',
    type: 'object',
    is_type_definition: false,
    ancestors: ['root'],
    color: 'blue',
    object_type: 'block'
})

// Initial state: blocks on table
CREATE (red_block)-[:LOCATED_AT]->(table)
CREATE (blue_block)-[:LOCATED_AT]->(table)

// Create action primitives
CREATE (move1:Node {
    uuid: 'move_to_red_block',
    name: 'Move to Red Block',
    type: 'action',
    is_type_definition: false,
    ancestors: ['root'],
    action_type: 'MOVE',
    target: 'red_block'
})
CREATE (grasp1:Node {
    uuid: 'grasp_red_block',
    name: 'Grasp Red Block',
    type: 'action',
    is_type_definition: false,
    ancestors: ['root'],
    action_type: 'GRASP',
    target: 'red_block'
})
CREATE (move2:Node {
    uuid: 'move_to_bin',
    name: 'Move to Bin',
    type: 'action',
    is_type_definition: false,
    ancestors: ['root'],
    action_type: 'MOVE',
    target: 'bin'
})
CREATE (release1:Node {
    uuid: 'release_red_block',
    name: 'Release Red Block',
    type: 'action',
    is_type_definition: false,
    ancestors: ['root'],
    action_type: 'RELEASE',
    target: 'red_block'
})

// Action preconditions and effects
CREATE (move1)-[:ENABLES]->(grasp1)
CREATE (grasp1)-[:ENABLES]->(move2)
CREATE (move2)-[:ENABLES]->(release1)
CREATE (release1)-[:ACHIEVES {state: 'red_block_in_bin'}]->(bin)

// Goal definition
CREATE (goal:Node {
    uuid: 'goal_red_block_in_bin',
    name: 'Goal: Red Block in Bin',
    type: 'goal',
    is_type_definition: false,
    ancestors: ['root'],
    description: 'red block in bin',
    target_state: 'red_block_in_bin'
})
CREATE (goal)-[:REQUIRES]->(release1)
