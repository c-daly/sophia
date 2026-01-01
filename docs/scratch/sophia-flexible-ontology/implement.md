# Implementation: Sophia Flexible Ontology Alignment

## Tasks

### Phase 1: Independent changes (can parallelize)
- [ ] Update HCGClient.add_node() - new signature, Cypher query
- [ ] Update knowledge_graph/node.py - add uuid, name, ancestors, is_type_definition
- [ ] Update test_data_pick_and_place.cypher - convert to :Node with standard props

### Phase 2: Dependent changes (sequential after Phase 1)
- [ ] Update HCGClient other methods - get_node, delete_node, query_neighbors, etc.
- [ ] Update seeder.py - all add_node calls with new signature
- [ ] Update tests/data/__init__.py - Node usage with new fields
- [ ] Update test_client_wrapper.py - tests for new signature
- [ ] Add add_node_legacy() for backward compat

## Progress Log
(Updated as work progresses)
