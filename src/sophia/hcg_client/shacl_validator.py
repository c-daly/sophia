"""SHACL validator for HCG graph mutations."""

from typing import Dict, Any, List, Optional
from pyshacl import validate
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SH


class SHACLValidator:
    """SHACL validator for knowledge graph mutations.
    
    Enforces constraints on graph mutations using SHACL shapes.
    """

    def __init__(self, shapes_graph: Optional[str] = None) -> None:
        """Initialize SHACL validator.
        
        Args:
            shapes_graph: Optional SHACL shapes graph in Turtle format.
                        If None, uses default HCG shapes.
        """
        if shapes_graph:
            self._shapes = Graph()
            self._shapes.parse(data=shapes_graph, format="turtle")
        else:
            self._shapes = self._create_default_shapes()
    
    def _create_default_shapes(self) -> Graph:
        """Create default SHACL shapes for HCG validation.
        
        Returns:
            RDF graph with default SHACL shapes
        """
        g = Graph()
        ex = Namespace("http://example.org/hcg/")
        
        # Node shape: All nodes must have a type
        node_shape = ex.NodeShape
        g.add((node_shape, RDF.type, SH.NodeShape))
        g.add((node_shape, SH.targetClass, ex.Node))
        g.add((node_shape, SH.property, ex.NodeTypeProperty))
        
        # Type property shape
        type_prop = ex.NodeTypeProperty
        g.add((type_prop, RDF.type, SH.PropertyShape))
        g.add((type_prop, SH.path, ex.nodeType))
        g.add((type_prop, SH.minCount, Literal(1)))
        g.add((type_prop, SH.maxCount, Literal(1)))
        g.add((type_prop, SH.datatype, RDFS.Literal))
        
        # Edge shape: All edges must have source, target, and relation
        edge_shape = ex.EdgeShape
        g.add((edge_shape, RDF.type, SH.NodeShape))
        g.add((edge_shape, SH.targetClass, ex.Edge))
        
        # Source property
        g.add((edge_shape, SH.property, ex.EdgeSourceProperty))
        source_prop = ex.EdgeSourceProperty
        g.add((source_prop, RDF.type, SH.PropertyShape))
        g.add((source_prop, SH.path, ex.source))
        g.add((source_prop, SH.minCount, Literal(1)))
        g.add((source_prop, SH.maxCount, Literal(1)))
        
        # Target property
        g.add((edge_shape, SH.property, ex.EdgeTargetProperty))
        target_prop = ex.EdgeTargetProperty
        g.add((target_prop, RDF.type, SH.PropertyShape))
        g.add((target_prop, SH.path, ex.target))
        g.add((target_prop, SH.minCount, Literal(1)))
        g.add((target_prop, SH.maxCount, Literal(1)))
        
        # Relation property
        g.add((edge_shape, SH.property, ex.EdgeRelationProperty))
        relation_prop = ex.EdgeRelationProperty
        g.add((relation_prop, RDF.type, SH.PropertyShape))
        g.add((relation_prop, SH.path, ex.relation))
        g.add((relation_prop, SH.minCount, Literal(1)))
        g.add((relation_prop, SH.maxCount, Literal(1)))
        
        return g
    
    def validate_node(self, node_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a node against SHACL shapes.
        
        Args:
            node_data: Node data with 'id', 'type', and optional 'properties'
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        data_graph = self._node_to_graph(node_data)
        conforms, results_graph, results_text = validate(
            data_graph,
            shacl_graph=self._shapes,
            inference='rdfs',
            abort_on_first=False,
        )
        
        if conforms:
            return True, []
        
        errors = self._extract_validation_errors(results_text)
        return False, errors
    
    def validate_edge(self, edge_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate an edge against SHACL shapes.
        
        Args:
            edge_data: Edge data with 'id', 'source', 'target', 'relation',
                      and optional 'properties'
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        data_graph = self._edge_to_graph(edge_data)
        conforms, results_graph, results_text = validate(
            data_graph,
            shacl_graph=self._shapes,
            inference='rdfs',
            abort_on_first=False,
        )
        
        if conforms:
            return True, []
        
        errors = self._extract_validation_errors(results_text)
        return False, errors
    
    def _node_to_graph(self, node_data: Dict[str, Any]) -> Graph:
        """Convert node data to RDF graph.
        
        Args:
            node_data: Node data dictionary
            
        Returns:
            RDF graph representation
        """
        g = Graph()
        ex = Namespace("http://example.org/hcg/")
        
        node_id = node_data.get("id", "unknown")
        node_uri = ex[node_id]
        
        g.add((node_uri, RDF.type, ex.Node))
        
        if "type" in node_data:
            g.add((node_uri, ex.nodeType, Literal(node_data["type"])))
        
        if "properties" in node_data:
            for key, value in node_data["properties"].items():
                g.add((node_uri, ex[key], Literal(str(value))))
        
        return g
    
    def _edge_to_graph(self, edge_data: Dict[str, Any]) -> Graph:
        """Convert edge data to RDF graph.
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            RDF graph representation
        """
        g = Graph()
        ex = Namespace("http://example.org/hcg/")
        
        edge_id = edge_data.get("id", "unknown")
        edge_uri = ex[edge_id]
        
        g.add((edge_uri, RDF.type, ex.Edge))
        
        if "source" in edge_data:
            g.add((edge_uri, ex.source, ex[edge_data["source"]]))
        
        if "target" in edge_data:
            g.add((edge_uri, ex.target, ex[edge_data["target"]]))
        
        if "relation" in edge_data:
            g.add((edge_uri, ex.relation, Literal(edge_data["relation"])))
        
        if "properties" in edge_data:
            for key, value in edge_data["properties"].items():
                g.add((edge_uri, ex[key], Literal(str(value))))
        
        return g
    
    def _extract_validation_errors(self, results_text: str) -> List[str]:
        """Extract error messages from SHACL validation results.
        
        Args:
            results_text: SHACL validation results text
            
        Returns:
            List of error messages
        """
        errors = []
        if results_text:
            # Parse the results text to extract meaningful error messages
            lines = results_text.split('\n')
            for line in lines:
                if 'Validation Result' in line or 'Message:' in line:
                    errors.append(line.strip())
        
        if not errors:
            errors.append("Validation failed but no specific errors found")
        
        return errors
    
    def validate_mutation(
        self, 
        mutation_type: str,
        data: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate a graph mutation (node or edge operation).
        
        Args:
            mutation_type: Type of mutation ('add_node', 'add_edge', 'update_node', etc.)
            data: Data for the mutation
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if mutation_type in ['add_node', 'update_node']:
            return self.validate_node(data)
        elif mutation_type in ['add_edge', 'update_edge']:
            return self.validate_edge(data)
        else:
            return False, [f"Unknown mutation type: {mutation_type}"]
