import sys
from rdflib import Graph, Namespace, URIRef

def getPath(graph: Graph) -> str:
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/path":
            return str(o)   


def getLogicalSourceDetails(graph: Graph) -> tuple[str, bool, str]:
    logical_source = None
    path = ""
    is_json_data = False
    iterator = "$"

    for s, p, o in graph:
        if str(p) == "http://w3id.org/rml/logicalSource":
            logical_source = str(o)
            break

    if logical_source is None:
        print("Error in getLogicalSourceDetails")
        sys.exit(1)

    source_node = None
    for s, p, o in graph:
        if str(s) != logical_source:
            continue
        if str(p) == "http://w3id.org/rml/referenceFormulation":
            is_json_data = str(o) == "http://w3id.org/rml/JSONPath"
        elif str(p) == "http://w3id.org/rml/iterator":
            iterator = str(o)
        elif str(p) == "http://w3id.org/rml/source":
            source_node = str(o)

    if source_node is None:
        print("Error in getLogicalSourceDetails")
        sys.exit(1)

    for s, p, o in graph:
        if str(s) == source_node and str(p) == "http://w3id.org/rml/path":
            path = str(o)
            break

    return path, is_json_data, iterator

def getSubject(graph: Graph) -> tuple[str,str]:
    subjectMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/subjectMap":
            subjectMap = str(o)
            break
    has_blanknode_termtype = False
    for s,p,o in graph:
        if str(s) == subjectMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
            elif str(p) == "http://w3id.org/rml/termType" and str(o) == "http://w3id.org/rml/BlankNode":
                has_blanknode_termtype = True
        
    if has_blanknode_termtype:
        return "", "none"
    print("Error in getSubject")
    sys.exit(1)
            
def getPredicate(graph: Graph) -> tuple[str,str]:
    predicateMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/predicateMap":
            predicateMap = str(o)
            break
    
    for s,p,o in graph:
        if str(s) == predicateMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    
    print("Error in getPredicate")
    sys.exit(1)
            
def getObject(graph: Graph) -> tuple[str,str]:
    objectMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/objectMap":
            objectMap = str(o)
            break
    for s,p,o in graph:
        if str(s) == objectMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    print("Error in getObject")
    sys.exit(1)

def getDatatype(graph: Graph) -> tuple[str, str]:
    objectMap = ""
    datatypeMap = ""
    for s, p, o in graph:
        if str(p) == "http://w3id.org/rml/objectMap":
            objectMap = str(o)
            break
    for s, p, o in graph:
        if str(s) == objectMap and str(p) == "http://w3id.org/rml/datatypeMap":
            datatypeMap = str(o)
            break
    if datatypeMap == "":
        return "", ""
    for s, p, o in graph:
        if str(s) == datatypeMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    return "", ""

def getLanguage(graph: Graph) -> tuple[str, str]:
    objectMap = ""
    for s, p, o in graph:
        if str(p) == "http://w3id.org/rml/objectMap":
            objectMap = str(o)
            break
    for s, p, o in graph:
        if str(s) == objectMap and str(p) == "http://w3id.org/rml/language":
            return str(o), "constant"
    return "", ""

def getGraph(graph: Graph) -> tuple[str,str]:
    graphMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/graphMap":
            graphMap = str(o)
            break
    for s,p,o in graph:
        if str(s) == graphMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    return "", ""



#######################################################################
# Code for merging triple maps

from collections import defaultdict
from rdflib import Graph, BNode, URIRef, Literal, Namespace
from rdflib.namespace import RDF

RML = Namespace("http://w3id.org/rml/")

def node_signature(g: Graph, node):
    """
    Build a structural signature for a node and its outgoing subtree.
    Works for URIRefs, Literals, and BNodes.
    """
    if isinstance(node, URIRef):
        return ("URI", str(node))

    if isinstance(node, Literal):
        return (
            "LIT",
            str(node),
            str(node.datatype) if node.datatype else None,
            node.language,
        )

    if isinstance(node, BNode):
        outgoing = []
        for p, o in g.predicate_objects(node):
            outgoing.append((str(p), node_signature(g, o)))
        outgoing.sort()
        return ("BNODE", tuple(outgoing))

    return ("OTHER", str(node))


def clone_subtree(src_graph: Graph, root, dst_graph: Graph, memo=None):
    """
    Deep-copy a node subtree from src_graph into dst_graph.
    Recreates blank nodes, preserves URIRefs/Literals.
    """
    if memo is None:
        memo = {}

    if isinstance(root, URIRef) or isinstance(root, Literal):
        return root

    if isinstance(root, BNode):
        if root in memo:
            return memo[root]
        new_root = BNode()
        memo[root] = new_root

        for p, o in src_graph.predicate_objects(root):
            new_o = clone_subtree(src_graph, o, dst_graph, memo)
            dst_graph.add((new_root, p, new_o))

        return new_root

    return root


def clone_triples_map(src_graph: Graph, tm, dst_graph: Graph):
    dst_graph.add((tm, RDF.type, RML.TriplesMap))

    logical_source = src_graph.value(tm, RML.logicalSource)
    if logical_source is not None:
        new_logical_source = clone_subtree(src_graph, logical_source, dst_graph)
        dst_graph.add((tm, RML.logicalSource, new_logical_source))

    subject_map = src_graph.value(tm, RML.subjectMap)
    if subject_map is not None:
        new_subject_map = clone_subtree(src_graph, subject_map, dst_graph)
        dst_graph.add((tm, RML.subjectMap, new_subject_map))

    for pom in src_graph.objects(tm, RML.predicateObjectMap):
        new_pom = clone_subtree(src_graph, pom, dst_graph)
        dst_graph.add((tm, RML.predicateObjectMap, new_pom))


def merge_triples_maps(graphs):
    """
    graphs: list of rdflib.Graph, each containing one rml:TriplesMap subgraph
    returns: list of merged graphs
    """
    grouped = defaultdict(list)

    for g in graphs:
        tm = next(g.subjects(RDF.type, RML.TriplesMap), None)
        if tm is None:
            continue

        logical_source = g.value(tm, RML.logicalSource)
        subject_map = g.value(tm, RML.subjectMap)

        key = (
            node_signature(g, logical_source),
            node_signature(g, subject_map),
        )
        grouped[key].append((g, tm))

    merged_graphs = []
    replacements = {}

    for _, items in grouped.items():
        base_g, base_tm = items[0]

        # create a fresh result graph from base graph
        merged = Graph()
        for triple in base_g:
            merged.add(triple)

        for g, tm in items[1:]:
            replacements[tm] = base_tm
            for pom in g.objects(tm, RML.predicateObjectMap):
                new_pom = clone_subtree(g, pom, merged)
                merged.add((base_tm, RML.predicateObjectMap, new_pom))
            for other_tm in g.subjects(RDF.type, RML.TriplesMap):
                if other_tm == tm or (other_tm, RDF.type, RML.TriplesMap) in merged:
                    continue
                clone_triples_map(g, other_tm, merged)

        merged_graphs.append(merged)

    if replacements:
        parent_triples_map = URIRef("http://w3id.org/rml/parentTriplesMap")
        for merged in merged_graphs:
            updates = []
            for s, p, o in merged.triples((None, parent_triples_map, None)):
                replacement = replacements.get(o)
                if replacement is not None and replacement != o:
                    updates.append((s, o, replacement))
            for s, old_o, new_o in updates:
                merged.remove((s, parent_triples_map, old_o))
                merged.add((s, parent_triples_map, new_o))

    return merged_graphs
