import sys
from rdflib import Graph, Namespace, URIRef

def getPath(graph: Graph) -> str:
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/path":
            return str(o)   

def getSubject(graph: Graph) -> tuple[str,str]:
    subjectMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/subjectMap":
            subjectMap = str(o)
            break
    cnt = 0
    for s,p,o in graph:
        if str(s) == subjectMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
            else:
                cnt += 1
        
    if cnt == 1:
        pass
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

    for _, items in grouped.items():
        base_g, base_tm = items[0]

        # create a fresh result graph from base graph
        merged = Graph()
        for triple in base_g:
            merged.add(triple)

        for g, tm in items[1:]:
            for pom in g.objects(tm, RML.predicateObjectMap):
                new_pom = clone_subtree(g, pom, merged)
                merged.add((base_tm, RML.predicateObjectMap, new_pom))

        merged_graphs.append(merged)

    return merged_graphs