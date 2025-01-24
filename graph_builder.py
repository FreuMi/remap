import vocabulary as voc
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from datetime import datetime
import sys

# Gernerate a random id based on current time 
def get_random_id() -> str:
    current_time = datetime.now()
    time_str = str(current_time)
    time_str = time_str.replace(" ", "").replace(":","").replace("-","").replace(".","")
    return time_str

# Add inital graph data
def init_template(g: Graph) -> str:
    # Generate new TM name
    id = get_random_id()
    tm_name = f"http://example.org/{id}"
    g.add((URIRef(tm_name), voc.RDF_TYPE, voc.TM_CLASS))

    return tm_name

# Add logical source triples
def add_logical_source(g: Graph, tm_name: str, path: str) -> None:
    # Generate blank nodes
    bn1 = BNode()
    bn2 = BNode()

    # Add to graph
    g.add((URIRef(tm_name), voc.LOGICAL_SOURCE, bn1))
    g.add((bn1, voc.LOGICAL_SOURCE, voc.LOGICAL_SOURCE_CLASS))
    g.add((bn1, voc.REF_FORMULATION, voc.CSV_FORMAT)) # Only support for csv
    g.add((bn1, voc.SOURCE, bn2))
    g.add((bn2, voc.RDF_TYPE, voc.PATH_SOURCE_CLASS))
    g.add((bn2, voc.ROOT_DIR, voc.MAPPING_DIR))
    g.add((bn2, voc.MAPPING_PATH, Literal(path)))

# Add subject information
def add_subject(g: Graph, tm_name: str, term_map: str, term_map_type: str, term_type: str) -> None:
    bn1 = BNode()

    g.add((URIRef(tm_name), voc.SUBJECT_MAP, bn1))

    # Add term map and term map type
    if term_map_type == "template":
        g.add((bn1, voc.TEMPLATE, Literal(term_map)))
    elif term_map_type == "reference":
        g.add((bn1, voc.REFERENCE, Literal(term_map)))
    elif term_map_type == "constant":
        g.add((bn1, voc.CONSTANT, Literal(term_map)))
    else:
        print("Error: Subject term_map_type unsupported! Found", term_map_type)
        sys.exit(1)

    # Add term_type
    if term_type == "iri":
        g.add((bn1, voc.TERM_TYPE, voc.IRI))
    elif term_type == "blanknode":
        g.add((bn1, voc.TERM_TYPE, voc.BLANKNODE))
    elif term_type == "literal":
        g.add((bn1, voc.TERM_TYPE, voc.LITERAL))
    else:
        print("Error: Subject term_type unsupported!")
        sys.exit(1)
        
# Add a POM
def add_predicate_object_map(g: Graph, tm_name: str, p_term_map: str, p_term_map_type: str, p_term_type: str, o_term_map: str, o_term_map_type: str, o_term_type: str) -> None:
    bn1 = BNode()
    bn2 = BNode()
    bn3 = BNode()
    
    g.add((URIRef(tm_name), voc.POM, bn1))

    # Predicate
    g.add((bn1, voc.PREDICATE_MAP, bn2))
    if p_term_map_type == "constant":
        g.add((bn2, voc.CONSTANT, Literal(p_term_map)))
    else:
        print("Error: Prediacte term_map_type unsupported! Found", p_term_map_type)
        sys.exit(1)

    if p_term_type == "iri":
        g.add((bn2, voc.TERM_TYPE, voc.IRI))
    elif p_term_type == "blanknode":
        g.add((bn2, voc.TERM_TYPE, voc.BLANKNODE))
    elif p_term_type == "literal":
        g.add((bn2, voc.TERM_TYPE, voc.LITERAL))
    else:
        print("Error: Predicate term_type unsupported! Found", p_term_type)
        sys.exit(1)

    # Object
    g.add((bn1, voc.OBJECT_MAP, bn3))
    if o_term_map_type == "reference":
        g.add((bn3, voc.REFERENCE, Literal(o_term_map)))
    elif o_term_map_type == "template":
        g.add((bn1, voc.TEMPLATE, Literal(o_term_map)))
    elif o_term_map_type == "constant":
        g.add((bn1, voc.CONSTANT, Literal(o_term_map)))
    else:
        print("Error: Object term_map_type unsupported! Found", o_term_map_type)
        sys.exit(1)
 
    if o_term_type == "iri":
        g.add((bn3, voc.TERM_TYPE, voc.IRI))
    elif o_term_type == "blanknode":
        g.add((bn3, voc.TERM_TYPE, voc.BLANKNODE))
    elif o_term_type == "literal":
        g.add((bn3, voc.TERM_TYPE, voc.LITERAL))
    else:
        print("Error: Predicate term_type unsupported!")
        sys.exit(1)


def build_sub_graph(file_path_csv: str, s_term_map: str, s_term_map_type: str, s_term_type: str, p_term_map: str, p_term_map_type: str, p_term_type: str, o_term_map: str, o_term_map_type: str, o_term_type: str) -> Graph:
    rml_sub_graph = Graph()

    # Set RML namespace
    RML = Namespace("http://w3id.org/rml/")
    rml_sub_graph.bind("rml", RML)
    
    tm = init_template(rml_sub_graph)
    add_logical_source(rml_sub_graph, tm, file_path_csv)
    add_subject(rml_sub_graph, tm, s_term_map, s_term_map_type, s_term_type)
    add_predicate_object_map(rml_sub_graph, tm, p_term_map, p_term_map_type, p_term_type, o_term_map, o_term_map_type, o_term_type)

    return rml_sub_graph