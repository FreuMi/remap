from .vocabulary import *
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from datetime import datetime
import sys
from .helper import *
from .join_identification import *

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
    g.add((URIRef(tm_name), RDF_TYPE, TM_CLASS))

    return tm_name

# Add logical source triples
def add_logical_source(
    g: Graph, tm_name: str, path: str, is_json_data: bool, iterator: str = "$"
) -> None:
    # Generate blank nodes
    bn1 = BNode()
    bn2 = BNode()

    # Add to graph
    g.add((URIRef(tm_name), LOGICAL_SOURCE, bn1))
    g.add((bn1, RDF_TYPE, LOGICAL_SOURCE_CLASS))
    if not is_json_data:
        g.add((bn1, REF_FORMULATION, CSV_FORMAT)) # CSV
    else:
        g.add((bn1, REF_FORMULATION, JSON_FORMAT)) # JSON
        g.add((bn1, ITERATOR, Literal(iterator)))

    g.add((bn1, SOURCE, bn2))
    g.add((bn2, RDF_TYPE, PATH_SOURCE_CLASS))
    g.add((bn2, ROOT_DIR, MAPPING_DIR))
    g.add((bn2, MAPPING_PATH, Literal(path)))

# Add subject information
def add_subject(g: Graph, tm_name: str, term_map: str, term_map_type: str, term_type: str, g_term_type: str, g_term_map: str, g_term_map_type: str) -> None:
    bn1 = BNode()

    g.add((URIRef(tm_name), SUBJECT_MAP, bn1))

     # If constant & blanknode
    if term_map_type == "constant" and term_type == "blanknode":
        g.add((bn1, TERM_TYPE, BLANKNODE))
        g.add((bn1, CONSTANT, URIRef(term_map)))
    else:
        # Add term map and term map type
        if term_map_type == "template":
            g.add((bn1, TEMPLATE, Literal(term_map)))
        elif term_map_type == "reference":
            g.add((bn1, REFERENCE, Literal(term_map)))
        elif term_map_type == "constant":
            g.add((bn1, CONSTANT, URIRef(term_map)))
        else:
            print("Error: Subject term_map_type unsupported! Found", term_map_type)
            sys.exit(1)
        
        # Add term_type
        if term_map_type != "constant":
            if term_type == "iri":
                g.add((bn1, TERM_TYPE, IRI))
            elif term_type == "blanknode":
                g.add((bn1, TERM_TYPE, BLANKNODE))
            elif term_type == "literal":
                g.add((bn1, TERM_TYPE, LITERAL))
            else:
                print("Error: Subject term_type unsupported!")
                sys.exit(1)

    ## Add graph if needed
    if g_term_map_type == "":
        return
    
    # Add graph map  
    bn2 = BNode()
    g.add((bn1, GRAPH_MAP, bn2))   
    if g_term_map_type == "template":
        g.add((bn2, TEMPLATE, Literal(g_term_map)))
    elif g_term_map_type == "reference":
        g.add((bn2, REFERENCE, Literal(g_term_map)))
    elif g_term_map_type == "constant":
        g.add((bn2, CONSTANT, URIRef(g_term_map)))
    else:
        print("Error: Graph term_map_type unsupported! Found", g_term_map_type)
        sys.exit(1)

        
# Add a POM
def add_predicate_object_map(g: Graph, tm_name: str,\
                            is_json_data: bool,\
                            p_term_map: str, p_term_map_type: str, p_term_type: str,\
                            o_term_map: str, o_term_map_type: str, o_term_type: str, \
                            data_type_term_type: str, data_type_term_map: str, data_type_term_map_type: str,\
                            lang_tag_term_type: str, lang_tag_term_map: str, lang_tag_term_map_type: str) -> None:
    bn1 = BNode()
    bn2 = BNode()
    bn3 = BNode()
    
    g.add((URIRef(tm_name), POM, bn1))

    # Predicate
    g.add((bn1, PREDICATE_MAP, bn2))
    if p_term_map_type == "constant":
        g.add((bn2, CONSTANT, URIRef(p_term_map)))
    elif p_term_map_type == "reference":
        g.add((bn2, REFERENCE, URIRef(p_term_map)))
    elif p_term_map_type == "template":
        g.add((bn2, TEMPLATE, URIRef(p_term_map)))
    else:
        print("Error: Prediacte term_map_type unsupported! Found", p_term_map_type)
        sys.exit(1)

    # Object
    g.add((bn1, OBJECT_MAP, bn3))
    if o_term_map_type == "reference":
        g.add((bn3, REFERENCE, Literal(o_term_map)))
    elif o_term_map_type == "template":
        g.add((bn3, TEMPLATE, Literal(o_term_map)))
    elif o_term_map_type == "constant":
        if o_term_type == "literal":
            g.add((bn3, CONSTANT, Literal(o_term_map)))
        else: 
            g.add((bn3, CONSTANT, URIRef(o_term_map)))
    else:
        print("Error: Object term_map_type unsupported! Found", o_term_map_type)
        sys.exit(1)
    
    if o_term_map_type != "constant" and not (
        is_json_data and o_term_map_type == "reference" and o_term_type == "literal"
    ):
        if o_term_type == "iri":
            g.add((bn3, TERM_TYPE, IRI))
        elif o_term_type == "blanknode":
            g.add((bn3, TERM_TYPE, BLANKNODE))
        elif o_term_type == "literal":
            g.add((bn3, TERM_TYPE, LITERAL))
        else:
            print("Error: Predicate term_type unsupported!")
            sys.exit(1)

    # Add datatype map 
    if data_type_term_type != "" and not (
        is_json_data and o_term_map_type == "reference" and o_term_type == "literal"
    ):
        bn4 = BNode()
        g.add((bn3, DATATYPE_MAP, bn4))
        if data_type_term_map_type == "reference":
            g.add((bn4, REFERENCE, Literal(data_type_term_map)))
        elif data_type_term_map_type == "template":
            g.add((bn4, TEMPLATE, Literal(data_type_term_map)))
        elif data_type_term_map_type == "constant":
            g.add((bn4, CONSTANT, URIRef(data_type_term_map)))
    elif lang_tag_term_type != "":
        if lang_tag_term_map_type == "constant":
            g.add((bn3, LANG_TAG_SHORT, Literal(lang_tag_term_map)))
        else:
            print("Error handling language tag term map type!. Got: ", lang_tag_term_map_type)


# Add a POM
def add_predicate_object_map_join(g: Graph, tm_name: str,\
                            p_term_map: str, p_term_map_type: str, tm2: str, parent: str, child: str) -> None:
    bn1 = BNode()
    bn2 = BNode()
    bn3 = BNode()
    
    g.add((URIRef(tm_name), POM, bn1))

    # Predicate
    g.add((bn1, PREDICATE_MAP, bn2))
    if p_term_map_type == "constant":
        g.add((bn2, CONSTANT, URIRef(p_term_map)))
    elif p_term_map_type == "reference":
        g.add((bn2, REFERENCE, URIRef(p_term_map)))
    elif p_term_map_type == "template":
        g.add((bn2, TEMPLATE, URIRef(p_term_map)))
    else:
        print("Error: Prediacte term_map_type unsupported! Found", p_term_map_type)
        sys.exit(1)

    # Object
    g.add((bn1, OBJECT_MAP, bn3))
    g.add((bn3, RDF_TYPE, REF_OBJ_MAP))

    # Add join condition
    bn5 = BNode()
    g.add((bn3, JOIN_COND, bn5))
    g.add((bn5, PARENT, Literal(parent)))
    g.add((bn5, CHILD, Literal(child)))

    # Add parent tm
    g.add((bn3, PARENT_TM, URIRef(tm2)))

def build_sub_graph(file_path_csv: str, is_json_data: bool, json_iterator: str, s_term_map: str, s_term_map_type: str, s_term_type: str,\
                    p_term_map: str, p_term_map_type: str, p_term_type: str,\
                    o_term_map: str, o_term_map_type: str, o_term_type: str,\
                    g_term_type: str, g_term_map: str, g_term_map_type: str,\
                    data_type_term_type: str, data_type_term_map: str, data_type_term_map_type: str,\
                    lang_tag_term_type: str, lang_tag_term_map: str, lang_tag_term_map_type: str) -> Graph:
    rml_sub_graph = Graph()
    # Set RML namespace
    RML = Namespace("http://w3id.org/rml/")
    rml_sub_graph.bind("rml", RML)
    
    tm = init_template(rml_sub_graph)
    add_logical_source(rml_sub_graph, tm, file_path_csv, is_json_data, json_iterator)
    add_subject(rml_sub_graph, tm, s_term_map, s_term_map_type, s_term_type, g_term_type, g_term_map, g_term_map_type)
    add_predicate_object_map(rml_sub_graph, tm, is_json_data, p_term_map, p_term_map_type, p_term_type,\
                            o_term_map, o_term_map_type, o_term_type,\
                            data_type_term_type, data_type_term_map, data_type_term_map_type,\
                            lang_tag_term_type, lang_tag_term_map, lang_tag_term_map_type)

    return rml_sub_graph


def get_term_type_of_graph(g: Graph, node_type: str):
    # p is always iri
    if node_type == "p":
        return "iri"

    node = ""

    if node_type == "s":
        # Get subject map node
        for s,p,o in g:
            if str(p) == "http://w3id.org/rml/subjectMap":
                node = str(o)
                break
    elif node_type == "p":
        # Get subject map node
        for s,p,o in g:
            if str(p) == "http://w3id.org/rml/predicateMap":
                node = str(o)
                break
    elif node_type == "o":
        # Get subject map node
        for s,p,o in g:
            if str(p) == "http://w3id.org/rml/objectMap":
                node = str(o)
                break
    else:
        print("Error: got, ", node_type)
        sys.exit(1)

    # Get termtype
    term_type = ""
    for s,p,o in g:
        if str(p) == "http://w3id.org/rml/termType" and str(s) == node:
            term_type = str(o)
            break
        
    if term_type == "http://w3id.org/rml/IRI":
        return "iri"
    elif term_type == "http://w3id.org/rml/BlankNode":
        return "blanknode"
    elif term_type == "http://w3id.org/rml/Literal":
        return "literal"
    else:
        print("Error in get_term_type_graph", term_type)
        sys.exit(1)


def build_sub_graph_join(g: Graph, g2: Graph) -> Graph:
    # Generate second part first
    rml_sub_graph2 = Graph()

    # Set RML namespace
    RML = Namespace("http://w3id.org/rml/")
    rml_sub_graph2.bind("rml", RML)

    file_path_csv2 = getPath(g2)
    o_term_map2, o_term_map_type2 = getObject(g2)
    o_term_type2 = get_term_type_of_graph(g2, "o")

    # Init, add subject, and source
    tm2 = init_template(rml_sub_graph2)
    add_logical_source(rml_sub_graph2, tm2, file_path_csv2)
    add_subject(rml_sub_graph2, tm2, o_term_map2, o_term_map_type2, o_term_type2, "", "", "")

    # Generate first part
    rml_sub_graph = Graph()

    file_path_csv = getPath(g)

    s_term_map, s_term_map_type = getSubject(g)
    s_term_type = get_term_type_of_graph(g, "s")

    g_term_map, g_term_map_type = getGraph(g)
    g_term_type = "iri"

    p_term_map, p_term_map_type = getPredicate(g)

    o_term_map, o_term_map_type = getObject(g)

    # Get child and parents 
    child, parent = identify_join(file_path_csv, file_path_csv2)

    tm = init_template(rml_sub_graph)
    add_logical_source(rml_sub_graph, tm, file_path_csv)
    add_subject(rml_sub_graph, tm, s_term_map, s_term_map_type, s_term_type, g_term_type, g_term_map, g_term_map_type)
    add_predicate_object_map_join(rml_sub_graph, tm, p_term_map, p_term_map_type, tm2, parent, child )

    # Combine both graphs
    # res = rml_sub_graph + rml_sub_graph2

    return (rml_sub_graph, rml_sub_graph2)
