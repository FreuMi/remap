import pandas as pd
from rdflib import Graph, URIRef, Literal, BNode
import sys
import uuid
import ctypes
from dataclasses import dataclass

@dataclass
class Quad:
    s: str
    p: str
    o: str
    g: str

RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

TM_CLASS = URIRef("http://w3id.org/rml/TriplesMap")
LOGICAL_SOURCE = URIRef("http://w3id.org/rml/logicalSource")
LOGICAL_SOURCE_CLASS = URIRef("http://w3id.org/rml/LogicalSource")
REF_FORMULATION = URIRef("http://w3id.org/rml/referenceFormulation")
CSV_FORMAT = URIRef("http://w3id.org/rml/CSV")
SOURCE = URIRef("http://w3id.org/rml/source")
PATH_SOURCE_CLASS = URIRef("http://w3id.org/rml/RelativePathSource")
ROOT_DIR = URIRef("http://w3id.org/rml/root")
MAPPING_DIR = URIRef("http://w3id.org/rml/MappingDirectory")
MAPPING_PATH = URIRef("http://w3id.org/rml/path")

SUBJECT_MAP = URIRef("http://w3id.org/rml/subjectMap")
TEMPLATE = URIRef("http://w3id.org/rml/template")
CONSTANT = URIRef("http://w3id.org/rml/constant")
REFERENCE = URIRef("http://w3id.org/rml/reference")
TERM_TYPE = URIRef("http://w3id.org/rml/termType")
POM = URIRef("http://w3id.org/rml/predicateObjectMap")
PREDICATE_MAP = URIRef("http://w3id.org/rml/predicateMap")
OBJECT_MAP = URIRef("http://w3id.org/rml/objectMap")

IRI = URIRef("http://w3id.org/rml/IRI")
BLANKNODE = URIRef("http://w3id.org/rml/BlankNode")
LITERAL = URIRef("http://w3id.org/rml/Literal")

def get_id():
    random_uuid = str(uuid.uuid4())
    return random_uuid[:5]

def get_term_map_type(data, value):
    data = data.replace(value, "")
    if data == "":
        return "reference"
    elif data != "":
        return "template"

def parse_rdf(path):
    # Load the shared library
    lib = ctypes.CDLL("./libnormalizer.so")

    # Declare the argument and return types for our exported function
    # (NormalizeRmlMappingPy expects a char* and returns a char*)
    lib.NormalizeRmlMappingPy.argtypes = [ctypes.c_char_p]
    lib.NormalizeRmlMappingPy.restype = ctypes.c_char_p

    # Call the function by passing the path to mapping file as a byte string
    file_path = path.encode()
    result_ptr = lib.NormalizeRmlMappingPy(file_path)

    # Convert the result (C string) to a Python string
    result_str = ctypes.string_at(result_ptr).decode("utf-8")
    print("Normalized Mapping:")
    print(result_str)
    sys.exit(1)

def init_template(g):
    # Generate new TM name
    id = get_id()
    tm_name = f"http://example.org/{id}"
    g.add((URIRef(tm_name), RDF_TYPE, TM_CLASS))

    return tm_name

def add_logical_source(g, tm_name, path):
    # Generate blank nodes
    bn1 = BNode()
    bn2 = BNode()

    # Add to graph
    g.add((URIRef(tm_name), LOGICAL_SOURCE, bn1))
    g.add((bn1, LOGICAL_SOURCE, LOGICAL_SOURCE_CLASS))
    g.add((bn1, REF_FORMULATION, CSV_FORMAT)) # Only support for csv
    g.add((bn1, SOURCE, bn2))
    g.add((bn2, RDF_TYPE, PATH_SOURCE_CLASS))
    g.add((bn2, ROOT_DIR, MAPPING_DIR))
    g.add((bn2, MAPPING_PATH, Literal(path)))

def add_subject(g, tm_name, term_map, term_map_type, term_type):
    bn1 = BNode()

    g.add((URIRef(tm_name), SUBJECT_MAP, bn1))

    # Add term map and term map type
    if term_map_type == "template":
        g.add((bn1, TEMPLATE, Literal(term_map)))
    elif term_map_type == "reference":
        g.add((bn1, REFERENCE, Literal(term_map)))
    elif term_map_type == "constant":
        g.add((bn1, CONSTANT, Literal(term_map)))
    else:
        print("Error: Subject term_map_type unsupported! Found", term_map_type)
        sys.exit(1)

    # Add term_type
    if term_type == "iri":
        g.add((bn1, TERM_TYPE, IRI))
    elif term_type == "blanknode":
        g.add((bn1, TERM_TYPE, BLANKNODE))
    elif term_type == "literal":
        g.add((bn1, TERM_TYPE, LITERAL))
    else:
        print("Error: Subject term_type unsupported!")
        sys.exit(1)
        

def add_predicate_object_map(g, tm_name, p_term_map, p_term_map_type, p_term_type, o_term_map, o_term_map_type, o_term_type):
    bn1 = BNode()
    bn2 = BNode()
    bn3 = BNode()
    
    g.add((URIRef(tm_name), POM, bn1))

    # Predicate
    g.add((bn1, PREDICATE_MAP, bn2))
    if p_term_map_type == "constant":
        g.add((bn2, CONSTANT, Literal(p_term_map)))
    else:
        print("Error: Prediacte term_map_type unsupported! Found", p_term_map_type)
        sys.exit(1)

    if p_term_type == "iri":
        g.add((bn2, TERM_TYPE, IRI))
    elif p_term_type == "blanknode":
        g.add((bn2, TERM_TYPE, BLANKNODE))
    elif p_term_type == "literal":
        g.add((bn2, TERM_TYPE, LITERAL))
    else:
        print("Error: Predicate term_type unsupported! Found", p_term_type)
        sys.exit(1)

    # Object
    g.add((bn1, OBJECT_MAP, bn3))
    if o_term_map_type == "reference":
        g.add((bn3, REFERENCE, Literal(o_term_map)))
    elif o_term_map_type == "template":
        g.add((bn1, TEMPLATE, Literal(o_term_map)))
    elif o_term_map_type == "constant":
        g.add((bn1, CONSTANT, Literal(o_term_map)))
    else:
        print("Error: Object term_map_type unsupported! Found", o_term_map_type)
        sys.exit(1)
 
    if o_term_type == "iri":
        g.add((bn3, TERM_TYPE, IRI))
    elif o_term_type == "blanknode":
        g.add((bn3, TERM_TYPE, BLANKNODE))
    elif o_term_type == "literal":
        g.add((bn3, TERM_TYPE, LITERAL))
    else:
        print("Error: Predicate term_type unsupported!")
        sys.exit(1)

    
def parser(path):
    rdf_data = []
    # Load file
    with open(path, 'r') as f:
        for line in f:
            line_parts = line.split(" ")
            # Hanlde without graph
            if len(line_parts) == 4:
                x = Quad(line_parts[0], line_parts[1], line_parts[2], "")
                rdf_data.append(x)

    return rdf_data

def clean_entry(entry):
    if isURI(entry) or isLiteral(entry):
        return entry[1:-1]
    if isBlanknode(entry):
        return entry[2:]

def isURI(value):
    if value[0] == "<" and value[-1] == ">":
        return True
    else: 
        return False
    
def isBlanknode(value):
    if value[0] == "_" and value[1] == ":":
        return True
    else:
        return False

def isLiteral(value):
    if value[0] == "\"" and value[-1] == "\"":
        return True
    else: 
        return False

#def isDuplicateGraph(graph, all_graphs):


# Config
file_path_csv = 'student.csv'
file_path_rdf = 'output.nq'

# Load csv data
data = pd.read_csv(file_path_csv, dtype=str)

# Load RDF data
rdf_data = parser(file_path_rdf)

rml_sub_graphs = []

# Iterate over all csv data
for row in data.itertuples(index=False):
    row = row._asdict() 
    # Iterate over all elements in the row
    for key, value in row.items():
        # Iterate over the graph
        for element in rdf_data:
            # Access data
            s = element.s
            p = element.p
            o = element.o
            g = element.g

            # Handle subject
            s_term_type = ""
            s_term_map = ""
            s_term_map_type = ""

            if isURI(s):
                s_term_type = "iri"
            elif isLiteral(s):
                s_term_type = "literal"
            elif isBlanknode(s):
                s_term_type = "blanknode"
            else:
                print("Error detecting s_term_type.")
                sys.exit(1)
            
            # clean s value
            s = clean_entry(s)

            if value not in s:
                s_term_map_type = "constant"
                s_term_map = s
            else:
                s_term_map_type = get_term_map_type(s, value)
                if s_term_map_type == "template":
                    s_term_map = s.replace(value, "{"+key+"}")
                elif s_term_map_type == "reference":
                    s_term_map = key

            
            # Handle predicate
            p_term_type = ""
            p_term_map = ""
            p_term_map_type = ""

            if isURI(p):
                p_term_type = "iri"
            elif isLiteral(p):
                p_term_type = "literal"
            elif isBlanknode(p):
                p_term_type = "blanknode"
            else:
                print("Error detecting p_term_type.")
                sys.exit(1)

            # clean p value
            p = clean_entry(p)

            if value not in p:
                p_term_map_type = "constant"
                p_term_map = p
            else:
                p_term_map_type = get_term_map_type(p, value)
                if p_term_map_type == "template":
                    p_term_map = p.replace(value, "{"+key+"}")
                elif p_term_map_type == "reference":
                    p_term_map = key 


            # Handle object
            o_term_type = ""
            o_term_map = ""
            o_term_map_type = ""

            if isURI(o):
                o_term_type = "iri"
            elif isLiteral(o):
                o_term_type = "literal"
            elif isBlanknode(o):
                o_term_type = "blanknode"
            else:
                print("Error detecting o_term_type.")
                sys.exit(1)

            # clean o value
            o = clean_entry(o)

            if value not in o:
                o_term_map_type = "constant"
                o_term_map = o
            else:
                o_term_map_type = get_term_map_type(o, value)
                if o_term_map_type == "template":
                    o_term_map = o.replace(value, "{"+key+"}")
                elif o_term_map_type == "reference":
                    o_term_map = key 

            
            # Build rml graph
            rml_sub_graph = Graph()

            tm = init_template(rml_sub_graph)
            add_logical_source(rml_sub_graph, tm, file_path_csv)
            add_subject(rml_sub_graph, tm, s_term_map, s_term_map_type, s_term_type)
            add_predicate_object_map(rml_sub_graph, tm, p_term_map, p_term_map_type, p_term_type, o_term_map, o_term_map_type, o_term_type)

            rml_sub_graphs.append(rml_sub_graph)

# Print output
for rml_sub_graph in rml_sub_graphs:
    print(rml_sub_graph.serialize(format="nt"))