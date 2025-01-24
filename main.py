import pandas as pd
from rdflib import Graph, Namespace
import sys
from dataclasses import dataclass
import graph_builder

# Class to store quad data
@dataclass
class Quad:
    s: str
    p: str
    o: str
    g: str

# Itentify term_map_type of (constant, reference, template) and generate term_map
def get_term_map_type(rdf_term: str, csv_header: str, csv_data: str) -> tuple[str, str]:
    rdf_term_map_type = ""
    if csv_data not in rdf_term:
        rdf_term_map_type = "constant"
    else:
        csv_data = csv_data.replace(rdf_term, "")
        if csv_data == "":
            rdf_term_map_type = "reference"
        elif csv_data != "":
            rdf_term_map_type = "template"
        else:
            print("Error detecting term_map_type! Found:", rdf_term_map_type)
            sys.exit(1)
    term_map = ""
    if rdf_term_map_type == "constant":
        term_map = rdf_term
    elif rdf_term_map_type == "template":
        term_map = rdf_term.replace(csv_data, "{"+csv_header+"}")
    elif rdf_term_map_type == "reference":
        term_map = csv_header
    else:
        print("Error generating term_map!")
        sys.exit(1)

    return term_map, rdf_term_map_type
    
def get_term_type(data: str) -> str:
    res = ""
    if isURI(data):
        res = "iri"
    elif isLiteral(data):
        res = "literal"
    elif isBlanknode(data):
        res = "blanknode"
    else:
        print("Error detecting s_term_type.")
        sys.exit(1)

    return res


def tokenizer(input_val: str) -> list[str]:
    result = []
    in_quotation = False
    word = ""
    
    for char in input_val:
        if char == "\"":
            # Toggle the in_quotation flag
            in_quotation = not in_quotation
            word += char 
        elif char == " ":
            if in_quotation:
                # Inside quotes, spaces are part of the word
                word += char
            else:
                # Outside quotes, space marks the end of a word
                if word:
                    result.append(word)
                    word = ""
        else:
            # Add other characters to the current word
            word += char
    
    # Append the last word if it exists
    if word:
        result.append(word)
    
    return result



# funciton to parse rdf data in nquads
def parse(path: str) -> list[Quad]:
    rdf_data = []
    # Load file
    with open(path, 'r') as f:
        for line in f:
            line_parts = tokenizer(line)
            # Hanlde without graph
            if len(line_parts) == 4:
                x = Quad(line_parts[0], line_parts[1], line_parts[2], "")
                rdf_data.append(x)

    return rdf_data

# Remove < >, " ", _: from string 
def clean_entry(entry: str) -> str:
    if isURI(entry) or isLiteral(entry):
        return entry[1:-1]
    if isBlanknode(entry):
        return entry[2:]

# Check if entry is an URI
def isURI(value: str) -> bool:
    if value[0] == "<" and value[-1] == ">":
        return True
    else: 
        return False

# Check if entry is a URI
def isBlanknode(value: str) -> bool:
    if value[0] == "_" and value[1] == ":":
        return True
    else:
        return False

# Check if entry is literal
def isLiteral(value: str) -> bool:
    if value[0] == "\"" and value[-1] == "\"":
        return True
    else: 
        return False

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
    for s,p,o in graph:
        if str(s) == subjectMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
            
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

def extract_information(graph: Graph) -> tuple[str,str,str,str,str,str,str]:
    # Get source path 
    new_path = getPath(graph)
    # Get subject
    sub, subject_type = getSubject(graph)
    # Get predicate
    pred, predicate_type = getPredicate(graph)
    # Get object
    obj, object_type = getObject(graph)

    return (new_path, sub, subject_type, pred, predicate_type, obj, object_type)

def isDuplicateGraph(graph: Graph, all_graphs: list[Graph]) -> bool:
    reference_info = extract_information(graph)

    for stored_graph in all_graphs:
        # Check if source, subject, predicate, and object are the same.
        stored_information = extract_information(stored_graph)
        if reference_info == stored_information:
            return True

    return False

def main():
    # Config
    file_path_csv = 'ious.csv'
    file_path_rdf = 'output.nq'
    print("Starting...")

    # Load csv data
    data: pd.DataFrame = pd.read_csv(file_path_csv, dtype=str)
    # Load RDF data
    rdf_data = parse(file_path_rdf)
    rml_sub_graphs = []

    # Iterate over all csv data
    for row in data.itertuples(index=False):
        row: dict[str, str] = row._asdict() 
        
        # Iterate over the graph
        for element in rdf_data:
            # Access data
            s = element.s
            p = element.p
            o = element.o
            g = element.g

            ## Handle subject ##
            s_term_type = get_term_type(s)
            s_term_map = ""
            s_term_map_type = ""
            
            # clean s value
            s = clean_entry(s)
            # Iterate over all elements in the row and detect type
            s_term_map = s
            for key, value in row.items():
                term_map, term_map_type = get_term_map_type(s_term_map, key, value)
                if s_term_map_type == "":
                    s_term_map = term_map
                    s_term_map_type = term_map_type
                elif term_map_type != "constant":
                    s_term_map = term_map
                    s_term_map_type = term_map_type
                print()
                
            ## Handle predicate ##
            p_term_type = get_term_type(p)
            p_term_map = ""
            p_term_map_type = ""

            # clean p value
            p = clean_entry(p)
            p_term_map = p
            for key, value in row.items():
                term_map, term_map_type = get_term_map_type(p_term_map, key, value)
                if p_term_map_type == "":
                    p_term_map = term_map
                    p_term_map_type = term_map_type
                elif term_map_type != "constant":
                    p_term_map = term_map
                    p_term_map_type = term_map_type

            ## Handle object ##
            o_term_type = get_term_type(o)
            o_term_map = ""
            o_term_map_type = ""

            # clean o value
            o = clean_entry(o)
            o_term_map = o
            for key, value in row.items():
                term_map, term_map_type = get_term_map_type(o_term_map, key, value)
                if o_term_map_type == "":
                    o_term_map = term_map
                    o_term_map_type = term_map_type
                elif term_map_type != "constant":
                    o_term_map = term_map
                    o_term_map_type = term_map_type
            
            ## Build rml graph ##
            rml_sub_graph = graph_builder.build_sub_graph(file_path_csv, s_term_map, s_term_map_type, s_term_type, p_term_map, p_term_map_type, p_term_type, o_term_map, o_term_map_type, o_term_type)
            
            if not isDuplicateGraph(rml_sub_graph, rml_sub_graphs):
                rml_sub_graphs.append(rml_sub_graph)
            else:
                print("DUPLICATE")

    # Print output
    result_graph = Graph()
    # Set RML namespace
    RML = Namespace("http://w3id.org/rml/")
    result_graph.bind("rml", RML)
    for rml_sub_graph in rml_sub_graphs:
        result_graph += rml_sub_graph
    print(result_graph.serialize(format="turtle"))

main()