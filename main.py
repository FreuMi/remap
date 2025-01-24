import pandas as pd
from rdflib import Graph, URIRef, Literal, BNode
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
    if rdf_term not in csv_data:
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



# funciton to parse rdf data in nquads
def parse(path: str) -> list[Quad]:
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

#def isDuplicateGraph(graph, all_graphs):

def main():
    # Config
    file_path_csv = 'student.csv'
    file_path_rdf = 'output.nq'

    # Load csv data
    data: pd.DataFrame = pd.read_csv(file_path_csv, dtype=str)

    # Load RDF data
    rdf_data = parse(file_path_rdf)

    rml_sub_graphs = []

    # Iterate over all csv data
    for row in data.itertuples(index=False):
        row: dict[str, str] = row._asdict() 
        # Iterate over all elements in the row
        for key, value in row.items():
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

                s_term_map, s_term_map_type = get_term_map_type(s, key, value)

                
                ## Handle predicate ##
                p_term_type = get_term_type(p)
                p_term_map = ""
                p_term_map_type = ""

                # clean p value
                p = clean_entry(p)

                p_term_map, p_term_map_type = get_term_map_type(p, key, value)

                ## Handle object ##
                o_term_type = get_term_type(o)
                o_term_map = ""
                o_term_map_type = ""

                # clean o value
                o = clean_entry(o)
                o_term_map, o_term_map_type = get_term_map_type(o, key, value)

                
                ## Build rml graph ##
                rml_sub_graph = graph_builder.build_sub_graph(file_path_csv, s_term_map, s_term_map_type, s_term_type, p_term_map, p_term_map_type, p_term_type, o_term_map, o_term_map_type, o_term_type)

                rml_sub_graphs.append(rml_sub_graph)

    # Print output
    for rml_sub_graph in rml_sub_graphs:
        print(rml_sub_graph.serialize(format="turtle"))

main()