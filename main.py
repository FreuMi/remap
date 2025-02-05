import pandas as pd
from rdflib import Graph, Namespace, URIRef
import sys
from dataclasses import dataclass
import graph_builder
import re
import argparse
from urllib.parse import urlparse, quote
import vocabulary as voc

# Class to store quad data
@dataclass
class Quad:
    s: str
    p: str
    o: str
    g: str

def encode_uris(graph):
    updated_graph = Graph()
    for s, p, o in graph:
        # Encode only if the subject or object is a URI
        s = URIRef(quote(str(s), safe=":/#")) if isinstance(s, URIRef) else s
        o = URIRef(quote(str(o), safe=":/#")) if isinstance(o, URIRef) else o
        updated_graph.add((s, p, o))
    return updated_graph

def check_protected_iris(iri: str) -> tuple[bool, str, str]:
    protected_iris = ["http://www.w3.org/2000/01/rdf-schema#", "http://w3id.org/rml/", "http://www.w3.org/2001/XMLSchema#" ]
    	
    for protected_iri in protected_iris:
        if protected_iri in iri:
            res = iri.replace(protected_iri,"")
            return True, res, protected_iri
    return False, "", ""

def is_valid_uri(uri):
    try:
        result = urlparse(uri)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def remove_graph(reference_g: Graph, graphs: list[Graph]) -> list[Graph]:
    new_graphs = []
    for g in graphs:
        # Are graphs identical?
        if g.isomorphic(reference_g):
            continue
        new_graphs.append(g)

    return new_graphs

def is_join_graph(g):
    for s,p,o in g:
        if o == voc.REF_OBJ_MAP:
            return True
    return False

# Itentify term_map_type of (constant, reference, template) and generate term_map
def get_term_map_type(rdf_term: str, csv_header: str, csv_data: str, base_uri: str) -> tuple[str, str]:
    rdf_term_map_type = ""

    # Check for protected iri
    is_protected, rest_iri, protected_prefix = check_protected_iris(rdf_term)

    if is_protected:
        rdf_term = rest_iri

    if csv_data not in rdf_term:
        rdf_term_map_type = "constant"
    else:
        org_csv_data = csv_data
        csv_data = csv_data.replace(rdf_term, "")

        if csv_data != "":
            rdf_term_map_type = "template"
        elif is_valid_uri(base_uri + org_csv_data):
            rdf_term_map_type = "template"
            csv_data = org_csv_data
        elif csv_data == "":
            rdf_term_map_type = "reference"
        else:
            print("Error detecting term_map_type! Found:", rdf_term_map_type)
            sys.exit(1)

    if is_protected:
        rdf_term = protected_prefix + rest_iri

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
    if "^^" in data:
        data = data.split("^^")[0]

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


def decode_safe_iri(safe_iri: str) -> str:
    # Reverse lookup table for decoding
    decode_map = {
        "%20": " ", "%21": "!", "%22": "\"", "%23": "#", "%24": "$",
        "%25": "%", "%26": "&", "%27": "'", "%28": "(", "%29": ")",
        "%2A": "*", "%2B": "+", "%2C": ",", "%2F": "/", "%3A": ":",
        "%3B": ";", "%3C": "<", "%3D": "=", "%3E": ">", "%3F": "?",
        "%40": "@", "%5B": "[", "%5C": "\\", "%5D": "]", "%7B": "{",
        "%7C": "|", "%7D": "}"
    }

    # Create a regex pattern to match all encoded sequences in the string
    pattern = re.compile(r"%[0-9A-Fa-f]{2}")

    # Function to replace matched encoded symbols with their decoded values
    def decode_match(match):
        encoded = match.group(0)
        return decode_map.get(encoded, encoded)  # Return the decoded character or keep as-is

    # Use re.sub to perform decoding
    decoded_string = pattern.sub(decode_match, safe_iri)

    return decoded_string


# funciton to parse rdf data in nquads
def parse(path: str) -> list[Quad]:
    rdf_data = []
    # Load file
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line_parts = tokenizer(line)      

            # Decode iri safety
            for i in range(len(line_parts)):
                line_parts[i] = decode_safe_iri(line_parts[i])


            # Hanlde without graph
            if len(line_parts) == 4:
                x = Quad(line_parts[0], line_parts[1], line_parts[2], "")
                rdf_data.append(x)
            elif len(line_parts) == 5:
                x = Quad(line_parts[0], line_parts[1], line_parts[2], line_parts[3])
                rdf_data.append(x)
            else:
                print("Error parsing data. Found:", line_parts)
    return rdf_data

# Remove < >, " ", _: from string 
def clean_entry(entry: str) -> tuple[str,str]:
    result = ""

    if isURI(entry):
        result = entry[1:-1]
    elif isLiteral(entry):
        # Check for datatype
        if "^^" in entry:
            split_data = entry.split("^^")
            if len(split_data) != 2:
                print("Error Cleaning! Found:", split_data)
                sys.exit(1)
            entry = split_data[0]
        result = entry[1:-1]
    elif isBlanknode(entry):
        result = entry[2:]
    
    return result.strip()

def remove_base_uri(entry: str, base_uri: str) -> str:
    if entry != None:
        if base_uri in entry:
            entry = entry.replace(base_uri, "")
    return entry

# Check if entry is an URI
def isURI(value: str) -> bool:
    if value == "":
        return False

    if value[0] == "<" and value[-1] == ">":
        return True
    else: 
        return False

# Check if entry is a URI
def isBlanknode(value: str) -> bool:
    if value == "":
        return False
    
    if value[0] == "_" and value[1] == ":":
        return True
    else:
        return False

# Check if entry is literal
def isLiteral(value: str) -> bool:
    if value == "":
        return False
    
    # Filter datatype
    if "^^" in value:
        split_value = value.split("^^")
        if len(split_value) != 2:
            print("Error splitting data! Found", split_value)
            sys.exit(1)
        value = split_value[0]
        
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

def extract_information(graph: Graph) -> tuple[str,str,str,str,str,str,str, str]:
    # Get source path 
    new_path = getPath(graph)
    # Get subject
    sub, subject_type = getSubject(graph)
    # Get predicate
    pred, predicate_type = getPredicate(graph)
    # Get object
    obj, object_type = getObject(graph)

    return (new_path, sub, subject_type, pred, predicate_type, obj, object_type, "")

def get_parent(g: Graph) -> str:
    for s,p,o in g:
        if p == voc.PARENT:
            return str(o)

def get_child(g: Graph) -> str:
    for s,p,o in g:
        if p == voc.CHILD:
            return str(o)

def extract_information_join(graph1: Graph, graph2: Graph) -> tuple[str,str,str,str,str,str,str,str]:
    # Get source path 
    new_path = getPath(graph1)
    new_path2 = getPath(graph2)
    # Get subject
    sub, subject_type = getSubject(graph1)
    # Get predicate
    pred, predicate_type = getPredicate(graph1)
    # Get object
    obj, object_type = getSubject(graph2)

    child = get_child(graph1)
    parent = get_parent(graph1)

    return (new_path, sub, subject_type, pred, predicate_type, obj, object_type, new_path2, child, parent)

def isDuplicateGraph(graph: Graph, all_graphs: list[Graph]) -> bool:
    reference_info = extract_information(graph)

    for stored_graph in all_graphs:
        # Check if source, subject, predicate, and object are the same.
        stored_information = extract_information(stored_graph)
        if reference_info == stored_information:
            return True

    return False


def mask_string(input_data: str, row: dict[str, str]) -> str:
    # Replace entries with |||key||| instead of {key} for easier masking
    for key, _ in row.items():
        input_data = input_data.replace(f"{{{key}}}", f"|||{key}|||")
    
    # Mask all { and }
    tmp_input_data = ""
    for char in input_data:
        if char == "{":
            tmp_input_data += "\\{"
            continue
        elif char == "}":
            tmp_input_data += "\\}"
            continue
        tmp_input_data += char

    # replace |||key||| with {key}
    for key, _ in row.items():
        tmp_input_data = tmp_input_data.replace(f"|||{key}|||", f"{{{key}}}")

    return tmp_input_data

# Generate expected triple
def generate_expected_triple(data: pd.DataFrame, info, data2: pd.DataFrame = pd.DataFrame()) -> set[str]:
    # Generate data
    generated_triples = set()

    ### With Join ###
    if info[7] != "":
        ## Perform Join ##
        # Rename columns
        data = data.rename(columns=lambda col: f"{info[0].replace('.','')}_{col}")
        data2 = data2.rename(columns=lambda col: f"{info[7].replace('.','')}_{col}")
        
        child = f"{info[0].replace('.','')}_{info[8]}"
        parent = f"{info[7].replace('.','')}_{info[9]}"

        # Perform join
        join_result_df = pd.merge(
            data,
            data2,
            left_on=child,
            right_on=parent,
            suffixes=("_RxPxx", "_SxPxx"),
        )

        if info[0] == info[7]:
            # Handle self join
            columns_to_drop = [col for col in join_result_df.columns if col.endswith("_SxPxx")]
            simplified_df = join_result_df.drop(columns=columns_to_drop)
            simplified_df.columns = [col.replace("_RxPxx", "") for col in simplified_df.columns]
            join_result_df = simplified_df

        # Iterate over data
        for row in join_result_df.itertuples(index=False):
            row: dict[str, str] = row._asdict()
            
            s = ""
            p = ""
            o = ""

            # Subject
            if info[2] == "constant":
                s = info[1]
            elif info[2] == "reference":
                key = info[1]
                key = key.replace(" ", "a___")
                key = key.replace("{", "ab____")
                key = key.replace("}", "abb_____")
                key = key.replace("\\", "abbb______")
                s = row[key]
            elif info[2] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[1])
                s = info[1]
                for match in matches:
                    match_org = match
                    match = match.replace(" ", "a___")
                    match = match.replace("{", "ab____")
                    match = match.replace("}", "abb_____")
                    match = match.replace("\\", "abbb______")
                    match = f"{info[0].replace('.','')}_{match}"
                    s = s.replace("{"+match_org+"}", row[match])

            s = s.replace("a___", " ")
            s = s.replace("ab____", "\\{")
            s = s.replace("abb_____", "\\}")
            s = s.replace("abbb______", "\\")

            if info[4] == "constant":
                p = info[3]
            elif info[4] == "reference":
                key = info[3]
                key = key.replace(" ", "a___")
                key = key.replace("{", "ab____")
                key = key.replace("}", "abb_____")
                key = key.replace("\\", "abbb______")
                p = row[key]
            elif info[4] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[3])
                p = info[3]
                for match in matches:
                    match_org = match
                    match = match.replace(" ", "a___")
                    match = match.replace("{", "ab____")
                    match = match.replace("}", "abb_____")
                    match = match.replace("\\", "abbb______")
                    p = p.replace("{"+match_org+"}", row[match])
            
            p = p.replace("a___", " ")
            p = p.replace("ab____", "\\{")
            p = p.replace("abb_____", "\\}")
            p = p.replace("abbb______", "\\")

            if info[6] == "constant":
                o = info[5]
            elif info[6] == "reference":
                key = info[5]
                key = key.replace(" ", "a___")
                key = key.replace("{", "ab____")
                key = key.replace("}", "abb_____")
                key = key.replace("\\", "abbb______")
                o = row[key]
            elif info[6] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[5])
                o = info[5]
                for match in matches:
                    match_org = match
                    match = match.replace(" ", "a___")
                    match = match.replace("{", "ab____")
                    match = match.replace("}", "abb_____")
                    match = match.replace("\\", "abbb______")
                    match = f"{info[7].replace('.','')}_{match}"

                    o = o.replace("{"+match_org+"}", row[match])


            o = o.replace("a___", " ")
            o = o.replace("ab____", "\\{")
            o = o.replace("abb_____", "\\}")
            o = o.replace("abbb______", "\\")

            # Early exit if not valid
            #if "None" in f"{s}|{p}|{o}":
                #return set()

            generated_triples.add(f"{s}|{p}|{o}")
        return generated_triples 

    ### Without Join ###
    for row in data.itertuples(index=False):
        row: dict[str, str] = row._asdict() 
        s = ""
        p = ""
        o = ""

        # Subject
        if info[2] == "constant":
            s = info[1]
        elif info[2] == "reference":
            key = info[1]
            key = key.replace(" ", "a___")
            key = key.replace("{", "ab____")
            key = key.replace("}", "abb_____")
            key = key.replace("\\", "abbb______")
            s = row[key]
        elif info[2] == "template":
            # Get template refernces
            matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[1])
            s = info[1]
            for match in matches:
                match_org = match
                match = match.replace(" ", "a___")
                match = match.replace("{", "ab____")
                match = match.replace("}", "abb_____")
                match = match.replace("\\", "abbb______")

                s = s.replace("{"+match_org+"}", row[match])

        s = s.replace("a___", " ")
        s = s.replace("ab____", "\\{")
        s = s.replace("abb_____", "\\}")
        s = s.replace("abbb______", "\\")

        if info[4] == "constant":
            p = info[3]
        elif info[4] == "reference":
            key = info[3]
            key = key.replace(" ", "a___")
            key = key.replace("{", "ab____")
            key = key.replace("}", "abb_____")
            key = key.replace("\\", "abbb______")
            p = row[key]
        elif info[4] == "template":
            # Get template refernces
            matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[3])
            p = info[3]
            for match in matches:
                match_org = match
                match = match.replace(" ", "a___")
                match = match.replace("{", "ab____")
                match = match.replace("}", "abb_____")
                match = match.replace("\\", "abbb______")
                p = p.replace("{"+match_org+"}", row[match])
        
        p = p.replace("a___", " ")
        p = p.replace("ab____", "\\{")
        p = p.replace("abb_____", "\\}")
        p = p.replace("abbb______", "\\")

        if info[6] == "constant":
            o = info[5]
        elif info[6] == "reference":
            key = info[5]
            key = key.replace(" ", "a___")
            key = key.replace("{", "ab____")
            key = key.replace("}", "abb_____")
            key = key.replace("\\", "abbb______")
            o = row[key]
        elif info[6] == "template":
            # Get template refernces
            matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[5])
            o = info[5]
            for match in matches:
                match_org = match
                match = match.replace(" ", "a___")
                match = match.replace("{", "ab____")
                match = match.replace("}", "abb_____")
                match = match.replace("\\", "abbb______")
                o = o.replace("{"+match_org+"}", row[match])

        o = o.replace("a___", " ")
        o = o.replace("ab____", "\\{")
        o = o.replace("abb_____", "\\}")
        o = o.replace("abbb______", "\\")

        # Early exit if not valid
        #if "None" in f"{s}|{p}|{o}":
            #return set()

        generated_triples.add(f"{s}|{p}|{o}")


    return generated_triples

# Filter data
def filter_mappings(possible_triples: list[set[str]], rml_sub_graphs: list[Graph]) -> list[Graph]:
    # Pair each set of triples with the corresponding graph
    sets_and_graphs = [(s, g) for s, g in zip(possible_triples, rml_sub_graphs)]

    #
    # STEP 1: Remove duplicate
    seen_frozensets = {}         # Map frozenset to first index where it appears
    duplicate_indices = set()    # Indices of sets that are duplicates
    for i, (triples_i, graph_i) in enumerate(sets_and_graphs):
        as_frozenset = frozenset(triples_i)
        if as_frozenset not in seen_frozensets:
            # First time so store its index
            seen_frozensets[as_frozenset] = i
        else:
            # mark index to remove
            duplicate_indices.add(i)

    # reduced list without duplicates
    reduced_sets_and_graphs = [
        (triples_i, graph_i)
        for i, (triples_i, graph_i) in enumerate(sets_and_graphs)
        if i not in duplicate_indices
    ]

    ### Remove sets that are subsets of another set
    to_remove = set()
    for i, (si, gi) in enumerate(reduced_sets_and_graphs):
        for j, (sj, gj) in enumerate(reduced_sets_and_graphs):
            # Do not compare an element with itself
            if i != j and si.issubset(sj):
                # If si is a subset of sj, mark i for removal
                to_remove.add(i)

    # Build the final list (only keep maximal sets, ignoring subsets)
    final_graphs = [
        g for i, (s, g) in enumerate(reduced_sets_and_graphs)
        if i not in to_remove
    ]

    return final_graphs

def get_invar(term_map, term_map_type):
    if term_map_type == "template":
        return term_map.split("{")[0]
    elif term_map_type == "constant":
        return term_map
    elif term_map_type == "reference":
        return ""
    else:
        print("ERR")
        sys.exit(1)

def main():
    # Config
    file_path_csv = ""
    file_path_rdf = ""
    base_uri = "http://example.com/base/"
    output_file = "generated_mapping.ttl"

    parser = argparse.ArgumentParser(description="A simple CLI example")
    parser.add_argument("--csv", type=str, nargs="+", help="Paths to one or more CSV files")
    parser.add_argument("--rdf", type=str, help="The path to the RDF file")

    args = parser.parse_args()

    if args.csv:
        file_path_csv = args.csv
    if args.rdf:
        file_path_rdf = args.rdf

    if file_path_csv == "":
        print("--csv is required!")
        sys.exit(1)
    if file_path_rdf == "":
        print("--rdf is required!")
        sys.exit(1)

    print("Starting...")
    # Load RDF data
    rdf_data = parse(file_path_rdf)
    rml_sub_graphs = []
    stored_data = {}

    for csv_path in file_path_csv:
        # Load csv data
        data: pd.DataFrame = pd.read_csv(csv_path, dtype=str)
        data = data.fillna("None")

        # Rename cols to remove whitespace
        data.columns = [col.replace(" ", "a___") for col in data.columns]
        # Rename cols to remove {}
        data.columns = [col.replace("{", "ab____") for col in data.columns]
        data.columns = [col.replace("}", "abb_____") for col in data.columns]
        # Rename cols to remove \
        data.columns = [col.replace("\\", "abbb______") for col in data.columns]

        # if the column contains http://www.w3.org/2001/XMLSchema# rename for easier processing.
        data = data.replace(to_replace=r'http://www\.w3\.org/2001/XMLSchema#', value='|||', regex=True)

        # Store generated graphs for this interation
        tmp_rml_sub_graphs = []

        # Iterate over all csv data
        for row in data.itertuples(index=False):
            row: dict[str, str] = row._asdict() 
            # Sort so longer ones are first
            row = dict(sorted(row.items(), key=lambda item: len(item[1]), reverse=True))
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
                if s_term_type == "iri":
                    s = remove_base_uri(s, base_uri)

                # Iterate over all elements in the row and detect type
                s_term_map = s
                for key, value in row.items():
                    term_map, term_map_type = get_term_map_type(s_term_map, key, value, base_uri)
                    if s_term_map_type == "":
                        s_term_map = term_map
                        s_term_map_type = term_map_type
                    elif term_map_type != "constant":
                        s_term_map = term_map
                        s_term_map_type = term_map_type
                
                s_term_map = mask_string(s_term_map, row)

                # Rename inserted values from pandas headline
                s_term_map = s_term_map.replace("a___", " ")
                s_term_map = s_term_map.replace("ab____", "\\{")
                s_term_map = s_term_map.replace("abb_____", "\\}")
                s_term_map = s_term_map.replace("abbb______", "\\")

                ## Handle predicate ##
                p_term_type = get_term_type(p)
                p_term_map = ""
                p_term_map_type = ""

                # clean p value
                p = clean_entry(p)
                if p_term_type == "iri":
                    p = remove_base_uri(p, base_uri)

                p_term_map = p
                for key, value in row.items():
                    term_map, term_map_type = get_term_map_type(p_term_map, key, value, base_uri)
                    if p_term_map_type == "":
                        p_term_map = term_map
                        p_term_map_type = term_map_type
                    elif term_map_type != "constant":
                        p_term_map = term_map
                        p_term_map_type = term_map_type

                # Rename inserted values from pandas headline
                p_term_map = p_term_map.replace("a___", " ")
                p_term_map = p_term_map.replace("ab____", "{")
                p_term_map = p_term_map.replace("abb_____", "}")
                p_term_map = p_term_map.replace("abbb______", "\\")

                ## Handle object ##
                o_term_type = get_term_type(o)
                o_term_map = ""
                o_term_map_type = ""

                # clean o value
                o = clean_entry(o)
                if o_term_type == "iri":
                    o = remove_base_uri(o, base_uri)

                o_term_map = o
                for key, value in row.items():
                    term_map, term_map_type = get_term_map_type(o_term_map, key, value, base_uri)
                    if o_term_map_type == "":
                        o_term_map = term_map
                        o_term_map_type = term_map_type
                    elif term_map_type != "constant":
                        o_term_map = term_map
                        o_term_map_type = term_map_type

                # Mask string
                o_term_map = mask_string(o_term_map, row)

                # Rename inserted values from pandas headline
                o_term_map = o_term_map.replace("a___", " ")
                o_term_map = o_term_map.replace("ab____", "\\\\{")
                o_term_map = o_term_map.replace("abb_____", "\\\\}")
                o_term_map = o_term_map.replace("abbb______", "\\")

                
                ## Handle graph ##
                g_term_type = "iri"
                g_term_map = ""
                g_term_map_type = ""

                # clean o value
                g = clean_entry(g)
                if g_term_type == "iri":
                    g = remove_base_uri(g, base_uri)
                
                if g == "":
                    g_term_type = ""
                    g_term_map = ""
                    g_term_map_type = ""
                else:
                    g_term_map = g
                    for key, value in row.items():
                        term_map, term_map_type = get_term_map_type(g_term_map, key, value, base_uri)
                        if g_term_map_type == "":
                            g_term_map = term_map
                            g_term_map_type = term_map_type
                        elif term_map_type != "constant":
                            g_term_map = term_map
                            g_term_map_type = term_map_type
                
                # Rename inserted values from pandas headline
                g_term_map = g_term_map.replace("a___", " ")
                g_term_map = g_term_map.replace("ab____", "{")
                g_term_map = g_term_map.replace("abb_____", "}")
                g_term_map = g_term_map.replace("abbb______", "\\")


                ## Handle datatype ##
                raw_o_value = element.o
                # Check for datatype
                res = raw_o_value.split("^^")
                if len(res) != 2:
                    data_type_term_type = ""
                    data_type_term_map = ""
                    data_type_term_map_type = ""
                else:
                    data_type_string = res[1]
                    # clean data_type_string value
                    data_type_string = clean_entry(data_type_string)

                    data_type_term_type = "iri"
                    data_type_term_map = ""
                    data_type_term_map_type = ""

                    data_type_term_map = data_type_string
                    # Remove xsd schema prefix
                    data_type_term_map = data_type_term_map.replace("http://www.w3.org/2001/XMLSchema#", "|||")

                    for key, value in row.items():
                        term_map, term_map_type = get_term_map_type(data_type_term_map, key, value, base_uri)
                        if data_type_term_map_type == "":
                            data_type_term_map = term_map
                            data_type_term_map_type = term_map_type
                        elif term_map_type != "constant":
                            data_type_term_map = term_map
                            data_type_term_map_type = term_map_type
                    
                    # Rename inserted values from pandas headline
                    data_type_term_map = data_type_term_map.replace("a___", " ")
                    data_type_term_map = data_type_term_map.replace("ab____", "{")
                    data_type_term_map = data_type_term_map.replace("abb_____", "}")
                    data_type_term_map = data_type_term_map.replace("abbb______", "\\")


                    # Add xsd prefix back
                    data_type_term_map = data_type_term_map.replace("|||", "http://www.w3.org/2001/XMLSchema#")

                ## Build rml graph ##
                tmp_rml_sub_graph = graph_builder.build_sub_graph(csv_path, s_term_map, s_term_map_type, s_term_type,\
                                                            p_term_map, p_term_map_type, p_term_type, \
                                                            o_term_map, o_term_map_type, o_term_type, \
                                                            g_term_type, g_term_map, g_term_map_type, \
                                                            data_type_term_type, data_type_term_map, data_type_term_map_type)
                if not isDuplicateGraph(tmp_rml_sub_graph, tmp_rml_sub_graphs):
                    tmp_rml_sub_graphs.append(tmp_rml_sub_graph)

        # Store data
        stored_data[csv_path] = data

        ### FILTER ###
        # Extract possible triples
        possible_triples = []
        for g in tmp_rml_sub_graphs:
            info = extract_information(g)
            res = generate_expected_triple(data, info)
            possible_triples.append(res)
        filtered_graphs = filter_mappings(possible_triples, tmp_rml_sub_graphs)

        # Print the final filtered graphs
        for fg in filtered_graphs:
            rml_sub_graphs.append(fg)

    ### Identify joins
    join_graphs = []
    graphs_to_remove = []
    for g in rml_sub_graphs:
        info_g = extract_information(g)
        # Compare to all other files
        for g2 in rml_sub_graphs:
            info_g2 = extract_information(g2)
            # Do not compare with itself
            if info_g2 == info_g:
                continue

            # source files must be different
            #if info_g[0] == info_g2[0]:
                #continue

            # second part must be constant in subject
            if info_g2[2] != "constant":
                continue
            
            # predicate and prediacte term type must be the same
            if not (info_g[3] == info_g2[3] and info_g[4] == info_g2[4]):
                continue

            # object term type must be the same
            if info_g[6] != info_g2[6]:
                continue

            # Object map can not be constant
            if info_g[6] == "constant" or info_g2[6] == "constant":
                continue

            # object term map invar must be the same
            invar_g = get_invar(info_g[5], info_g[6])
            invar_g2 = get_invar(info_g2[5], info_g2[6])
            if invar_g != invar_g2:
                continue
            
            # If we arrive here, generate the mapping
            new_join_graph = graph_builder.build_sub_graph_join(g, g2)
            join_graphs.append(new_join_graph)
            graphs_to_remove.append(g)
            graphs_to_remove.append(g2)
            
    # Remvoe graphs that are used in join
    for g in graphs_to_remove:
        rml_sub_graphs = remove_graph(g, rml_sub_graphs)

    ### Filter combined result of all input files
    # Prepare data
    possible_triples = []
    new_graphs = []
    for g in rml_sub_graphs:
        info = extract_information(g)
        data = stored_data[info[0]]
        res = generate_expected_triple(data, info)
        add = True
        for element in res:
            if "None" in element:
                add = False
        if add == False:
            continue
        possible_triples.append(res)
        new_graphs.append(g)
    rml_sub_graphs = new_graphs

    # Prepare joins 
    for g1, g2 in join_graphs:
        info = extract_information_join(g1,g2)
        data = stored_data[info[0]]
        data2 = stored_data[info[7]]
        res = generate_expected_triple(data, info, data2)
        # Add to original data
        g = g1 + g2
        rml_sub_graphs.append(g)
        possible_triples.append(res)

    # Filter data
    filtered_graphs = filter_mappings(possible_triples, rml_sub_graphs)
    rml_sub_graphs = filtered_graphs

    for i in range(len(rml_sub_graphs)):
        g = rml_sub_graphs[i]
        t = possible_triples[i]

        g = encode_uris(g)
        print(g.serialize())
        print(t)
        print("=====")

    # Check if all triples are expected
    filtered_graphs = []
    for g in rml_sub_graphs:
        if not is_join_graph(g):
            info = extract_information(g)
            data = stored_data[info[0]]
            res = generate_expected_triple(data, info)
        else:
            # Identify graphs pairs
            g1 = None
            g2 = None
            for g1_ref, g2_ref in join_graphs:
                g_ref = g1_ref + g2_ref
                if g_ref.isomorphic(g):
                    g1 = g1_ref
                    g2 = g2_ref
                    break

            if g1 == None and g2 == None:
                print("Error in separating graphs!")
                sys.exit(1)
            
            info = extract_information_join(g1,g2)
            data = stored_data[info[0]]
            data2 = stored_data[info[7]]
            
            res = generate_expected_triple(data, info, data2)

        cnt_found = 0
        for entry in res:
            for rdf in rdf_data:
                s = clean_entry(rdf.s)
                p = clean_entry(rdf.p)
                o = clean_entry(rdf.o)

                # Handle base uri
                if base_uri in s:
                    s = s.replace(base_uri, "")
                if base_uri in p:
                    p = p.replace(base_uri, "")
                if base_uri in o:
                    o = o.replace(base_uri, "")

                comp = f"{s}|{p}|{o}"
                if comp == entry:
                    cnt_found+=1

        if len(res) == cnt_found:
            filtered_graphs.append(g)
    # Final result
    rml_sub_graphs = filtered_graphs

    # Print output
    result_graph = Graph()
    # Set RML namespace
    RML = Namespace("http://w3id.org/rml/")
    result_graph.bind("rml", RML)
    for rml_sub_graph in rml_sub_graphs:
            result_graph += rml_sub_graph

    str_result_graph = result_graph.serialize(format="turtle")
    
    # Add base uri if needed
    if base_uri != "":
        str_result_graph = f"@base <{base_uri}> .\n" + str_result_graph

    # Adjust ////
    str_result_graph = str_result_graph.replace("\\\\","\\")
    str_result_graph = str_result_graph.replace("\\\\\\\\","\\\\")


    # Write to file.
    with open(output_file, "w") as file:
        file.write(str_result_graph)

    print("Finished. Generated mapping stored in:", output_file)

if __name__ == "__main__":  
    main()