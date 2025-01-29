import pandas as pd
from rdflib import Graph, Namespace
import sys
from dataclasses import dataclass
import graph_builder
import re
import argparse
import subprocess

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
def generate_expected_triple(data: pd.DataFrame, info) -> set[str]:
    # Generate data
    generated_triples = set()

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

        generated_triples.add(f"{s}|{p}|{o}")


    return generated_triples

# Filter data
def filter_mappings(possible_triples: list[set[str]], rml_sub_graphs: list[Graph]) -> tuple[list[Graph]]:
    # Track unique sets of triples
    unique_triples = set()
    filtered_graphs: list[Graph] = []

    for i in range(len(possible_triples)):
        triples = possible_triples[i]
        g = rml_sub_graphs[i]

        # Check if this triple set is already stored (duplicate) or is a subset of another
        is_subset = False
        for other_triples in unique_triples:
            if triples.issubset(other_triples) or triples == other_triples:  # Handles subsets and duplicates
                is_subset = True
                break

        # If it's not a subset or duplicate, keep it and add it to the unique set
        if not is_subset:
            unique_triples.add(frozenset(triples))
            filtered_graphs.append(g)

    return filtered_graphs

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
                    term_map, term_map_type = get_term_map_type(s_term_map, key, value)
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
                    term_map, term_map_type = get_term_map_type(p_term_map, key, value)
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
                    term_map, term_map_type = get_term_map_type(o_term_map, key, value)
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
                        term_map, term_map_type = get_term_map_type(g_term_map, key, value)
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
                        term_map, term_map_type = get_term_map_type(data_type_term_map, key, value)
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


    ### Filter combined result of all input files
    possible_triples = []
    for g in rml_sub_graphs:
        info = extract_information(g)
        data = stored_data[info[0]]
        res = generate_expected_triple(data, info)
        possible_triples.append(res)

    filtered_graphs = filter_mappings(possible_triples, rml_sub_graphs)
    rml_sub_graphs = filtered_graphs

    # Check if all triples are expected
    filtered_graphs = []
    for g in rml_sub_graphs:
        info = extract_information(g)
        data = stored_data[info[0]]
        res = generate_expected_triple(data, info)

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