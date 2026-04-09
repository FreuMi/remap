import pandas as pd
from io import StringIO
from rdflib import Graph, Namespace, URIRef
import sys
import re
from urllib.parse import urlparse, quote, unquote
from .vocabulary import *
import logging
from .helper import *

from .graph_builder import build_sub_graph_join, build_sub_graph, get_term_type_of_graph
from .utils import parse, parse_rdf_as_nt
import json
import io
import csv
from collections.abc import Mapping
from itertools import product

DEBUG = False

# Suppress warnings of rdflib
logging.getLogger("rdflib").setLevel(logging.ERROR)  

def encode_uris(graph):
    updated_graph = Graph()
    for s, p, o in graph:
        # Encode only if the subject or object is a URI
        s = URIRef(quote(str(s), safe=":/#")) if isinstance(s, URIRef) else s
        o = URIRef(quote(str(o), safe=":/#")) if isinstance(o, URIRef) else o
        updated_graph.add((s, p, o))
    return updated_graph

def check_protected_iris(iri: str) -> tuple[bool, str, str]:
    protected_iris = [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://w3id.org/rml/",
        "http://www.w3.org/2001/XMLSchema#",
    ]
    	
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

def is_unsafe_iri_value(uri: str) -> bool:
    return quote(uri, safe=":/#%[]@!$&'()*+,;=-._~") != uri

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
        if o == REF_OBJ_MAP:
            return True
    return False

def isIn(sub, s):
    if s.count("{") > 1:
        return sub in s

    cleaned_s = re.sub(r'\{[^}]*\}', '', s)
    return sub in cleaned_s

# Itentify term_map_type of (constant, reference, template) and generate term_map
def get_term_map_type(rdf_term: str, csv_header: str, csv_data: str, base_uri: str) -> tuple[str, str]:
    rdf_term_map_type = ""
    full_rdf_term = rdf_term

    # Check for protected iri
    is_protected, rest_iri, protected_prefix = check_protected_iris(rdf_term)

    if is_protected:
        rdf_term = rest_iri

    org_csv_data = csv_data
    normalized_csv_data = csv_data
    if is_valid_uri(org_csv_data):
        normalized_csv_data = remove_base_uri(org_csv_data, base_uri)

    full_iri_match = is_protected and org_csv_data == full_rdf_term
    comparison_csv_data = normalized_csv_data
    encoded_csv_data = quote(org_csv_data, safe="")
    if encoded_csv_data != org_csv_data and isIn(encoded_csv_data, rdf_term):
        comparison_csv_data = encoded_csv_data

    if not full_iri_match and not isIn(comparison_csv_data, rdf_term):
        rdf_term_map_type = "constant"
    else:
        matched_csv_data = org_csv_data if full_iri_match else comparison_csv_data
        exact_match = full_iri_match or rdf_term == matched_csv_data
        if not exact_match:
            rdf_term_map_type = "template"
        elif comparison_csv_data == encoded_csv_data and encoded_csv_data != org_csv_data:
            rdf_term_map_type = "template"
        elif is_protected and exact_match and not full_iri_match:
            rdf_term_map_type = "template"
        elif is_valid_uri(org_csv_data):
            rdf_term_map_type = "reference"
        elif is_valid_uri(base_uri + org_csv_data):
            rdf_term_map_type = "reference"
        elif exact_match:
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
        term_map = rdf_term.replace(matched_csv_data, "{"+csv_header+"}", 1)
    elif rdf_term_map_type == "reference":
        term_map = csv_header
    else:
        print("Error generating term_map!")
        sys.exit(1)

    return term_map, rdf_term_map_type
    
def get_term_type(data: str) -> str:
    if "^^" in data:
        data = data.split("^^")[0]
    elif "@" in data:
        data = data.split("@")[0]

    res = ""
    if isURI(data):
        cleaned = clean_entry(data)
        res = "unsafeiri" if is_unsafe_iri_value(cleaned) else "iri"
    elif isLiteral(data):
        res = "literal"
    elif isBlanknode(data):
        res = "blanknode"
    else:
        print("Error detecting s_term_type. Got:", data)
        sys.exit(1)

    return res

#################################################################################################

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
        elif "@" in entry:
            split_data = entry.split("@")
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


def rdf_term_variants(raw_term: str, cleaned_term: str, base_uri: str) -> list[str]:
    variants = [cleaned_term]
    if raw_term != "" and isURI(raw_term) and base_uri in cleaned_term:
        stripped = cleaned_term.replace(base_uri, "")
        if stripped not in variants:
            variants.append(stripped)
        decoded = unquote(stripped)
        if decoded not in variants:
            variants.append(decoded)
    return variants


def build_actual_triple_variants(rdf, base_uri: str, blank_subject: bool = False) -> set[str]:
    s = clean_entry(rdf.s)
    p = clean_entry(rdf.p)
    o = clean_entry(rdf.o)
    g = clean_entry(rdf.g)
    datatype = ""
    language = ""
    if "^^" in rdf.o:
        datatype = clean_entry(rdf.o.split("^^", 1)[1])
    elif "@" in rdf.o:
        language = rdf.o.rsplit("@", 1)[1]

    subject_variants = rdf_term_variants(rdf.s, s, base_uri)
    if blank_subject and isBlanknode(rdf.s):
        subject_variants = ["BLANK_NODE"]

    predicate_variants = rdf_term_variants(rdf.p, p, base_uri)
    object_variants = rdf_term_variants(rdf.o, o, base_uri)
    graph_variants = rdf_term_variants(rdf.g, g, base_uri) if rdf.g != "" else [g]
    datatype_variants = rdf_term_variants(
        f"<{datatype}>", datatype, base_uri
    ) if datatype != "" else [""]
    language_variants = [language]

    return {
        f"{subject}|{predicate}|{obj}|{graph}|{dt}|{lang}"
        for subject, predicate, obj, graph, dt, lang in product(
            subject_variants,
            predicate_variants,
            object_variants,
            graph_variants,
            datatype_variants,
            language_variants,
        )
    }


def build_canonical_actual_triple(rdf) -> str:
    datatype = ""
    language = ""
    if "^^" in rdf.o:
        datatype = clean_entry(rdf.o.split("^^", 1)[1])
    elif "@" in rdf.o:
        language = rdf.o.rsplit("@", 1)[1]
    return (
        f"{clean_entry(rdf.s)}|{clean_entry(rdf.p)}|"
        f"{clean_entry(rdf.o)}|{clean_entry(rdf.g)}|{datatype}|{language}"
    )

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
    elif "@" in value:
        split_value = value.split("@")
        if len(split_value) != 2:
            print("Error splitting data! Found", split_value)
            sys.exit(1)
        value = split_value[0]
        
    if value[0] == "\"" and value[-1] == "\"":
        return True
    else: 
        return False

def extract_information(graph: Graph) -> tuple[str,str,str,str,str,str,str,str]:
    # Get source path 
    new_path = getPath(graph)
    # Get subject
    sub, subject_type = getSubject(graph)
    # Get predicate
    pred, predicate_type = getPredicate(graph)
    # Get object
    obj, object_type = getObject(graph)
    data_type_map, data_type_type = getDatatype(graph)
    language_map, language_type = getLanguage(graph)
    # Get graph
    gra, graph_type = getGraph(graph)

    return (
        new_path, sub, subject_type, pred, predicate_type, obj, object_type,
        gra, graph_type, "", data_type_map, data_type_type, language_map, language_type
    )

def get_parent(g: Graph) -> str:
    for s,p,o in g:
        if p == PARENT:
            return str(o)

def get_child(g: Graph) -> str:
    for s,p,o in g:
        if p == CHILD:
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
    data_type_map = ""
    data_type_type = ""
    language_map = ""
    language_type = ""
    # Get graph
    gra, graph_type = getGraph(graph1)
    child = get_child(graph1)
    parent = get_parent(graph1)

    return (
        new_path, sub, subject_type, pred, predicate_type, obj, object_type, gra,
        graph_type, new_path2, child, parent, data_type_map, data_type_type,
        language_map, language_type
    )

def isDuplicateGraph(graph: Graph, all_graphs: list[Graph]) -> bool:
    reference_info = extract_information(graph)
    reference_term_types = (
        get_term_type_of_graph(graph, "s"),
        get_term_type_of_graph(graph, "o"),
    )
    for stored_graph in all_graphs:
        # Check if source, subject, predicate, and object are the same.
        stored_information = extract_information(stored_graph)
        stored_term_types = (
            get_term_type_of_graph(stored_graph, "s"),
            get_term_type_of_graph(stored_graph, "o"),
        )
        if reference_info == stored_information and reference_term_types == stored_term_types:
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
            tmp_input_data += r"\\{"
            continue
        elif char == "}":
            tmp_input_data += r"\\}"
            continue
        tmp_input_data += char

    # replace |||key||| with {key}
    for key, _ in row.items():
        tmp_input_data = tmp_input_data.replace(f"|||{key}|||", f"{{{key}}}")

    return tmp_input_data


def sanitize_lookup_key(key: str) -> str:
    key = key.replace(r"\{", "{")
    key = key.replace(r"\}", "}")
    key = key.replace(" ", "a___")
    key = key.replace("{", "ab____")
    key = key.replace("}", "abb_____")
    key = key.replace("\\", "abbb______")
    return key

# Generate expected triple
def generate_expected_triple(data: pd.DataFrame, info, data2: pd.DataFrame = pd.DataFrame()) -> set[str]:
    # Generate data
    generated_triples = set()
    data_type_map = info[12] if info[9] != "" else info[10]
    data_type_type = info[13] if info[9] != "" else info[11]
    language_map = info[14] if info[9] != "" else info[12]
    language_type = info[15] if info[9] != "" else info[13]

    ### With Join ###
    if info[9] != "":
        ## Perform Join ##
        # Rename columns
        data = data.rename(columns=lambda col: f"{info[0].replace('.','')}_{col}")
        data2 = data2.rename(columns=lambda col: f"{info[9].replace('.','')}_{col}")
        
        child = f"{info[0].replace('.','')}_{info[10]}"
        parent = f"{info[9].replace('.','')}_{info[11]}"

        # Perform join
        join_result_df = pd.merge(
            data,
            data2,
            left_on=child,
            right_on=parent,
            suffixes=("_RxPxx", "_SxPxx"),
        )

        if info[0] == info[9]:
            # Handle self join
            columns_to_drop = [col for col in join_result_df.columns if col.endswith("_SxPxx")]
            simplified_df = join_result_df.drop(columns=columns_to_drop)
            simplified_df.columns = [col.replace("_RxPxx", "") for col in simplified_df.columns]
            join_result_df = simplified_df

        # Iterate over data
        for _, row in join_result_df.iterrows():
            try:
                row = row.to_dict()          
                s = ""
                p = ""
                o = ""
                g = ""

                # Subject
                if info[2] == "none":
                    s = "BLANK_NODE"
                elif info[2] == "constant":
                    s = info[1]
                elif info[2] == "reference":
                    key = sanitize_lookup_key(info[1])
                    s = row[key]
                elif info[2] == "template":
                    # Get template refernces
                    matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[1])
                    s = info[1]
                    for match in matches:
                        match_org = match
                        match = sanitize_lookup_key(match)
                        match = f"{info[0].replace('.','')}_{match}"
                        s = s.replace("{"+match_org+"}", row[match])

                s = s.replace("a___", " ")
                s = s.replace("ab____", "\\{")
                s = s.replace("abb_____", "\\}")
                s = s.replace("abbb______", "\\")

                if info[4] == "constant":
                    p = info[3]
                elif info[4] == "reference":
                    key = sanitize_lookup_key(info[3])
                    p = row[key]
                elif info[4] == "template":
                    # Get template refernces
                    matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[3])
                    p = info[3]
                    for match in matches:
                        match_org = match
                        match = sanitize_lookup_key(match)
                        p = p.replace("{"+match_org+"}", row[match])
                
                p = p.replace("a___", " ")
                p = p.replace("ab____", "\\{")
                p = p.replace("abb_____", "\\}")
                p = p.replace("abbb______", "\\")

                if info[6] == "constant":
                    o = info[5]
                elif info[6] == "reference":
                    key = sanitize_lookup_key(info[5])
                    o = row[key]
                elif info[6] == "template":
                    # Get template refernces
                    matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[5])
                    o = info[5]
                    for match in matches:
                        match_org = match
                        match = sanitize_lookup_key(match)
                        match = f"{info[9].replace('.','')}_{match}"
                        o = o.replace("{"+match_org+"}", row[match])


                o = o.replace("a___", " ")
                o = o.replace("ab____", "\\{")
                o = o.replace("abb_____", "\\}")
                o = o.replace("abbb______", "\\")
                
                datatype = ""
                if data_type_type == "constant":
                    datatype = data_type_map
                elif data_type_type == "reference":
                    key = sanitize_lookup_key(data_type_map)
                    datatype = row[key]
                elif data_type_type == "template":
                    matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', data_type_map)
                    datatype = data_type_map
                    for match in matches:
                        match_org = match
                        match = sanitize_lookup_key(match)
                        datatype = datatype.replace("{"+match_org+"}", row[match])

                language = ""
                if language_type == "constant":
                    language = language_map

                # Handle graph
                if info[8] == "constant":
                    g = info[7]
                elif info[8] == "reference":
                    key = sanitize_lookup_key(info[7])
                    g = row[key]
                elif info[8] == "template":
                    # Get template refernces
                    matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[7])
                    g = info[7]
                    for match in matches:
                        match_org = match
                        match = sanitize_lookup_key(match)
                        g = g.replace("{"+match_org+"}", row[match])

                g = g.replace("a___", " ")
                g = g.replace("ab____", "\\{")
                g = g.replace("abb_____", "\\}")
                g = g.replace("abbb______", "\\")
                g = g.replace(r"\\{", "{")
                g = g.replace(r"\\}", "}")

                generated_triples.add(f"{s}|{p}|{o}|{g}|{datatype}|{language}")
            except KeyError:
                pass
        return generated_triples 

    ### Without Join ###
    for _, row in data.iterrows():
        try:
            row = row.to_dict()
            s = ""
            p = ""
            o = ""
            g = ""

            # Subject
            if info[2] == "none":
                s = "BLANK_NODE"
            elif info[2] == "constant":
                s = info[1]
            elif info[2] == "reference":
                key = sanitize_lookup_key(info[1])
                s = row[key]
            elif info[2] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[1])
                s = info[1]
                for match in matches:
                    match_org = match
                    match = sanitize_lookup_key(match)

                    s = s.replace("{"+match_org+"}", row[match])

            s = s.replace("a___", " ")
            s = s.replace("ab____", "\\{")
            s = s.replace("abb_____", "\\}")
            s = s.replace("abbb______", "\\")

            if info[4] == "constant":
                p = info[3]
            elif info[4] == "reference":
                key = sanitize_lookup_key(info[3])
                p = row[key]
            elif info[4] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[3])
                p = info[3]
                for match in matches:
                    match_org = match
                    match = sanitize_lookup_key(match)
                    p = p.replace("{"+match_org+"}", row[match])
            
            p = p.replace("a___", " ")
            p = p.replace("ab____", "\\{")
            p = p.replace("abb_____", "\\}")
            p = p.replace("abbb______", "\\")

            if info[6] == "constant":
                o = info[5]
            elif info[6] == "reference":
                key = sanitize_lookup_key(info[5])
                o = row[key]
            elif info[6] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[5])
                o = info[5]
                for match in matches:
                    match_org = match
                    match = sanitize_lookup_key(match)
                    o = o.replace("{"+match_org+"}", row[match])

            o = o.replace("a___", " ")
            o = o.replace("ab____", "\\{")
            o = o.replace("abb_____", "\\}")
            o = o.replace("abbb______", "\\")
            o = o.replace(r"\\{", "{")
            o = o.replace(r"\\}", "}")

            datatype = ""
            if data_type_type == "constant":
                datatype = data_type_map
            elif data_type_type == "reference":
                key = sanitize_lookup_key(data_type_map)
                datatype = row[key]
            elif data_type_type == "template":
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', data_type_map)
                datatype = data_type_map
                for match in matches:
                    match_org = match
                    match = sanitize_lookup_key(match)
                    datatype = datatype.replace("{"+match_org+"}", row[match])

            language = ""
            if language_type == "constant":
                language = language_map

            if info[8] == "constant":
                g = info[7]
            elif info[8] == "reference":
                key = sanitize_lookup_key(info[7])
                g = row[key]
            elif info[8] == "template":
                # Get template refernces
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', info[7])
                g = info[7]
                for match in matches:
                    match_org = match
                    match = sanitize_lookup_key(match)
                    g = g.replace("{"+match_org+"}", row[match])

            g = g.replace("a___", " ")
            g = g.replace("ab____", "\\{")
            g = g.replace("abb_____", "\\}")
            g = g.replace("abbb______", "\\")
            g = g.replace(r"\\{", "{")
            g = g.replace(r"\\}", "}")

            generated_triples.add(f"{s}|{p}|{o}|{g}|{datatype}|{language}")
        except KeyError:
            pass

    return generated_triples

# Filter data
def mapping_generality_score(graph: Graph) -> tuple[int, int, int, int]:
    rank = {"constant": 0, "none": 1, "template": 2, "reference": 3, "join": 4}
    term_type_rank = {"literal": 0, "iri": 1, "unsafeiri": 2, "blanknode": 3}
    if is_join_graph(graph):
        _, subject_type = getSubject(graph)
        _, predicate_type = getPredicate(graph)
        _, graph_type = getGraph(graph)
        return (
            rank.get(subject_type, -1),
            rank.get(predicate_type, -1),
            rank["join"],
            rank.get(graph_type or "constant", -1),
            0,
            0,
        )

    info = extract_information(graph)
    subject_term_type = get_term_type_of_graph(graph, "s")
    object_term_type = get_term_type_of_graph(graph, "o")
    return (
        rank.get(info[2], -1),
        rank.get(info[4], -1),
        rank.get(info[6], -1),
        rank.get(info[8], -1),
        term_type_rank.get(subject_term_type, -1),
        term_type_rank.get(object_term_type, -1),
    )


def filter_mappings(possible_triples: list[set[str]], rml_sub_graphs: list[Graph]) -> list[Graph]:
    # Pair each set of triples with the corresponding graph
    sets_and_graphs = [(s, g) for s, g in zip(possible_triples, rml_sub_graphs)]

    #
    # STEP 1: Remove duplicate
    seen_frozensets = {}         # Map frozenset to best index where it appears
    duplicate_indices = set()    # Indices of sets that are duplicates
    for i, (triples_i, graph_i) in enumerate(sets_and_graphs):
        as_frozenset = frozenset(triples_i)
        if as_frozenset not in seen_frozensets:
            seen_frozensets[as_frozenset] = i
        else:
            previous_index = seen_frozensets[as_frozenset]
            previous_graph = sets_and_graphs[previous_index][1]
            if mapping_generality_score(graph_i) > mapping_generality_score(previous_graph):
                duplicate_indices.add(previous_index)
                seen_frozensets[as_frozenset] = i
            else:
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


def normalize_literal_term_map(term_map: str, term_map_type: str, term_type: str) -> tuple[str, str]:
    if term_type != "literal" or term_map_type != "template":
        return term_map, term_map_type

    match = re.fullmatch(r"\{([^{}]+)\}", term_map)
    if match is None:
        return term_map, term_map_type

    return match.group(1), "reference"


def normalize_json_template_placeholders(term_map: str) -> str:
    return term_map


def normalize_escaped_json_iri_reference(
    term_map: str,
    term_map_type: str,
    term_type: str,
    base_uri: str,
) -> tuple[str, str]:
    if term_map_type != "reference" or term_type != "iri":
        return term_map, term_map_type
    if "\\" not in term_map:
        return term_map, term_map_type
    if base_uri == "":
        return term_map, term_map_type
    return f"{base_uri}{{{term_map}}}", "template"


def collapse_expressionless_blanknode_subjects(graphs: list[Graph]) -> None:
    graphs_by_source = {}
    RML = Namespace("http://w3id.org/rml/")

    for graph in graphs:
        if is_join_graph(graph):
            continue
        info = extract_information(graph)
        graphs_by_source.setdefault(info[0], []).append((graph, info))

    for source_graphs in graphs_by_source.values():
        has_non_constant_blanknode = any(
            info[2] in {"template", "reference", "none"}
            and any((sm, RML.termType, RML.BlankNode) in graph for sm in graph.objects(None, RML.subjectMap))
            for graph, info in source_graphs
        )
        if has_non_constant_blanknode:
            continue

        for graph, info in source_graphs:
            if info[2] != "constant":
                continue
            for sm in list(graph.objects(None, RML.subjectMap)):
                if (sm, RML.termType, RML.BlankNode) in graph:
                    graph.remove((sm, RML.constant, None))

##########################################################################################

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

def flatten_dict(d, parent_key="", sep="."):
    if isinstance(d, str):
        d = json.loads(d)
    items = {}
    for k, v in d.items():
        new_key = append_json_path_segment(parent_key, k)
        if isinstance(v, Mapping):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v, ensure_ascii=False)
        else:
            items[new_key] = v

    return items


def is_simple_json_key(key: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is not None


def append_json_path_segment(parent_key: str, key: str) -> str:
    escaped_key = key.replace("\\", "\\\\").replace("'", "\\'")
    if parent_key == "":
        if is_simple_json_key(key):
            return key
        return f"['{escaped_key}']"
    if is_simple_json_key(key):
        return f"{parent_key}.{key}"
    return f"{parent_key}['{escaped_key}']"


def extract_json_records(raw_json_text: str) -> tuple[list[dict], str]:
    data = json.loads(raw_json_text)

    if isinstance(data, list):
        return data, "$[*]"

    if isinstance(data, dict):
        list_entries = [
            (key, value)
            for key, value in data.items()
            if isinstance(value, list)
        ]
        if len(list_entries) == 1 and all(
            isinstance(entry, Mapping) for entry in list_entries[0][1]
        ):
            key, value = list_entries[0]
            return value, f"$.{key}[*]"
        return [data], "$"

    return [{"value": data}], "$"


def json_to_dataframe(raw_json_text: str) -> tuple[pd.DataFrame, str]:
    records, iterator = extract_json_records(raw_json_text)
    flattened_records = []
    for record in records:
        if isinstance(record, Mapping):
            flat_record = flatten_dict(record)
        else:
            flat_record = {"value": record}
        flattened_records.append(
            {
                f"${key if key.startswith('[') else '.' + key}": "None" if value is None else str(value)
                for key, value in flat_record.items()
            }
        )

    if not flattened_records:
        return pd.DataFrame(), iterator

    return pd.DataFrame(flattened_records, dtype=str), iterator


def build_empty_mapping(
    source_path: str,
    is_json_data: bool,
    json_iterator: str,
    base_uri: str,
) -> str:
    if is_json_data:
        subject_map = "$.missing_subject"
        predicate_map = "$.missing_predicate"
        object_map = "$.missing_object"
    else:
        subject_map = "missing_subject"
        predicate_map = "missing_predicate"
        object_map = "missing_object"

    empty_graph = build_sub_graph(
        source_path,
        is_json_data,
        json_iterator,
        subject_map,
        "reference",
        "iri",
        predicate_map,
        "reference",
        "iri",
        object_map,
        "reference",
        "literal",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )
    result_graph = Graph()
    result_graph.bind("rml", Namespace("http://w3id.org/rml/"))
    result_graph += empty_graph
    str_result_graph = result_graph.serialize(format="turtle")
    if base_uri != "":
        str_result_graph = f"@base <{base_uri}> .\n" + str_result_graph
    return str_result_graph

def generate_rml_from_file(file_path_rdf: str, file_path_csv, base_uri: str = "http://example.com/base/", debug_log = False):

    # Load RDF
    with open(file_path_rdf, "r") as f:
        raw_rdf_data = f.read()

    raw_csv_data = []
    for csv_path in file_path_csv:
        with open(csv_path, "r") as f:
            data = f.read()
            raw_csv_data.append(data)

    
    return generate_rml(raw_rdf_data, raw_csv_data, base_uri, file_path_csv, debug_log)

def generate_rml(raw_rdf_data: str, csv_data, base_uri: str = "http://example.com/base/", csv_paths = None, debug_log = False) -> str:
    DEBUG = debug_log
    
    if DEBUG:
        print("Starting...")
    # Load RDF data
    ntriple = parse_rdf_as_nt(raw_rdf_data)

    # Parse data
    rdf_data = parse(ntriple)

    rml_sub_graphs = []
    stored_data = {}

    if csv_paths is None:
        csv_paths = []
    else:
        csv_paths = list(csv_paths)

    if len(csv_paths) == 0:
        for i in range(len(csv_data)):
            csv_paths.append(f"data{i}")

    if len(rdf_data) == 0:
        csv_text = csv_data[0] if len(csv_data) > 0 else ""
        source_path = csv_paths[0] if len(csv_paths) > 0 else "data0"
        if is_valid_json(csv_text):
            _, json_iterator = json_to_dataframe(csv_text)
            return build_empty_mapping(source_path, True, json_iterator, base_uri)
        return build_empty_mapping(source_path, False, "$", base_uri)

    for i in range(len(csv_data)):
        csv_text = csv_data[i]
        is_json_data = False
        json_iterator = "$"

        # Test if json
        if is_valid_json(csv_text):
            is_json_data = True
            data, json_iterator = json_to_dataframe(csv_text)
        else:
            data = pd.read_csv(StringIO(csv_text), dtype=str)

        csv_path = csv_paths[i]
        data = data.drop_duplicates()
        data = data.fillna("None")

        # Rename cols to remove whitespace
        data.columns = [col.replace(" ", "a___") for col in data.columns]
        # Rename cols to remove {}
        data.columns = [col.replace("{", "ab____") for col in data.columns]
        data.columns = [col.replace("}", "abb_____") for col in data.columns]
        # Rename cols to remove \
        data.columns = [col.replace("\\", "abbb______") for col in data.columns]

        # Store generated graphs for this interation
        tmp_rml_sub_graphs = []

        if DEBUG:
            print(f"Processing {len(data)} entries...")
    
        # Iterate over all csv data
        for _, row in data.iterrows():
            row = row.to_dict()
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
                if is_json_data and s_term_map_type == "template":
                    s_term_map = normalize_json_template_placeholders(s_term_map)
                if is_json_data:
                    s_term_map, s_term_map_type = normalize_escaped_json_iri_reference(
                        s_term_map, s_term_map_type, s_term_type, base_uri
                    )

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
                if is_json_data and p_term_map_type == "template":
                    p_term_map = normalize_json_template_placeholders(p_term_map)

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
                if is_json_data and o_term_map_type == "template":
                    o_term_map = normalize_json_template_placeholders(o_term_map)
                o_term_map, o_term_map_type = normalize_literal_term_map(
                    o_term_map, o_term_map_type, o_term_type
                )

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
                if is_json_data and g_term_map_type == "template":
                    g_term_map = normalize_json_template_placeholders(g_term_map)

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
                    if is_json_data and data_type_term_map_type == "template":
                        data_type_term_map = normalize_json_template_placeholders(data_type_term_map)

                ## Hanlde Language Tag ##
                raw_o_value = element.o
                # Check for datatype
                res = raw_o_value.split("@")
                if len(res) != 2:
                    lang_tag_term_type = ""
                    lang_tag_term_map = ""
                    lang_tag_term_map_type = ""
                else:
                    lang_tag_string = res[1]

                    lang_tag_term_type = "literal"
                    lang_tag_term_map = ""
                    lang_tag_term_map_type = ""

                    lang_tag_term_map = lang_tag_string

                    for key, value in row.items():
                        term_map, term_map_type = get_term_map_type(lang_tag_term_map, key, value, base_uri)
                        if lang_tag_term_map_type == "":
                            lang_tag_term_map = term_map
                            lang_tag_term_map_type = term_map_type
                        elif term_map_type != "constant":
                            lang_tag_term_map = term_map
                            lang_tag_term_map_type = term_map_type
                    
                    # Rename inserted values from pandas headline
                    lang_tag_term_map = lang_tag_term_map.replace("a___", " ")
                    lang_tag_term_map = lang_tag_term_map.replace("ab____", "{")
                    lang_tag_term_map = lang_tag_term_map.replace("abb_____", "}")
                    lang_tag_term_map = lang_tag_term_map.replace("abbb______", "\\")
                    if is_json_data and lang_tag_term_map_type == "template":
                        lang_tag_term_map = normalize_json_template_placeholders(lang_tag_term_map)

                ## Build rml graph ##
                tmp_rml_sub_graph = build_sub_graph(csv_path, is_json_data, json_iterator, s_term_map, s_term_map_type, s_term_type,\
                                                            p_term_map, p_term_map_type, p_term_type, \
                                                            o_term_map, o_term_map_type, o_term_type, \
                                                            g_term_type, g_term_map, g_term_map_type, \
                                                            data_type_term_type, data_type_term_map, data_type_term_map_type,\
                                                            lang_tag_term_type, lang_tag_term_map, lang_tag_term_map_type)
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

        ############# TODO: Funciton prioritize template #########################
        graphs_to_delete = []
        # check for similiar mappings where difference is only subject
        for i in range(len(possible_triples)):
            g = tmp_rml_sub_graphs[i]
            info = extract_information(g)
            possible_triple = possible_triples[i]  

            for j in range(len(possible_triples)):
                g_2 = tmp_rml_sub_graphs[j]
                # Skip the same element
                if g.isomorphic(g_2):
                    continue
                info_2 = extract_information(g_2)
                possible_triple_2 = possible_triples[j] 

                if possible_triple != possible_triple_2:
                    continue

                if not(info[2] == "template" and info_2[2] == "reference"):
                    continue

                # must be equivalent to a reference
                if not(info[1][0] == "{" and info[1][-1] == "}"):
                    continue

                info = list(info)
                del info[2]
                info[1] = info[1].split("{")[1].split("}")[0]

                info_2 = list(info_2)
                del info_2[2]

                if not(info == info_2):
                    continue
                
                graphs_to_delete.append(g_2)                  

        index_to_remove = []
        for g in graphs_to_delete:
            for i in range(len(tmp_rml_sub_graphs)):
                graph = tmp_rml_sub_graphs[i]
                if not g.isomorphic(graph):
                    continue
                index_to_remove.append(i)

        for i in index_to_remove:
            del tmp_rml_sub_graphs[i]
            del possible_triples[i]

        #####################################################

        filtered_graphs = filter_mappings(possible_triples, tmp_rml_sub_graphs)

        # Print the final filtered graphs
        for fg in filtered_graphs:
            rml_sub_graphs.append(fg)

        if DEBUG: 
            print(f"Finished processing: {csv_path}")
    
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

            # tm1 must not be constant in subject
            if info_g[2] == "constant":
                continue

            # tm1 must be constant in object
            if info_g[6] != "constant":
                continue

            # tm2 must be constant in subject
            if info_g2[2] != "constant":
                continue

            # tm2 must not be constant in object
            if info_g2[6] == "constant":
                continue
            
            # predicate and prediacte term type must be the same
            if not (info_g[3] == info_g2[3] and info_g[4] == info_g2[4]):
                continue

            # invar of tm1 subject must be in tm2
            invar_subject_tm1 = get_invar(info_g[1], info_g[2])
            if invar_subject_tm1 not in info_g2[1]:
                continue

            # invar of tm2 object must be in tm1
            invar_object_tm2 = get_invar(info_g2[5], info_g2[6])
            if invar_object_tm2 not in info_g[5]:
                continue

            # If we arrive here, generate the mapping
            new_join_graph = build_sub_graph_join(
                g,
                g2,
                stored_data[info_g[0]],
                stored_data[info_g2[0]],
            )
            join_graphs.append(new_join_graph)
            graphs_to_remove.append(g)
            graphs_to_remove.append(g2)
            
    # Remove graphs that are used in join
    for g in graphs_to_remove:
        rml_sub_graphs = remove_graph(g, rml_sub_graphs)
    
    ### Filter combined result of all input files
    actual_triples = set()
    actual_triples_blank_subject = set()
    actual_triple_records = []
    for rdf in rdf_data:
        variants = build_actual_triple_variants(rdf, base_uri)
        blank_variants = build_actual_triple_variants(rdf, base_uri, blank_subject=True)
        actual_triples.update(variants)
        actual_triples_blank_subject.update(blank_variants)
        actual_triple_records.append(
            (
                build_canonical_actual_triple(rdf),
                variants,
                blank_variants,
            )
        )

    # Prepare data
    possible_triples = []
    new_graphs = []
    for g in rml_sub_graphs:
        info = extract_information(g)
        data = stored_data[info[0]]
        actual_target = actual_triples_blank_subject if info[2] == "none" else actual_triples
        generated_entries = {
            element
            for element in generate_expected_triple(data, info)
            if "None" not in element and element in actual_target
        }
        variant_index = 2 if info[2] == "none" else 1
        res = {
            canonical
            for canonical, variants, blank_variants in actual_triple_records
            if generated_entries & (blank_variants if variant_index == 2 else variants)
        }
        if not res:
            continue
        possible_triples.append(res)
        new_graphs.append(g)
    rml_sub_graphs = new_graphs

    # Prepare joins 
    for g1, g2 in join_graphs:
        info = extract_information_join(g1,g2)
        data = stored_data[info[0]]
        data2 = stored_data[info[9]]
        generated_entries = {
            element
            for element in generate_expected_triple(data, info, data2)
            if "None" not in element and element in actual_triples
        }
        res = {
            canonical
            for canonical, variants, _ in actual_triple_records
            if generated_entries & variants
        }
        if not res:
            continue
        # Add to original data
        g = g1 + g2
        rml_sub_graphs.append(g)
        possible_triples.append(res)

    # Filter data
    filtered_graphs = filter_mappings(possible_triples, rml_sub_graphs)
    rml_sub_graphs = filtered_graphs

    collapse_expressionless_blanknode_subjects(rml_sub_graphs)

    # Check if all triples are expected
    filtered_graphs = []    

    for sub_g in rml_sub_graphs:
        if not is_join_graph(sub_g):
            info = extract_information(sub_g)
            data = stored_data[info[0]]
            res = {
                element for element in generate_expected_triple(data, info) if "None" not in element
            }
        else:
            # Identify graphs pairs
            g1 = None
            g2 = None
            for g1_ref, g2_ref in join_graphs:
                g_ref = g1_ref + g2_ref
                if g_ref.isomorphic(sub_g):
                    g1 = g1_ref
                    g2 = g2_ref
                    break
            if g1 == None and g2 == None:
                print("Error in separating graphs!")
                sys.exit(1)
            
            info = extract_information_join(g1,g2)
            data = stored_data[info[0]]
            data2 = stored_data[info[9]]
            res = {
                element
                for element in generate_expected_triple(data, info, data2)
                if "None" not in element
            }

        cnt_found = 0

        for entry in res:
            target_triples = (
                actual_triples_blank_subject
                if not is_join_graph(sub_g) and info[2] == "none"
                else actual_triples
            )
            if entry in target_triples:
                cnt_found += 1

        if len(res) == cnt_found:
            filtered_graphs.append(sub_g)
    # Final result
    rml_sub_graphs = filtered_graphs

    # Combine graphs with same subject
    rml_sub_graphs_merged = merge_triples_maps(rml_sub_graphs)

    # remove rml:constant only when the same subjectMap also has rml:termType rml:BlankNode
    # Set RML namespace
    RML = Namespace("http://w3id.org/rml/")
    for rml_sub_graph in rml_sub_graphs_merged:
        for sm in list(rml_sub_graph.objects(None, RML.subjectMap)):
            if (sm, RML.termType, RML.BlankNode) in rml_sub_graph:
                rml_sub_graph.remove((sm, RML.constant, None))


    # Print output
    result_graph = Graph()
    result_graph.bind("rml", RML)
    for rml_sub_graph in rml_sub_graphs_merged:
                result_graph += rml_sub_graph


    str_result_graph = result_graph.serialize(format="turtle")
    
    # Add base uri if needed
    if base_uri != "":
        str_result_graph = f"@base <{base_uri}> .\n" + str_result_graph

    # Adjust ////
    str_result_graph = str_result_graph.replace("\\\\","\\")
    str_result_graph = str_result_graph.replace("\\\\\\\\","\\\\")
    str_result_graph = str_result_graph.replace("$['\\{", "$['\\\\{")
    str_result_graph = str_result_graph.replace("\\}']", "\\\\}']")

    # If only base uri is there, an error occured
    if str_result_graph.strip() == f"@base <{base_uri}> .":
        print("An unknown error occured. Could not generate RML mapping document.")
        sys.exit(1)

    return str_result_graph
