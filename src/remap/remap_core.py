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
import xml.etree.ElementTree as ET
from typing import Optional

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


def build_actual_triple_variants(
    rdf,
    base_uri: str,
    blank_subject: bool = False,
    blank_node_templates = None,
) -> set[str]:
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
    if blank_node_templates is not None and isBlanknode(rdf.s) and rdf.s in blank_node_templates:
        subject_variants.append(blank_node_templates[rdf.s][0])

    predicate_variants = rdf_term_variants(rdf.p, p, base_uri)
    object_variants = rdf_term_variants(rdf.o, o, base_uri)
    if blank_node_templates is not None and isBlanknode(rdf.o) and rdf.o in blank_node_templates:
        object_variants.append(blank_node_templates[rdf.o][0])
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
                elif language_type == "reference":
                    key = sanitize_lookup_key(language_map)
                    language = row[key]
                elif language_type == "template":
                    matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', language_map)
                    language = language_map
                    for match in matches:
                        match_org = match
                        match = sanitize_lookup_key(match)
                        language = language.replace("{"+match_org+"}", row[match])

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
            elif language_type == "reference":
                key = sanitize_lookup_key(language_map)
                language = row[key]
            elif language_type == "template":
                matches = re.findall(r'(?<!\\)\{(.*?)(?<!\\)\}', language_map)
                language = language_map
                for match in matches:
                    match_org = match
                    match = sanitize_lookup_key(match)
                    language = language.replace("{"+match_org+"}", row[match])

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


def term_map_invariants(term_map: str, term_map_type: str) -> set[str]:
    invariants = {get_invar(term_map, term_map_type)}
    if term_map_type == "constant":
        decoded = unquote(term_map)
        if decoded != term_map:
            invariants.add(decoded)
    return {value for value in invariants if value != ""}


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


def unescape_column_name(name: str) -> str:
    name = name.replace("a___", " ")
    name = name.replace("ab____", "{")
    name = name.replace("abb_____", "}")
    name = name.replace("abbb______", "\\")
    return name


def blank_node_token(value: str) -> str:
    value = clean_entry(value)
    if "#" in value:
        value = value.rsplit("#", 1)[1]
    elif "/" in value:
        value = value.rstrip("/").rsplit("/", 1)[1]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value


def find_value_reference(value: str, row: dict[str, str], base_uri: str) -> str:
    value = clean_entry(value)
    value = remove_base_uri(value, base_uri)

    for key, source_value in row.items():
        if source_value == "None":
            continue
        if str(source_value) == value:
            return f"{{{key}}}"

    for key, source_value in row.items():
        if source_value == "None":
            continue
        if str(source_value) != "" and str(source_value) in value:
            return f"{{{key}}}"

    if value.startswith("http://") or value.startswith("https://"):
        for key in row:
            key_token = blank_node_token(unescape_column_name(key))
            if key_token != "" and key_token in value:
                return key_token

    return ""


def infer_blank_node_templates(
    rdf_data,
    row: dict[str, str],
    base_uri: str,
) -> dict[str, tuple[str, str]]:
    blank_nodes = {
        term
        for element in rdf_data
        for term in (element.s, element.o)
        if isBlanknode(term)
    }
    templates: dict[str, tuple[str, str]] = {}
    resolving: set[str] = set()

    def template_for(blank_node: str) -> str:
        if blank_node in templates:
            return templates[blank_node][0]
        if blank_node in resolving:
            return blank_node_token(blank_node)

        resolving.add(blank_node)
        parts = []
        outgoing = sorted(
            (element for element in rdf_data if element.s == blank_node),
            key=lambda element: (element.p, element.o),
        )
        for element in outgoing:
            if isBlanknode(element.o):
                child_template = template_for(element.o)
                if child_template != "":
                    parts.append(child_template)
                continue

            ref = find_value_reference(element.o, row, base_uri)
            if ref != "":
                parts.append(ref)
                continue

            token = blank_node_token(element.o)
            if token != "":
                parts.append(token)

        resolving.remove(blank_node)

        deduped_parts = []
        for part in parts:
            if part not in deduped_parts:
                deduped_parts.append(part)

        if not deduped_parts:
            deduped_parts.append("node")

        template = "blank-" + "-".join(deduped_parts)
        templates[blank_node] = (template, "template")
        return template

    for blank_node in sorted(blank_nodes):
        template_for(blank_node)

    return templates


def materialize_template(term_map: str, row: dict[str, str]) -> str:
    result = term_map
    for key, value in row.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def materialize_blank_node_templates(
    templates: dict[str, tuple[str, str]],
    row: dict[str, str],
) -> dict[str, tuple[str, str]]:
    return {
        blank_node: (materialize_template(term_map, row), term_map_type)
        for blank_node, (term_map, term_map_type) in templates.items()
    }


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


def collect_json_scalar_values(value) -> set[str]:
    if isinstance(value, Mapping):
        values = set()
        for child in value.values():
            values.update(collect_json_scalar_values(child))
        return values
    if isinstance(value, list):
        values = set()
        for child in value:
            values.update(collect_json_scalar_values(child))
        return values
    if value is None:
        return set()
    return {str(value)}


def rdf_sample_values(rdf_data) -> set[str]:
    values = set()
    for element in rdf_data:
        for term in (element.s, element.p, element.o, element.g):
            if term == "":
                continue
            cleaned = clean_entry(term)
            if "^^" in cleaned:
                cleaned = cleaned.split("^^", 1)[0]
            elif "@" in cleaned and not cleaned.startswith("http://") and not cleaned.startswith("https://"):
                cleaned = cleaned.split("@", 1)[0]
            values.add(cleaned)
    return values


def json_repeated_record_candidate(data, rdf_data=None) -> tuple[str, list] | None:
    list_entries = [
        (key, value)
        for key, value in data.items()
        if isinstance(value, list)
    ]
    if len(list_entries) != 1 or not all(
        isinstance(entry, Mapping) for entry in list_entries[0][1]
    ):
        return None

    if rdf_data is None:
        return list_entries[0]

    key, value = list_entries[0]
    sample_values = rdf_sample_values(rdf_data)
    repeated_values = collect_json_scalar_values(value)
    root_values = collect_json_scalar_values(
        {
            root_key: child
            for root_key, child in data.items()
            if root_key != key and not isinstance(child, list)
        }
    )

    if sample_values & repeated_values:
        return key, value
    if sample_values & root_values:
        return None
    return key, value


def extract_json_records(raw_json_text: str, rdf_data=None) -> tuple[list[dict], str]:
    data = json.loads(raw_json_text)

    if isinstance(data, list):
        return data, "$[*]"

    if isinstance(data, dict):
        repeated_record = json_repeated_record_candidate(data, rdf_data)
        if repeated_record is not None:
            key, value = repeated_record
            return value, f"$.{key}[*]"
        return [data], "$"

    return [{"value": data}], "$"


def json_to_dataframe(raw_json_text: str, rdf_data=None) -> tuple[pd.DataFrame, str]:
    records, iterator = extract_json_records(raw_json_text, rdf_data)
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


def is_valid_xml(s: str) -> bool:
    try:
        ET.fromstring(s.lstrip())
        return True
    except ET.ParseError:
        return False


def singularize_tag(tag: str) -> str:
    if tag.endswith("ies") and len(tag) > 3:
        return tag[:-3] + "y"
    if tag.endswith("ses") and len(tag) > 3:
        return tag[:-2]
    if tag.endswith("s") and not tag.endswith("ss") and len(tag) > 1:
        return tag[:-1]
    return tag


def flatten_xml_element(element: ET.Element, parent_key: str = "") -> dict[str, str]:
    items = {}

    for attr_name, attr_value in element.attrib.items():
        key = f"{parent_key}/@{attr_name}" if parent_key else f"@{attr_name}"
        items[key] = attr_value

    children = list(element)
    text = (element.text or "").strip()
    if not children:
        key = parent_key or element.tag
        items[key] = text
        return items

    grouped_children: dict[str, list[ET.Element]] = {}
    for child in children:
        grouped_children.setdefault(child.tag, []).append(child)

    if text:
        key = f"{parent_key}/#text" if parent_key else "#text"
        items[key] = text

    for child_tag, same_tag_children in grouped_children.items():
        child_key = f"{parent_key}/{child_tag}" if parent_key else child_tag
        if len(same_tag_children) == 1:
            items.update(flatten_xml_element(same_tag_children[0], child_key))
            continue

        values = []
        complex_entries = []
        for child in same_tag_children:
            if len(list(child)) == 0 and not child.attrib:
                values.append((child.text or "").strip())
            else:
                complex_entries.append(flatten_xml_element(child, child_key))
        if values:
            items[child_key] = json.dumps(values, ensure_ascii=False)
        for idx, entry in enumerate(complex_entries):
            for entry_key, entry_value in entry.items():
                items[f"{entry_key}[{idx}]"] = entry_value

    return items


def extract_xml_records(raw_xml_text: str) -> tuple[list[dict], str]:
    root = ET.fromstring(raw_xml_text.lstrip())
    candidate = find_repeated_xml_record_candidate(root)
    if candidate is not None:
        record_path, record_elements = candidate
        context_values = collect_xml_scalar_context(root)
        records = []
        for record_element in record_elements:
            record = flatten_xml_element(record_element)
            for value_path, value in context_values:
                if is_descendant_xml_path(value_path, record_path):
                    continue
                reference = xml_relative_reference(record_path, value_path)
                if reference and reference not in record:
                    record[reference] = value
            records.append(record)
        return records, "/" + "/".join(record_path)

    children = list(root)
    if not children:
        guessed_child = singularize_tag(root.tag)
        return [], f"/{root.tag}/{guessed_child}"

    child_tag_counts: dict[str, int] = {}
    for child in children:
        child_tag_counts[child.tag] = child_tag_counts.get(child.tag, 0) + 1

    if len(child_tag_counts) == 1:
        record_tag = next(iter(child_tag_counts))
        records = [flatten_xml_element(child) for child in children]
        return records, f"/{root.tag}/{record_tag}"

    repeated_tags = [tag for tag, count in child_tag_counts.items() if count > 1]
    if len(repeated_tags) == 1:
        record_tag = repeated_tags[0]
        records = [flatten_xml_element(child) for child in children if child.tag == record_tag]
        return records, f"/{root.tag}/{record_tag}"

    return [flatten_xml_element(root)], f"/{root.tag}"


def find_repeated_xml_record_candidate(
    root: ET.Element,
) -> Optional[tuple[list[str], list[ET.Element]]]:
    candidates = []

    def visit(element: ET.Element, path: list[str]) -> None:
        grouped_children: dict[str, list[ET.Element]] = {}
        for child in list(element):
            grouped_children.setdefault(child.tag, []).append(child)

        for child_tag, same_tag_children in grouped_children.items():
            child_path = [*path, child_tag]
            complex_children = [
                child for child in same_tag_children if list(child) or child.attrib
            ]
            if len(same_tag_children) > 1 and len(complex_children) == len(same_tag_children):
                candidates.append((child_path, same_tag_children))
            for child in same_tag_children:
                visit(child, child_path)

    visit(root, [root.tag])
    if not candidates:
        return None

    candidates.sort(key=lambda entry: (len(entry[0]), len(entry[1])), reverse=True)
    return candidates[0]


def collect_xml_scalar_context(root: ET.Element) -> list[tuple[list[str], str]]:
    values = []

    def visit(element: ET.Element, path: list[str]) -> None:
        for attr_name, attr_value in element.attrib.items():
            values.append(([*path, f"@{attr_name}"], attr_value))

        children = list(element)
        text = (element.text or "").strip()
        if text:
            if children:
                values.append(([*path, "#text"], text))
            else:
                values.append((path, text))

        for child in children:
            visit(child, [*path, child.tag])

    visit(root, [root.tag])
    return values


def is_descendant_xml_path(path: list[str], ancestor_path: list[str]) -> bool:
    return len(path) > len(ancestor_path) and path[: len(ancestor_path)] == ancestor_path


def xml_relative_reference(record_path: list[str], value_path: list[str]) -> str:
    common_len = 0
    for record_part, value_part in zip(record_path, value_path):
        if record_part != value_part:
            break
        common_len += 1

    up_segments = [".."] * (len(record_path) - common_len)
    down_segments = value_path[common_len:]
    return "/".join([*up_segments, *down_segments])


def xml_to_dataframe(raw_xml_text: str) -> tuple[pd.DataFrame, str]:
    records, iterator = extract_xml_records(raw_xml_text)
    if not records:
        return pd.DataFrame(), iterator
    flattened_records = []
    for record in records:
        flattened_records.append(
            {
                key: "None" if value is None else str(value)
                for key, value in record.items()
            }
        )
    return pd.DataFrame(flattened_records, dtype=str), iterator


def detect_source_format(raw_text: str, source_path: str = "") -> str:
    suffix = source_path.lower().rsplit(".", 1)[-1] if "." in source_path else ""
    content_is_json = is_valid_json(raw_text)
    content_is_xml = is_valid_xml(raw_text)
    if suffix == "json" and content_is_json:
        return "json"
    if suffix == "xml" and content_is_xml:
        return "xml"
    if content_is_json:
        return "json"
    if content_is_xml:
        return "xml"
    if suffix == "json":
        return "json"
    if suffix == "xml":
        return "xml"
    return "csv"


def build_empty_mapping(
    source_path: str,
    source_format: str,
    iterator: str,
    base_uri: str,
) -> str:
    if source_format == "json":
        subject_map = "$.missing_subject"
        predicate_map = "$.missing_predicate"
        object_map = "$.missing_object"
    elif source_format == "xml":
        subject_map = "missing_subject"
        predicate_map = "missing_predicate"
        object_map = "missing_object"
    else:
        subject_map = "missing_subject"
        predicate_map = "missing_predicate"
        object_map = "missing_object"

    empty_graph = build_sub_graph(
        source_path,
        source_format,
        iterator,
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
    stored_source_formats = {}

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
        source_format = detect_source_format(csv_text, source_path)
        if source_format == "json":
            _, iterator = json_to_dataframe(csv_text, rdf_data)
        elif source_format == "xml":
            _, iterator = xml_to_dataframe(csv_text)
        else:
            iterator = "$"
        return build_empty_mapping(source_path, source_format, iterator, base_uri)

    for i in range(len(csv_data)):
        csv_text = csv_data[i]
        source_format = detect_source_format(csv_text, csv_paths[i] if i < len(csv_paths) else "")
        iterator = "$"

        if source_format == "json":
            data, iterator = json_to_dataframe(csv_text, rdf_data)
        elif source_format == "xml":
            data, iterator = xml_to_dataframe(csv_text)
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
            blank_node_templates = infer_blank_node_templates(rdf_data, row, base_uri)
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

                if isBlanknode(element.s) and element.s in blank_node_templates:
                    s_term_map, s_term_map_type = blank_node_templates[element.s]
                else:
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
                    if source_format == "json" and s_term_map_type == "template":
                        s_term_map = normalize_json_template_placeholders(s_term_map)
                    if source_format == "json":
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
                if source_format == "json" and p_term_map_type == "template":
                    p_term_map = normalize_json_template_placeholders(p_term_map)

                ## Handle object ##
                o_term_type = get_term_type(o)
                o_term_map = ""
                o_term_map_type = ""

                if isBlanknode(element.o) and element.o in blank_node_templates:
                    o_term_map, o_term_map_type = blank_node_templates[element.o]
                else:
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
                    if source_format == "json" and o_term_map_type == "template":
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
                if source_format == "json" and g_term_map_type == "template":
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
                    if source_format == "json" and data_type_term_map_type == "template":
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
                    if source_format == "json" and lang_tag_term_map_type == "template":
                        lang_tag_term_map = normalize_json_template_placeholders(lang_tag_term_map)

                ## Build rml graph ##
                tmp_rml_sub_graph = build_sub_graph(csv_path, source_format, iterator, s_term_map, s_term_map_type, s_term_type,\
                                                            p_term_map, p_term_map_type, p_term_type, \
                                                            o_term_map, o_term_map_type, o_term_type, \
                                                            g_term_type, g_term_map, g_term_map_type, \
                                                            data_type_term_type, data_type_term_map, data_type_term_map_type,\
                                                            lang_tag_term_type, lang_tag_term_map, lang_tag_term_map_type)
                if not isDuplicateGraph(tmp_rml_sub_graph, tmp_rml_sub_graphs):
                    tmp_rml_sub_graphs.append(tmp_rml_sub_graph)
            
        # Store data
        stored_data[csv_path] = data
        stored_source_formats[csv_path] = source_format

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
        object_term_type_g = get_term_type_of_graph(g, "o")
        # Compare to all other files
        for g2 in rml_sub_graphs:
            info_g2 = extract_information(g2)
            subject_term_type_g2 = get_term_type_of_graph(g2, "s")
            
            # Do not compare with itself
            if info_g2 == info_g:
                continue

            # child TM must have a generated subject
            if info_g[2] == "constant":
                continue

            # child object must be an IRI-like value
            if object_term_type_g not in {"iri", "unsafeiri"}:
                continue

            # parent TM must have a generated subject
            if info_g2[2] == "constant":
                continue

            if subject_term_type_g2 not in {"iri", "unsafeiri"}:
                continue

            # The parent subject pattern must match the child object pattern.
            child_object_invariants = term_map_invariants(info_g[5], info_g[6])
            parent_subject_invariants = term_map_invariants(info_g2[1], info_g2[2])
            if not child_object_invariants or not parent_subject_invariants:
                continue

            if not any(
                child_inv in parent_inv or parent_inv in child_inv
                for child_inv in child_object_invariants
                for parent_inv in parent_subject_invariants
            ):
                continue

            # If we arrive here, generate the mapping
            new_join_graph = build_sub_graph_join(
                g,
                g2,
                stored_data[info_g[0]],
                stored_data[info_g2[0]],
            )
            if len(new_join_graph[0]) == 0 or len(new_join_graph[1]) == 0:
                continue
            join_graphs.append(new_join_graph)
            graphs_to_remove.append(g)
            
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
        for data in stored_data.values():
            for _, row in data.iterrows():
                row = row.to_dict()
                row = dict(sorted(row.items(), key=lambda item: len(item[1]), reverse=True))
                inferred_blank_nodes = materialize_blank_node_templates(
                    infer_blank_node_templates(rdf_data, row, base_uri),
                    row,
                )
                variants.update(
                    build_actual_triple_variants(
                        rdf,
                        base_uri,
                        blank_node_templates=inferred_blank_nodes,
                    )
                )
                blank_variants.update(
                    build_actual_triple_variants(
                        rdf,
                        base_uri,
                        blank_subject=True,
                        blank_node_templates=inferred_blank_nodes,
                    )
                )
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

        if len(res) == cnt_found or (
            stored_source_formats.get(info[0]) == "xml" and cnt_found > 0
        ):
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
