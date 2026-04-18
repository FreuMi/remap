import pandas as pd
import json
from collections.abc import Mapping
from pathlib import Path
import xml.etree.ElementTree as ET


def is_simple_json_key(key: str) -> bool:
    return key.replace("_", "a").isalnum() and key[:1].isalpha() or key[:1] == "_"


def append_json_path_segment(parent_key: str, key: str) -> str:
    escaped_key = key.replace("\\", "\\\\").replace("'", "\\'")
    if parent_key == "":
        if is_simple_json_key(key):
            return key
        return f"['{escaped_key}']"
    if is_simple_json_key(key):
        return f"{parent_key}.{key}"
    return f"{parent_key}['{escaped_key}']"


def flatten_dict(d, parent_key=""):
    items = {}
    for k, v in d.items():
        new_key = append_json_path_segment(parent_key, k)
        if isinstance(v, Mapping):
            items.update(flatten_dict(v, new_key))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v, ensure_ascii=False)
        else:
            items[new_key] = v
    return items


def extract_json_records(raw_json_text: str) -> list[dict]:
    data = json.loads(raw_json_text)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        list_entries = [
            value
            for value in data.values()
            if isinstance(value, list) and all(isinstance(entry, Mapping) for entry in value)
        ]
        if len(list_entries) == 1:
            return list_entries[0]
        return [data]

    return [{"value": data}]


def flatten_xml_element(element: ET.Element, parent_key=""):
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

    grouped_children = {}
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


def extract_xml_records(raw_xml_text: str) -> list[dict]:
    root = ET.fromstring(raw_xml_text.lstrip())
    children = list(root)

    if not children:
        return []

    child_tag_counts = {}
    for child in children:
        child_tag_counts[child.tag] = child_tag_counts.get(child.tag, 0) + 1

    if len(child_tag_counts) == 1:
        return [flatten_xml_element(child) for child in children]

    repeated_tags = [tag for tag, count in child_tag_counts.items() if count > 1]
    if len(repeated_tags) == 1:
        record_tag = repeated_tags[0]
        return [flatten_xml_element(child) for child in children if child.tag == record_tag]

    return [flatten_xml_element(root)]


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col.replace(" ", "a___")
        .replace("{", "ab____")
        .replace("}", "abb_____")
        .replace("\\", "abbb______")
        for col in df.columns
    ]
    return df


def load_dataframe(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        records = extract_json_records(file_path.read_text(encoding="utf-8"))
        flattened_records = []
        for record in records:
            flat_record = flatten_dict(record) if isinstance(record, Mapping) else {"value": record}
            flattened_records.append(
                {
                    f"${key if key.startswith('[') else '.' + key}": "None" if value is None else str(value)
                    for key, value in flat_record.items()
                }
            )
        if not flattened_records:
            return pd.DataFrame()
        return sanitize_columns(pd.DataFrame(flattened_records, dtype=str))
    if file_path.suffix.lower() == ".xml":
        records = extract_xml_records(file_path.read_text(encoding="utf-8"))
        if not records:
            return pd.DataFrame()
        return sanitize_columns(pd.DataFrame(records, dtype=str))

    return sanitize_columns(pd.read_csv(file_path, dtype=str))

### Function to compute overlap ###
def compute_overlap_ratio(dfA, colA, dfB, colB):
    distinctA = set(dfA[colA].dropna().unique())
    distinctB = set(dfB[colB].dropna().unique())
    
    # Intersection of distinct values
    common_values = distinctA.intersection(distinctB)
    intersection_size = len(common_values)
    
    # Overlap ratio = intersection / min(sizeA, sizeB)
    sizeA = len(distinctA)
    sizeB = len(distinctB)
    if min(sizeA, sizeB) == 0:
        overlap_ratio = 0
    else:
        overlap_ratio = intersection_size / min(sizeA, sizeB)
    
    return overlap_ratio

def identify_join(dfA_source, dfB_source):
    ### Load data ###
    dfA = dfA_source.copy() if isinstance(dfA_source, pd.DataFrame) else load_dataframe(dfA_source)
    dfB = dfB_source.copy() if isinstance(dfB_source, pd.DataFrame) else load_dataframe(dfB_source)
    
    ### Generate candidate column pairs and score ###
    results = []
    for colA in dfA.columns:
        for colB in dfB.columns:
            score = compute_overlap_ratio(dfA, colA, dfB, colB)
            
            results.append({
                'colA': colA,
                'colB': colB,
                'score': score
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    ### Identify top candidate for joining ###
    best_score = results_df['score'].max()
    if best_score <= 0:
        return "", ""
    best_pairs = results_df[results_df['score'] == best_score]

    # Print all best pairs
    #print("\nBest candidate pairs")
    #print(best_pairs[['colA', 'colB', 'score']].to_string(index=False))

    max_val = 0
    max_child = ""
    max_parent = ""
    for row in best_pairs.itertuples(index=False):
        row = row._asdict()
        child = row["colA"]
        parent = row["colB"]

        ### Perform join on the best pairs and find max ###
        df_joined = dfA.merge(dfB, left_on=child, right_on=parent, how='inner')

        if len(df_joined) >= max_val:
            max_child = child
            max_parent = parent
            max_val = len(df_joined)

    if max_val <= 0:
        return "", ""

    return max_child, max_parent
