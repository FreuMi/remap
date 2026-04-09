import rdflib
from dataclasses import dataclass
import re

# Class to store quad data
@dataclass
class Quad:
    s: str
    p: str
    o: str
    g: str

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

def parse_rdf_as_nt(raw_data: str):
    raw_data = "\n".join(
        line for line in raw_data.splitlines() if not line.lstrip().startswith("#")
    )

    # Preserve original blank node labels when the input is already line-based RDF.
    for format in ["nquads", "nt"]:
        try:
            temp_graph = rdflib.Dataset()
            temp_graph.parse(data=raw_data, format=format)
            return raw_data
        except Exception:
            continue

    ## Parse RDF
    supported_formats = ["nquads", "trig", "json-ld", "turtle", "xml", "nt", "n3"]
    parsed_successfully = False
    for format in supported_formats:
        try:
            temp_graph = rdflib.Dataset()
            temp_graph.parse(data=raw_data, format=format)

            # If parsing succeeds without an exception
            rdf_graph = temp_graph
            parsed_successfully = True
            #print(f"Successfully parsed input data as {format}.")
            break
        except Exception as e:
            #print(f"Attempt to parse as '{format}' failed: {e}")
            continue

    if not parsed_successfully:
        raise TypeError(f"Input data could not be parsed in any supported RDF format: {supported_formats}")

    ntriple_data = rdf_graph.serialize(format="nquads")

    return ntriple_data

# Function to parse rdf data to nquads
def parse(ntriple_data: str) -> list[Quad]:
    rdf_data = []
    # Iterate over data and separate
    for line in ntriple_data.split("\n"):
        line = line.strip()
        if line == "":
            continue

        line_parts = tokenizer(line)      

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
