import sys
import argparse

from .remap_core import generate_rml


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

    str_result_graph = generate_rml(file_path_rdf, file_path_csv, base_uri)

    # Write to file.
    with open(output_file, "w") as file:
        file.write(str_result_graph)

    print("Finished. Generated mapping stored in:", output_file)

if __name__ == "__main__":  
    main()