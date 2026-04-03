# ReMap: RML Mapping Reverse Engineering

ReMap reverse engineers an RML mapping document from CSV source data and an RDF output graph. The generated mapping is functionally equivalent to the original unknown RML mapping, even if it uses different RML constructs internally.

The project now ships as an installable Python package with:

- a library API under `src/remap`
- a CLI entrypoint exposed as `remap`

Tested on Python 3.12.3 on Ubuntu 24.04.

## Installation

Clone the repository and install it as a package:

```bash
git clone git@github.com:FreuMi/remap.git
cd remap
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

For editable local development:

```bash
pip install -e .
```

If you only want the dependencies without installing the package itself:

```bash
pip install -r requirements.txt
```

## CLI Usage

After installation, run:

```bash
remap --csv [LIST OF CSV INPUT FILES] --rdf RDF_OUTPUT_FILE
```

Example:

```bash
remap --csv sport.csv student.csv --rdf output.nq
```

This generates `generated_mapping.ttl` in the current working directory.

The CLI currently assumes the base URI `http://example.com/base/`.

## Library Usage

ReMap can also be used as a Python library.

Generate a mapping from file paths:

```python
from remap import generate_rml_from_file

mapping_ttl = generate_rml_from_file(
    "output.nq",
    ["sport.csv", "student.csv"],
)

print(mapping_ttl)
```

Generate a mapping from in-memory RDF and CSV content:

```python
from remap import generate_rml

with open("output.nq", "r", encoding="utf-8") as rdf_file:
    rdf_data = rdf_file.read()

csv_data = []
for path in ["sport.csv", "student.csv"]:
    with open(path, "r", encoding="utf-8") as csv_file:
        csv_data.append(csv_file.read())

mapping_ttl = generate_rml(
    rdf_data,
    csv_data,
    base_uri="http://example.com/base/",
    csv_paths=["sport.csv", "student.csv"],
)
```

## Build

If you still want to compile the CLI into a standalone binary with [Nuitka](https://nuitka.net/), use the package entrypoint instead of the top-level `remap.py` script.

Install build requirements:

```bash
pip install nuitka
pip install .
sudo apt install python3-dev patchelf build-essential
```

Then compile:

```bash
nuitka --standalone --onefile --include-package=remap -m remap.remap_cli
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{mappingByExample,
  title={Mapping by Example: Towards an RML Mapping Reverse Engineering Pipeline},
  author={Freund, Michael and Dorsch, Rene and Schmid, Sebastian and Harth, Andreas},
  booktitle={Sixth International Workshop on Knowledge Graph Construction}
}
```

## License

This project is licensed under the GNU Affero General Public License version 3 (AGPLv3). See `LICENSE`.
