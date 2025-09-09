# ReMap: RML Mapping Reverse Engineering

This tool **reverse engineers an RML mapping document** for a given CSV source data and RDF output graph. The generated mapping document is functionally equivalent to the original **unknown RML mapping**, ensuring it produces the same RDF output but may use different RML constructs.

**Tested on Python 3.12.3 (Ubuntu 24.04)**.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:FreuMi/remap.git
   cd mapping_generator
    ```
2. **(Optional) Create a virtual environment**:
    ```bash
   python3 -m venv venv
    source venv/bin/activate 
    ```
3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt
    ```

## Compilation
In order to compile the project we use [Nuitka](https://nuitka.net/).

1. **Install required packages**
```bash
   pip install nuitka
   pip install -r requirements.txt
   sudo apt install python3-dev patchelf build-essential
```
2. **Start compilation**
```bash
nuitka --standalone --onefile --include-package=rdflib remap.py
```

## Usage
Run the tool with python:
```bash
   python3 remap.py --csv [LIST OF CSV INPUT FILES] --rdf RDF_OUTPUT_FILE
```

Run the compiled tool:
```bash
   ./remap.bin --csv [LIST OF CSV INPUT FILES] --rdf RDF_OUTPUT_FILE
```

## Example Usage
If you have two input CSV files (`sport.csv` and `student.csv`) and an RDF output file (`output.nq`), execute:
```bash
   python3 remap.py --csv sport.csv student.csv --rdf output.nq
```
This will generate an RML mapping document that, when executed, produces the same RDF output graph.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{mappingByExample,
  title={Mapping by Example: Towards an RML Mapping Reverse Engineering Pipeline},
  author={Freund, Michael and Dorsch, Rene and Schmid, Sebastian and Harth, Andreas},
  booktitle={Sixth International Workshop on Knowledge Graph Construction @ ESWC2025}
}
```

##  License
This project is licensed under the GNU Affero General Public License version 3 (AGPLv3). The full text of the license can be found in the `LICENSE` file in this repository.
