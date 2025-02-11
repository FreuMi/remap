# 🚀 RML Mapping Reverse Engineering

This tool **reverse engineers an RML mapping document** for a given CSV source data and RDF output graph. The generated mapping document is functionally equivalent to the original **unknown RML mapping**, ensuring it produces the same RDF output but may use different RML constructs.

**Tested on Python 3.12.3 (Ubuntu 24.04)**.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/RML-Mapping-Reverse-Engineering.git
   cd RML-Mapping-Reverse-Engineering
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

## Usage
Run the tool with:
```bash
   python3 qre.py --csv [LIST OF CSV INPUT FILES] --rdf RDF_OUTPUT_FILE
```

## Example Usage
If you have two input CSV files (`sport.csv` and `student.csv`) and an RDF output file (`output.nq`), execute:
```bash
   python3 qre.py --csv sport.csv student.csv --rdf output.nq
```
This will generate an RML mapping document that, when executed, produces the same RDF output graph.

##  License
This project is licensed under the GNU Affero General Public License version 3 (AGPLv3). The full text of the license can be found in the `LICENSE` file in this repository.