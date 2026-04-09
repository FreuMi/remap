import subprocess
import sys
from pathlib import Path
from datetime import datetime

from rdflib import Dataset
from rdflib.compare import graph_diff, to_isomorphic


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remap import generate_rml


CASES_DIR = REPO_ROOT / "test_cases"
FLEXRML = REPO_ROOT / "flexrml"
DEFAULT_BASE_URI = "http://example.com/"

GENERATED_MAPPING = "generated_mapping.ttl"
MATERIALIZED_OUTPUT = "materialized_output.nq"
VALIDATION_REPORT = REPO_ROOT / "validation_report.md"

IGNORED_CASE_FILES = {
    "README.md",
    "mapping.ttl",
    "output.nq",
    GENERATED_MAPPING,
    MATERIALIZED_OUTPUT,
}
IGNORED_SUFFIXES = {".md", ".ttl", ".nq", ".nt", ".trig", ".jsonld"}
RDF_FORMATS = {
    ".nq": "nquads",
    ".nt": "nt",
    ".trig": "trig",
    ".jsonld": "json-ld",
}


def parse_rdf_dataset(path: Path) -> Dataset:
    rdf_format = RDF_FORMATS.get(path.suffix.lower())
    if rdf_format is None:
        raise ValueError(f"Unsupported RDF format for comparison: {path}")

    dataset = Dataset()
    dataset.parse(path, format=rdf_format)
    return dataset


def normalize_rdf_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def serialize_graph_lines(graph) -> set[str]:
    data = graph.serialize(format="nt")
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return {
        line.strip()
        for line in data.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def compare_graphs(file1: Path, file2: Path) -> tuple[bool, set[str], set[str]]:
    try:
        dataset1 = parse_rdf_dataset(file1)
        dataset2 = parse_rdf_dataset(file2)
    except Exception:
        lines1 = normalize_rdf_lines(file1)
        lines2 = normalize_rdf_lines(file2)
        only_generated = lines1 - lines2
        only_expected = lines2 - lines1
        return not only_generated and not only_expected, only_generated, only_expected

    graphs1 = {str(graph.identifier): graph for graph in dataset1.graphs()}
    graphs2 = {str(graph.identifier): graph for graph in dataset2.graphs()}

    only_generated: set[str] = set()
    only_expected: set[str] = set()

    for graph_id in sorted(set(graphs1) - set(graphs2)):
        only_generated.add(f"graph {graph_id}")
    for graph_id in sorted(set(graphs2) - set(graphs1)):
        only_expected.add(f"graph {graph_id}")

    for graph_id in sorted(set(graphs1) & set(graphs2)):
        iso1 = to_isomorphic(graphs1[graph_id])
        iso2 = to_isomorphic(graphs2[graph_id])
        if iso1 == iso2:
            continue

        _, in_first, in_second = graph_diff(iso1, iso2)
        label = "default" if graph_id == "urn:x-rdflib:default" else graph_id
        only_generated.update(
            f"[{label}] {line}" for line in sorted(serialize_graph_lines(in_first))
        )
        only_expected.update(
            f"[{label}] {line}" for line in sorted(serialize_graph_lines(in_second))
        )

    return not only_generated and not only_expected, only_generated, only_expected


def is_error_expected(case_dir: Path) -> bool:
    readme_file = case_dir / "README.md"
    if not readme_file.exists():
        return False
    return "**Error expected?** Yes" in readme_file.read_text(encoding="utf-8")


def get_base_uri(case_dir: Path) -> str:
    readme_file = case_dir / "README.md"
    if not readme_file.exists():
        return DEFAULT_BASE_URI

    for line in readme_file.read_text(encoding="utf-8").splitlines():
        prefix = "**Default Base IRI**: "
        if line.startswith(prefix):
            return line[len(prefix):].strip()

    return DEFAULT_BASE_URI


def find_input_files(case_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.name not in IGNORED_CASE_FILES
        and path.suffix.lower() not in IGNORED_SUFFIXES
    )


def materialize_mapping(mapping_file: Path, output_file: Path, base_uri: str) -> None:
    subprocess.run(
        [
            str(FLEXRML.resolve()),
            "-m",
            str(mapping_file.resolve()),
            "-o",
            str(output_file.resolve()),
            "-b",
            base_uri,
            "--validate-shacl"
        ],
        check=True,
        text=True,
        cwd=mapping_file.parent,
    )


def cleanup(case_dir: Path) -> None:
    (case_dir / GENERATED_MAPPING).unlink(missing_ok=True)
    (case_dir / MATERIALIZED_OUTPUT).unlink(missing_ok=True)


def write_markdown_report(results: list[dict], output_path: Path) -> None:
    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed
    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# Validation Report",
        "",
        f"- Generated: `{timestamp}`",
        f"- Summary: `{passed}/{len(results)} passed, {failed} failed`",
        "",
        "## Results",
        "",
        "| Case | Status | Details |",
        "| --- | --- | --- |",
    ]

    for result in results:
        details = result["details"].replace("\n", "<br>")
        lines.append(
            f"| `{result['case']}` | {'PASSED' if result['passed'] else 'FAILED'} | {details or '-'} |"
        )

        if result["only_generated"]:
            lines.extend(
                [
                    "",
                    f"### {result['case']} Only In Generated Output",
                    "",
                ]
            )
            lines.extend(f"- `{line}`" for line in result["only_generated"])

        if result["only_expected"]:
            lines.extend(
                [
                    "",
                    f"### {result['case']} Only In Expected Output",
                    "",
                ]
            )
            lines.extend(f"- `{line}`" for line in result["only_expected"])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(case_dir: Path) -> dict:
    rdf_file = case_dir / "output.nq"
    generated_mapping = case_dir / GENERATED_MAPPING
    materialized_output = case_dir / MATERIALIZED_OUTPUT
    expected_error = is_error_expected(case_dir)
    base_uri = get_base_uri(case_dir)

    cleanup(case_dir)

    try:
        input_files = find_input_files(case_dir)
        if not input_files:
            raise FileNotFoundError(f"No input file found in {case_dir}")

        rdf_data = rdf_file.read_text(encoding="utf-8")
        input_data = [path.read_text(encoding="utf-8") for path in input_files]

        mapping_ttl = generate_rml(
            rdf_data,
            input_data,
            base_uri=base_uri,
            csv_paths=[path.name for path in input_files],
        )
        generated_mapping.write_text(mapping_ttl, encoding="utf-8")
        materialize_mapping(generated_mapping, materialized_output, base_uri)
    except Exception as exc:
        cleanup(case_dir)
        if expected_error:
            print(f"PASSED {case_dir.name} (expected failure: {exc})")
            return {
                "case": case_dir.name,
                "passed": True,
                "details": f"Expected failure: {exc}",
                "only_generated": [],
                "only_expected": [],
            }
        print(f"ERROR {case_dir.name}: {exc}")
        return {
            "case": case_dir.name,
            "passed": False,
            "details": f"Error: {exc}",
            "only_generated": [],
            "only_expected": [],
        }

    if expected_error:
        cleanup(case_dir)
        print(f"FAILED {case_dir.name} (expected failure, but succeeded)")
        return {
            "case": case_dir.name,
            "passed": False,
            "details": "Expected failure, but succeeded",
            "only_generated": [],
            "only_expected": [],
        }

    try:
        is_equal, only_generated, only_expected = compare_graphs(
            materialized_output,
            rdf_file,
        )
    except Exception as exc:
        cleanup(case_dir)
        print(f"ERROR {case_dir.name}: {exc}")
        return {
            "case": case_dir.name,
            "passed": False,
            "details": f"Error: {exc}",
            "only_generated": [],
            "only_expected": [],
        }
    finally:
        cleanup(case_dir)

    if is_equal:
        print(f"PASSED {case_dir.name}")
        return {
            "case": case_dir.name,
            "passed": True,
            "details": "",
            "only_generated": [],
            "only_expected": [],
        }

    print(f"FAILED {case_dir.name}")
    if only_generated:
        print("  Only in generated output:")
        for line in sorted(only_generated):
            print(f"    {line}")
    if only_expected:
        print("  Only in expected output:")
        for line in sorted(only_expected):
            print(f"    {line}")
    return {
        "case": case_dir.name,
        "passed": False,
        "details": "Output mismatch",
        "only_generated": sorted(only_generated),
        "only_expected": sorted(only_expected),
    }


def main() -> int:
    if not CASES_DIR.exists():
        print(f"Cases directory not found: {CASES_DIR}", file=sys.stderr)
        return 1
    if not FLEXRML.exists():
        print(f"FlexRML executable not found: {FLEXRML}", file=sys.stderr)
        return 1

    case_dirs = sorted(path for path in CASES_DIR.iterdir() if path.is_dir())
    if not case_dirs:
        print("No test cases found.", file=sys.stderr)
        return 1

    results: list[dict] = []

    for case_dir in case_dirs:
        results.append(run_case(case_dir))

    write_markdown_report(results, VALIDATION_REPORT)

    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed

    print(f"\nSummary: {passed}/{len(case_dirs)} passed, {failed} failed")
    print(f"Markdown report: {VALIDATION_REPORT}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
