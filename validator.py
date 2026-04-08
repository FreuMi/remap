import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remap import generate_rml


IGNORED_CASE_FILES = {
    "README.md",
    "mapping.ttl",
    "output.nq",
    "generated_mapping.ttl",
    "materialized_output.nq",
    "expected_materialized_output.nq",
    "official_materialized.nq",
}


def preprocess_line(line: str) -> str:
    # BURP emits fresh blank node ids on every run, so normalize them away.
    return " ".join(
        "BLANK_NODE" if part.startswith("_:") else part for part in line.split()
    )


def compare_files(file1: Path, file2: Path) -> tuple[bool, set[str], set[str]]:
    with file1.open("r", encoding="utf-8") as f1:
        lines1 = {
            preprocess_line(line.strip())
            for line in f1
            if line.strip() and not line.lstrip().startswith("#")
        }

    with file2.open("r", encoding="utf-8") as f2:
        lines2 = {
            preprocess_line(line.strip())
            for line in f2
            if line.strip() and not line.lstrip().startswith("#")
        }

    return lines1 == lines2, lines1 - lines2, lines2 - lines1


def find_input_files(case_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file() and path.name not in IGNORED_CASE_FILES
    )


def materialize_mapping(
    mapping_file: Path, output_file: Path, rmlmapper_jar: Path, base_uri: str
) -> None:
    command = [
        "java",
        "-jar",
        str(rmlmapper_jar.resolve()),
        "-m",
        mapping_file.name,
        "-o",
        output_file.name,
        "-b",
        base_uri,
        "-s",
        "nquads",
    ]
    subprocess.run(command, check=True, text=True, cwd=mapping_file.parent)


def run_case(
    case_dir: Path,
    rmlmapper_jar: Path,
    keep_artifacts: bool,
    base_uri: str,
    compare_mode: str,
) -> bool:
    rdf_file = case_dir / "output.nq"
    reference_mapping = case_dir / "mapping.ttl"
    generated_mapping = case_dir / "generated_mapping.ttl"
    materialized_output = case_dir / "materialized_output.nq"
    expected_materialized_output = case_dir / "expected_materialized_output.nq"

    if not keep_artifacts:
        generated_mapping.unlink(missing_ok=True)
        materialized_output.unlink(missing_ok=True)
        expected_materialized_output.unlink(missing_ok=True)

    input_files = find_input_files(case_dir)

    if not rdf_file.exists():
        raise FileNotFoundError(f"Missing expected RDF file: {rdf_file}")
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

    materialize_mapping(generated_mapping, materialized_output, rmlmapper_jar, base_uri)
    if compare_mode == "mapper":
        if not reference_mapping.exists():
            raise FileNotFoundError(f"Missing reference mapping: {reference_mapping}")
        materialize_mapping(
            reference_mapping,
            expected_materialized_output,
            rmlmapper_jar,
            base_uri,
        )
        expected_file = expected_materialized_output
    else:
        expected_file = rdf_file

    is_equal, only_generated, only_expected = compare_files(materialized_output, expected_file)

    if not keep_artifacts:
        generated_mapping.unlink(missing_ok=True)
        materialized_output.unlink(missing_ok=True)
        expected_materialized_output.unlink(missing_ok=True)

    if not is_equal:
        print(f"FAILED {case_dir.name}")
        if only_generated:
            print("  Only in generated output:")
            for line in sorted(only_generated):
                print(f"    {line}")
        if only_expected:
            print("  Only in expected output:")
            for line in sorted(only_expected):
                print(f"    {line}")
        return False

    print(f"PASSED {case_dir.name}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate reverse-engineered mappings against official test cases."
    )
    parser.add_argument(
        "--cases-dir",
        default="test_cases",
        help="Directory that contains the RML test case folders.",
    )
    parser.add_argument(
        "--mapper-jar",
        "--burp-jar",
        dest="rmlmapper_jar",
        default="rmlmapper.jar",
        help="Path to the mapper jar used to materialize the generated mapping.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only matching case directory names. Can be passed multiple times.",
    )
    parser.add_argument(
        "--base-uri",
        default="http://example.com/base/",
        help="Base URI passed to remap and BURP.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep generated_mapping.ttl and materialized_output.nq in each case directory.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("mapper", "output"),
        default="mapper",
        help="Compare against the official mapping materialized by the same mapper, or against output.nq directly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_dir = Path(args.cases_dir)
    rmlmapper_jar = Path(args.rmlmapper_jar)

    if not cases_dir.exists():
        print(f"Cases directory not found: {cases_dir}", file=sys.stderr)
        return 1
    if not rmlmapper_jar.exists():
        print(f"Mapper jar not found: {rmlmapper_jar}", file=sys.stderr)
        return 1

    case_dirs = sorted(path for path in cases_dir.iterdir() if path.is_dir())
    if args.case:
        selected = set(args.case)
        case_dirs = [path for path in case_dirs if path.name in selected]

    if not case_dirs:
        print("No matching test cases found.", file=sys.stderr)
        return 1

    passed = 0
    for case_dir in case_dirs:
        try:
            if run_case(
                case_dir,
                rmlmapper_jar,
                args.keep_artifacts,
                args.base_uri,
                args.compare_mode,
            ):
                passed += 1
        except BaseException as exc:
            print(f"ERROR {case_dir.name}: {exc}")

    total = len(case_dirs)
    print(f"\nSummary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
