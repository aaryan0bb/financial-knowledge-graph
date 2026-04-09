import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import subprocess
import re

try:
    import PyPDF2  # noqa: F401 – optional, used if exiftool missing
except ImportError:
    PyPDF2 = None  # type: ignore


def ensure_properties(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee obj has a 'properties' dict and return it."""
    if "properties" not in obj or obj["properties"] is None:
        obj["properties"] = {}
    return obj["properties"]


def add_pubdate_to_file(json_path: Path, pub_date: str | None, in_place: bool = True, out_dir: Path | None = None):
    """Read one triplet JSON file, inject publication_date, and save.

    Parameters
    ----------
    json_path : Path
        File to process.
    pub_date : str
        Publication date string to insert (e.g. '2025-07-29').
    in_place : bool, default True
        If True, overwrite the original; otherwise write to `out_dir`.
    out_dir : Path | None
        Required if in_place is False.
    """
    data = json.loads(json_path.read_text())

    # derive pub_date if not provided and a PDF exists alongside JSON
    if pub_date is None:
        pdf_date = auto_find_date(json_path)
        pub_val = pdf_date or datetime.today().strftime("%Y-%m-%d")
    else:
        pub_val = pub_date

    for ent in data.get("entities", []):
        props = ensure_properties(ent)
        props["publication_date"] = pub_val

    for rel in data.get("relationships", []):
        props = ensure_properties(rel)
        props["publication_date"] = pub_val

    if in_place:
        json_path.write_text(json.dumps(data, indent=2))
        print(f"Updated {json_path}")
    else:
        if out_dir is None:
            raise ValueError("out_dir must be provided when in_place is False")
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / json_path.name
        target.write_text(json.dumps(data, indent=2))
        print(f"Wrote {target}")


def main():
    parser = argparse.ArgumentParser(description="Inject publication_date into triplet JSON files")
    parser.add_argument("paths", nargs="+", help="One or more JSON files or directories containing them")
    parser.add_argument("--pub_date", help="If given, force this publication date string for all files.")
    parser.add_argument("--pdf_root", help="Directory containing original PDFs with the same basename as JSON (used when --pub_date not given)")
    parser.add_argument("--copy_to", help="Output directory; if omitted we overwrite in place")
    args = parser.parse_args()

    pub_date_param = args.pub_date  # may be None
    out_dir = Path(args.copy_to) if args.copy_to else None
    in_place = out_dir is None

    to_process = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            to_process.extend(path.rglob("*.json"))
        else:
            to_process.append(path)

    for json_file in to_process:
        try:
            add_pubdate_to_file(json_file, pub_date_param, in_place=in_place, out_dir=out_dir)
        except Exception as exc:
            print(f"Failed to process {json_file}: {exc}")


if __name__ == "__main__":
    main()

#######################
# PDF date extraction  #
#######################


_PDF_DATE_PATTERN = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


def extract_date_exiftool(pdf_path: Path) -> str | None:
    """Use exiftool (if available) to fetch CreateDate/ModifyDate/XMP date."""
    try:
        completed = subprocess.run(
            ["exiftool", "-j", "-n", "-CreateDate", "-MetadataDate", "-ModifyDate", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            return None
        arr = json.loads(completed.stdout)
        if not arr:
            return None
        rec = arr[0]
        for key in ("CreateDate", "MetadataDate", "ModifyDate"):
            if key in rec:
                return rec[key].split(" ")[0]  # take YYYY:MM:DD part
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    return None


def extract_date_pypdf(pdf_path: Path) -> str | None:
    if PyPDF2 is None:
        return None
    try:
        with pdf_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            meta = getattr(reader, "metadata", None) or reader.getDocumentInfo()
            if not meta:
                return None
            for key in ("/CreationDate", "/ModDate"):
                val = meta.get(key)
                if val:
                    m = _PDF_DATE_PATTERN.match(val)
                    if m:
                        y, mo, d = map(int, m.groups())
                        return f"{y:04d}-{mo:02d}-{d:02d}"
    except Exception:
        return None
    return None


def auto_find_date(json_path: Path) -> str | None:
    """Try to locate a PDF with matching basename and extract its date."""
    base = json_path.stem.split("_triplets")[0]  # crude
    possible = list(json_path.parent.glob(base + "*.pdf"))
    if not possible:
        return None
    pdf_path = possible[0]
    date = extract_date_exiftool(pdf_path) or extract_date_pypdf(pdf_path)
    if date is None:
        # fallback to OS mtime
        ts = pdf_path.stat().st_mtime
        date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return date 