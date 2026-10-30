"""Fetch immutable qualification inputs and rasterize real IRS document pages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request


SOURCES = [
    ("fw9.pdf", "https://www.irs.gov/pub/irs-pdf/fw9.pdf",
     "2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53"),
    ("receipt.jpeg", "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/"
     "28fe12cdf7816b5dde94e22051b2ec8dc74267b7/SROIE-receipt.jpeg",
     "d66a3287c847823c6ddab9f8b2080582b96a66b7816d178472f697ea84e227b3"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, url, digest in SOURCES:
        destination = output / name
        if not destination.exists():
            snapshot = Path(__file__).parent / 'sources' / name
            if snapshot.exists():
                shutil.copyfile(snapshot, destination)
            else:
                urllib.request.urlretrieve(url, destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Source changed: {name}, expected {digest}, received {actual}")
        sources.append(dict(file=name, url=url, sha256=actual))
    for page in [1, 2]:
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "144", "-png",
                        "-singlefile", str(output / "fw9.pdf"), str(output / f"w9-page-{page}")], check=True)
    root = Path(__file__).resolve().parents[3]
    shutil.copyfile(root / "docs/benchmarks/lightonocr3-2026-10-08/inputs/document.png", output / "document.png")

    control = dict(image="document.png", anchors=["MERE RUN DOCUMENT CHECK", "OCR-3-TEST-008", "17 samples",
                                                  "23 samples", "Harbour sensor", "42.00", "18.50", "60.50"])
    receipt = dict(image="receipt.jpeg", anchors=["CASH BILL", "25/12/2018", "MANIS", "MODELLING CLAY", "KIDDY FISH", "9.00"])
    form = dict(image="w9-page-1.png", anchors=["Request for Taxpayer", "Taxpayer Identification Number",
                                               "Social security number", "Certification instructions", "General Instructions",
                                               "What's New", "Purpose of Form"],
                forbidden=["Nonresident alien who becomes a resident alien", "Form 1099-NEC", "after 1983"])
    instructions = dict(image="w9-page-2.png", anchors=["Form 1099-INT", "Form 1099-NEC", "Definition of a U.S. person",
                                                       "Foreign person", "Nonresident alien who becomes a resident alien",
                                                       "Backup Withholding", "24%", "after 1983"],
                        forbidden=["Request for Taxpayer Identification Number and Certification", "City, state, and ZIP code"])
    cases = [
        dict(id="document-control", crop="base", maxTokens=1024, pages=[control]),
        dict(id="receipt-base", crop="base", maxTokens=1024, pages=[receipt]),
        dict(id="receipt-auto", crop="auto", maxTokens=1024, pages=[receipt]),
        dict(id="w9-form-base", crop="base", maxTokens=4096, pages=[form]),
        dict(id="w9-form-auto", crop="auto", maxTokens=4096, pages=[form]),
        dict(id="w9-instructions-base", crop="base", maxTokens=4096, pages=[instructions]),
        dict(id="w9-two-page-base", crop="base", maxTokens=8192, pages=[form, instructions]),
        dict(id="w9-reversed-base", crop="base", maxTokens=8192, pages=[instructions, form]),
    ]
    images = {page["image"] for case in cases for page in case["pages"]}
    image_hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in sorted(images)}
    manifest = dict(sources=sources, rasterizer="pdftoppm", rasterDPI=144,
                    imageSHA256=image_hashes, cases=cases)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(cases)} cases in {output}")


if __name__ == "__main__":
    main()
