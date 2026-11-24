# Review and export redacted images

Use Media Scrub to cover sensitive areas in a still or folder of frames before
sharing images with a vendor or publishing a demonstration. You approve the
rectangles on every frame. The exporter applies opaque black covers and
produces metadata-free PNGs with a receipt.

This workflow changes image pixels. The separate OCR scrub workflow reduces
sensitive text in extracted text; it doesn't replace visual approval of the
image you intend to share.

## Before you begin

Follow [Set up companion workflows](./plugin-workflows.md) and install the
plugin:

```bash
mere.run plugin install mere-media-scrub --yes
mere.run plugin doctor mere-media-scrub
```

The visual review and export commands require no models or network access.
Choose approved inputs and keep review and output directories outside the
source folder. A review directory contains full-size unredacted previews, so
store it privately.

## Review every source

Create the review package and open it locally:

```bash
umask 077
mere.run plugin run mere-media-scrub -- redaction-review   --input ./frames --output-dir ./private-review
open ./private-review/review.html
```

Use **Actual size** to inspect small text, badges, screens, reflected content,
and identifying objects. Drag rectangles over sensitive areas or enter pixel
coordinates. Approve every frame, including frames that require no rectangles.
Then select **Download approved review** to save the approved JSON file.

Both commands accept a single still or a folder of frames in filename order.
They don't inspect unprovided frames in a video or remove sensitive audio.
If the final deliverable is video, review its complete visual and audio scope
separately.

## Export the approved pixels

Use the downloaded review with the same source files:

```bash
mere.run plugin run mere-media-scrub -- redact-export   --input ./frames --review ~/Downloads/approved-redactions.json   --output-dir ./redacted-delivery
```

The exporter checks filenames, oriented dimensions, pixel bounds, and source
SHA-256 hashes. A changed, missing, or unapproved source stops the batch.
Rebuild and approve the review if an input changes.

The result contains numbered RGB PNGs and `export.json`, which maps source
names to outputs and records rectangles and output hashes. Export applies EXIF
orientation, flattens transparency onto white, and omits source metadata.

## Check what you share

Open every exported PNG and confirm that the selected regions are fully covered,
including edge pixels and small identifiers. Share only the approved outputs
and the receipt your recipient needs. Keep input and private review directories
separate. A valid
receipt proves which files and rectangles were used; it doesn't prove that a
reviewer found every sensitive region.

For the OCR-based text workflow and model requirements, see the
[Media Scrub reference](https://plugins-docs.mere.run/plugins/media-scrub).
