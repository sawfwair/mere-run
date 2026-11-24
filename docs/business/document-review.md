# Review documents and extract fields

Use Document Tools to turn a report into Markdown or extract named values from
an invoice. Conversion prepares readable text. Structured extraction prepares
a row for review, with field status and an optional routing suggestion. Neither
workflow posts a transaction or changes your business system.

## Choose the path

| Input and goal | Workflow | Prerequisites |
| --- | --- | --- |
| Office document or text-based PDF to Markdown | AnyDoc conversion | Document Tools; no model for conversion without redaction |
| Scan image to text | Native OCR | Compatible OCR runtime and checkpoint |
| Named fields from PDF, image, or UTF-8 text | Clairic extraction | Compatible Clairic checkpoint and enough unified memory |

Follow [Set up companion workflows](./plugin-workflows.md), then install the
plugin and check conversion:

```bash
mere.run plugin install mere-doc-tools --yes
mere.run plugin run mere-doc-tools -- doctor --extractor anydoc --no-redact
```

The signed Mac bundle supplies its conversion dependencies. For source
installations, AnyDoc requires Python 3.10 or later. See the
[Document Tools guide](https://plugins-docs.mere.run/plugins/document-tools)
for formats and interpreter setup.

## Convert one document

Use one approved local file and a fresh output directory:

```bash
umask 077
mere.run plugin run mere-doc-tools -- process   --extractor anydoc --input ./report.docx   --output-dir ./report-review --no-redact
```

The directory contains `report.md` and `run.json`, including the conversion
version and output hashes. Conversion doesn't preserve page layout or export
embedded binary assets. Compare tables, reading order, totals, and footnotes
with the source before using the Markdown downstream.

AnyDoc doesn't use hosted OCR. Scanned PDFs fail locally instead of uploading
the document. Render scanned pages as images and use the documented OCR path.
To enable local PII reduction, install `text-anonymize-privacy-filter` and omit
`--no-redact`. The output still contains the original unredacted Markdown.

## Extract invoice fields

Check the [Clairic runtime requirements](../runtime/clairic.md), then install a
compatible model such as `text-extract-clairic-flash-bf16`. Its approximately
19 GB of checkpoint files are not its full runtime memory requirement.

Save these field definitions in `fields.json`:

```json
{
  "vendor": {"type": "text", "instructions": "The invoice vendor"},
  "total": {"type": "number", "decimals": 2}
}
```

Preview extraction, then run it in a fresh output directory:

```bash
mere.run plugin run mere-doc-tools -- extract-fields   --input ./invoice.pdf --fields-json ./fields.json   --output-dir ./invoice-review --dry-run
mere.run plugin run mere-doc-tools -- extract-fields   --input ./invoice.pdf --fields-json ./fields.json   --output-dir ./invoice-review
```

`review.json` records the model response and field status. `fields.csv` exposes
the row for spreadsheet review, and `run.json` records output hashes. Compare
values, currency, decimal placement, and missing fields against the invoice.
Don't infer an absent value or approve payment from model output alone.

## Add reviewable routing

The optional `--classify-json` argument accepts a file of GLiNER tasks and adds
routing suggestions from extracted values. It requires GLiNER2.5 Decide.
Incomplete or invalid extraction holds routing for review. See the plugin
reference for the task schema.

Graph Studio exposes `document.extract` for the same structured workflow and
`document.convert` for conversion. Conversion graph outputs are unredacted.
Keep raw text, review JSON, CSV files, and receipts private until reviewed.
