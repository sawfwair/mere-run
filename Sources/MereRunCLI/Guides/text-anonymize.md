# Text Anonymize

## Purpose

Detect and redact personally identifiable information in text with the OpenAI Privacy Filter model. Treat this as a privacy aid, not a compliance guarantee.

## Required Models

Default managed id: `text-anonymize-privacy-filter`.

## Install And Check

```bash
mere.run model pull text-anonymize-privacy-filter
mere.run text anonymize --help
```

## Parameters

- positional text arguments: one or more texts. If omitted, UTF-8 stdin is read.
- `--model`, `-m`: managed id or local model path.
- `--max-tokens`: clamp input length.
- `--replacement`: template with `{label}` and `{index}`.
- `--json`: emit structured spans.
- `--pretty`: pretty-print JSON.
- `--output`, `-o`: write to file.

## Usage Patterns

- Use plain text output for quick redaction.
- Use `--json --pretty` when you need span labels for audit or review.
- Choose a replacement format that preserves document readability, such as `[{label}_{index}]`.
- Keep a human review path for medical, legal, financial, HR, or government data.

## Examples

```bash
mere.run text anonymize "My name is Alice Smith and my email is alice@example.com"
```

```bash
cat notes.txt | mere.run text anonymize \
  --replacement "[{label}_{index}]" \
  --output ./notes-redacted.txt
```

```bash
mere.run text anonymize --json --pretty "API key sk-live-123 and phone 555-0100"
```

## Iteration Tips

- Review false positives and false negatives on local representative samples before broad use.
- Keep original and redacted data separated.
- For domain-specific policies, validate against local policy references.

## Troubleshooting

- Nothing happens from terminal: pipe stdin or pass text arguments.
- Missed span: try shorter text around the sensitive field and inspect JSON spans.
- Over-redaction: change replacement formatting only after confirming the detected span labels.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/TextAnonymizeCommand.swift
- https://huggingface.co/openai/privacy-filter
- https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf
