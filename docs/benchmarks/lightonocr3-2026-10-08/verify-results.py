"""Verify saved native outputs against known text, table rows and region geometry."""
import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', html.unescape(text)).casefold().split())


class TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(normalized(''.join(self.cell)))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union > 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    checks = []
    for size in ['1B', '0.8B', '4B']:
        for case in manifest['cases']:
            for mode in ['plain', 'grounding']:
                key = f"{size}-{case['id']}-{mode}"
                record = json.loads((args.results/(key+'.json')).read_text())
                assert (record['size'], record['caseID'], record['mode']) == (size, case['id'], mode)
                failures = []
                text = record['text']
                missing = [anchor for anchor in case['anchors'] if normalized(anchor) not in normalized(text)]
                if missing:
                    failures.append({'check': 'text_anchors', 'missing': missing})
                boxes = [(label, [int(v) for v in coords]) for label, *coords in re.findall(
                    r'!\[([^\]]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', text)]
                invalid = [box for _, box in boxes if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)]
                if invalid:
                    failures.append({'check': 'coordinate_bounds', 'invalid': invalid})
                if case['id'] == 'blank':
                    if text.strip():
                        failures.append({'check': 'blank_page_content'})
                elif mode == 'grounding' and not boxes:
                    failures.append({'check': 'grounding_boxes_missing'})
                if record['tokensGenerated'] >= (64 if case['id'] == 'blank' else 1024):
                    failures.append({'check': 'token_cap'})
                table = TableRows()
                table.feed(text)
                missing_rows = [row for row in case.get('table_rows', []) if [normalized(cell) for cell in row] not in table.rows]
                if missing_rows:
                    failures.append({'check': 'table_row_alignment', 'missing': missing_rows})
                region_scores = []
                if mode == 'grounding':
                    for reference in case.get('grounding_regions', []):
                        score = max([iou(box, reference['box']) for label, box in boxes if label == reference['label']], default=0)
                        region_scores.append({'label': reference['label'], 'iou': score})
                        if score < 0.75:
                            failures.append({'check': 'grounding_region_iou', 'label': reference['label'], 'iou': score})
                checks.append({'id': key, 'passed': not failures, 'failures': failures,
                               'table_rows_checked': len(case.get('table_rows', [])), 'region_scores': region_scores})
    summary = {'cases': len(checks), 'passed': sum(check['passed'] for check in checks), 'checks': checks}
    args.summary.write_text(json.dumps(summary, indent=2)+'\n')
    print(f"{summary['passed']}/{summary['cases']} native output cases passed")
    for check in checks:
        if check['failures']:
            print(check['id'], json.dumps(check['failures']))
    return 0 if summary['passed'] == summary['cases'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
