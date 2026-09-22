"""Export Falcon RGB resize fixtures with Pillow 12.3.0 and pinned source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import PIL
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert PIL.__version__ == '12.3.0'
source = args.reference.read_bytes()
source_hash = hashlib.sha256(source).hexdigest()
# The source identity is checked before its resize method is executed.
assert source_hash == '80ba44e6aa6eaa5f11b335b10fbe62bfcb63bc2b73c7a8fdbe049304b75e5d8d'
node = next(n for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == 'resize_image_if_necessary')
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
             str(args.reference), 'exec'), namespace)
rows = []
for name, width, height, mode in [
    ('landscape-down', 39, 23, 'initial'),
    ('portrait-down', 23, 39, 'initial'),
    ('small-up', 7, 5, 'initial'),
    ('unchanged', 20, 18, 'initial'),
    ('smart-two-axis', 17, 11, 'smart'),
    ('smart-horizontal', 17, 12, 'smart'),
    ('smart-vertical', 16, 11, 'smart'),
    ('smart-unchanged', 16, 12, 'smart'),
]:
    rgba = bytes(v for y in range(height) for x in range(width)
                 for v in ((x*73+y*29)%256, (x*x*11+y*53)%256,
                           255*((x+y)%2), (x*31+y*17)%256))
    image = Image.frombytes('RGBA', (width, height), rgba).convert('RGB')
    if mode == 'initial':
        result = namespace['resize_image_if_necessary'](image, 16, 32)
    else:
        result = image.resize((round(width/4)*4, round(height/4)*4),
                              Image.Resampling.BICUBIC)
    rows.append({'name': name, 'mode': mode, 'width': width, 'height': height,
                 'rgba': list(rgba), 'expectedWidth': result.width,
                 'expectedHeight': result.height, 'expectedRGB': list(result.tobytes())})
args.output.write_text(json.dumps({'pillowVersion': PIL.__version__,
                                  'sourceSha256': source_hash, 'cases': rows},
                                 separators=(',', ':'))+'\n')
