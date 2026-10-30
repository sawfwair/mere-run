"""Generate small JPEG/RGB goldens with Pillow; tests need no Python runtime."""
import hashlib
import json
from pathlib import Path
from PIL import Image, features
import PIL

root = Path(__file__).parent
rgb = Image.new('RGB', (17, 13))
rgb.putdata([((x * 47 + y * 13) % 256, (x * 19 + y * 61) % 256, (x * 83 + y * 7) % 256)
             for y in range(13) for x in range(17)])
gray = Image.new('L', (17, 13))
gray.putdata([(x * 47 + y * 31) % 256 for y in range(13) for x in range(17)])
cmyk = Image.new('CMYK', (17, 13))
cmyk.putdata([((x * 11 + y * 17) % 256, (x * 23 + y * 3) % 256,
               (x * 5 + y * 31) % 256, (x * 37 + y * 43) % 256)
              for y in range(13) for x in range(17)])
cases = []
variants = [(f'rgb-{subsampling}', rgb, dict(subsampling=subsampling)) for subsampling in [0, 1, 2]]
variants += [('progressive-420', rgb, dict(subsampling=2, progressive=True)),
             ('grayscale', gray, {}), ('cmyk', cmyk, {}), ('rgb-coded', rgb, dict(keep_rgb=True))]
for name, image, options in variants:
    source = root / (name + '.jpeg')
    image.save(source, quality=91, **options)
    decoded = Image.open(source).convert('RGB')
    pixels = decoded.tobytes()
    (root / (name + '.rgb')).write_bytes(pixels)
    cases.append(dict(name=name, width=decoded.width, height=decoded.height,
                      jpegSHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      rgbSHA256=hashlib.sha256(pixels).hexdigest()))
rgb.save(root / 'rgb.png')
(root / 'manifest.json').write_text(json.dumps(dict(pillow=PIL.__version__,
    libjpegTurbo=features.version('libjpeg_turbo'), cases=cases), indent=2) + '\n')
