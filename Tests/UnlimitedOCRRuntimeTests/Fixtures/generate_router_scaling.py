"""Generate independent FP32 goldens for upstream DASH scalar promotion."""
import json
import math
from pathlib import Path
import torch

scores = torch.tensor([0.1, 0.21, 1.1, 3.14159, 100.7, -7.25, 1e-5], dtype=torch.float32)
dimension, scale = 128, 128.0
expected = scores * (scale / math.sqrt(dimension))
old = (scores * scale) / math.sqrt(dimension)
assert (expected != old).any()
record = dict(torch=torch.__version__, dimension=dimension, scale=scale, input=scores.tolist(),
              summary=(scores * (1.0 / math.sqrt(dimension))).tolist(), routing=expected.tolist(),
              separatelyRounded=old.tolist())
Path(__file__).with_name('router-scaling.json').write_text(json.dumps(record, indent=2) + '\n')
