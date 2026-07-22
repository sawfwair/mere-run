# Third-Party Notices

This repository ships a small number of vendored third-party runtime artifacts
so a clean checkout can build and run the public `mere.run` package without an
extra bootstrap step.

It also includes small third-party evaluation fixtures where noted below.

It includes Google's two canonical Gemma 4 chat-template variants published on
2026-07-15. They are bundled as small runtime resources so known stale model
packages can receive Google's tool-calling and turn-closure fixes without
re-downloading unchanged weights. The E4B template comes from
`google/gemma-4-E4B-it` at revision
`fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`; the shared 12B/26B/31B template
comes from `google/gemma-4-12B-it` at revision
`12ace6d648d72bd41519e140f1185f34d38c7e3d`. Both upstream model repositories
identify Google as the template author and publish the model artifacts under
their accompanying Gemma terms.

Some native runtime implementations are ports of upstream open-source model
code; those source-derived implementations are noted below as well.

When any vendored artifact changes, update this file in the same pull request
with the new upstream source, version or commit when known, and license data.

## Binary package runtime dependencies

### ONNX Runtime for macOS face analysis

- purpose: executes the managed Buffalo-L detector and ArcFace recognizer
- Swift package: [`readdle/swift-onnxruntime`](https://github.com/readdle/swift-onnxruntime),
  version `1.20.1`, revision `f5c95540cc857b797c0d61cc62e398ad8688e5de`; MIT
- binary runtime: [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime),
  version `1.20.1`; MIT
- package artifact checksum:
  `31017531ebb064f1903a73ca5d55597ac49b6ae32360eb2482fb2f435342fdad`

Face model weights are downloaded separately by `mere.run model pull` and are
not vendored in this repository.

The same separation applies to every managed model: mere.run ships model
runtime code and source-level notices, not model weights. Managed models with
non-commercial, research-only, gated, revenue-limited, or custom acceptable-
use terms require explicit user acknowledgement before a new download. Their
exact source revisions and component-level license URLs are preserved in the
installed `mererun_model.json`; the authoritative inventory is in
[`docs/model-sources.md`](./docs/model-sources.md#restricted-model-downloads).
These third-party terms are separate from mere.run's source license, and the
user is responsible for determining whether an intended use complies.

#### Microsoft ONNX Runtime license

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#### Readdle swift-onnxruntime license

```text
MIT License

Copyright (c) 2024 Readdle Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Evaluation fixtures

### HumanEval slice

- purpose: three public HumanEval tasks embedded in `model benchmark code`
- upstream project: [`openai/human-eval`](https://github.com/openai/human-eval)
- included tasks: `HumanEval/0`, `HumanEval/3`, `HumanEval/8`
- license: MIT

```
The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Source-derived runtime implementations

### Cosmos3-Edge native omnimodal runtime

- purpose: native Swift/MLX implementation of the Cosmos3-Edge transformer,
  VAE, action encoder/decoder, scheduler, tokenizer, vision reasoner, and
  resident world-session behavior; no model weights are vendored in this
  repository
- architecture and reference implementation:
  [`NVIDIA/cosmos-framework`](https://github.com/NVIDIA/cosmos-framework) commit
  `ed8287fd7477113f8ac4f6b84290514d55cf0cdc`; OpenMDW License Agreement 1.1
- upstream copyright:
  `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
- official model: [`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge)
  revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de`; installed separately
  through the managed model store and guarded by explicit terms acceptance
- VAE and scheduler behavioral reference:
  [`huggingface/diffusers`](https://github.com/huggingface/diffusers) v0.39.0,
  commit `a3608b512ed7248499a44c61d954965ed9bdae4d`; Apache License 2.0
- parity fixtures: small generated numerical traces, tensor inventories, and
  scheduler/action reference values only; they do not contain model weights

The OpenMDW-1.1 license accompanying the pinned NVIDIA Model Materials is
reproduced in full below:

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)

By exercising rights granted to you under this agreement, you accept and agree
to its terms.

As used in this agreement, "Model Materials" means the materials provided to
you under this agreement, consisting of: (1) one or more machine learning
models (including architecture and parameters); and (2) all related artifacts
(including associated data, documentation and software) that are provided to
you hereunder.

Subject to your compliance with this agreement, permission is hereby granted,
free of charge, to deal in the Model Materials without restriction, including
under all copyright, patent, database, and trade secret rights included or
embodied therein.

If you distribute any portion of the Model Materials, you shall retain in your
distribution (1) a copy of this agreement, and (2) all copyright notices and
other notices of origin included in the Model Materials that are applicable to
your distribution.

If you file, maintain, or voluntarily participate in a lawsuit against any
person or entity asserting that the Model Materials directly or indirectly
infringes any patent or copyright, then all rights and grants made to you
hereunder are terminated, unless that lawsuit was in response to a
corresponding lawsuit first brought against you.

This agreement does not impose any restrictions or obligations with respect to
any use, modification, or sharing of any outputs generated by using the Model
Materials.

THE MODEL MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE, NONINFRINGEMENT, ACCURACY, OR THE
ABSENCE OF LATENT OR OTHER DEFECTS OR ERRORS, WHETHER OR NOT DISCOVERABLE, ALL
TO THE GREATEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW.

YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY
APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF, INCLUDING WITHOUT LIMITATION
ANY PERSON'S COPYRIGHTS OR OTHER RIGHTS INCLUDED OR EMBODIED IN THE MODEL
MATERIALS; (2) OBTAINING ANY NECESSARY CONSENTS, PERMISSIONS OR OTHER RIGHTS
REQUIRED FOR ANY USE OF THE MODEL MATERIALS; OR (3) PERFORMING ANY DUE
DILIGENCE OR UNDERTAKING ANY OTHER INVESTIGATIONS INTO THE MODEL MATERIALS OR
ANYTHING INCORPORATED OR EMBODIED THEREIN.

IN NO EVENT SHALL THE PROVIDERS OF THE MODEL MATERIALS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL MATERIALS, THE
USE THEREOF OR OTHER DEALINGS THEREIN.
```

### Native VFX geometry and reconstruction ports

The following native Swift/MLX implementations reproduce model graphs and
pre/post-processing behavior from pinned permissively licensed upstream code.
Model weights are downloaded separately into the user's model store and are
verified against the exact repository revision, byte count, and SHA-256 listed
in `Sources/MereRunCore/Geometry/GeometryModelPins.swift`. Direct managed model
installs also contain `MERERUN_UPSTREAM_LICENSE`, materialized from the pinned
source revision and checksum-verified as part of runtime readiness.

- **MoGe-2 ViT-S Normal** — geometry, camera, normal, and validity inference
  derived from [`microsoft/MoGe`](https://github.com/microsoft/MoGe) commit
  `07444410f1e33f402353b99d6ccd26bd31e469e8`; MIT. Its DINOv2 encoder behavior
  derives from [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2);
  Apache License 2.0.
- **Video Depth Anything Small and Metric Small** — temporal depth graph and
  windowing derived from
  [`DepthAnything/Video-Depth-Anything`](https://github.com/DepthAnything/Video-Depth-Anything)
  commit `4f5ae23172ba60fd7bc11ef671cca678842c7072`; Apache License 2.0. The shared
  DINOv2 encoder remains Apache License 2.0.
- **Depth Anything 3 Small** — multiview depth, confidence, camera, and pose
  conditioning derived from
  [`ByteDance-Seed/Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3)
  commit `41736238f5bced4debf3f2a12375d2466874866d`; Apache License 2.0. The shared
  DINOv2 encoder remains Apache License 2.0.
- **TripoSR** — reconstruction graph, camera conditioning, field decoder, and
  preprocessing derived from
  [`VAST-AI-Research/TripoSR`](https://github.com/VAST-AI-Research/TripoSR)
  commit `107cefdc244c39106fa830359024f6a2f1c78871`; MIT. The exact MIT license is
  mandatory in every converted package.
- **InstantMesh Base reconstruction** — sparse-view reconstruction graph,
  official camera conditioning, learned field, deformation, color, and the
  behavioral contract for the upstream empty-field sign repair follow
  [`TencentARC/InstantMesh`](https://github.com/TencentARC/InstantMesh) commit
  `08822c52fdc399b93ea00e4fa9e596344ed52ccc`; Apache License 2.0. The exact
  Apache-2.0 license is mandatory in every converted package. The native sign
  repair and polygonizer are independently implemented in Swift.

The InstantMesh integration is reconstruction-only. It does not include the
Zero123++ view generator or its weights. It also does not copy, import, compile,
or distribute NVIDIA's separately proprietary FlexiCubes or renderer source;
mesh topology is produced by mere.run's native marching-tetrahedra code and no
FlexiCubes topology-parity claim is made.

### MuScriptor

- purpose: native MLX audio-to-MIDI model, tokenizer, event decoder, and MIDI behavior
- upstream project: [`muscriptor/muscriptor`](https://github.com/muscriptor/muscriptor)
- upstream commit used for the port: `964e2350d5677eb3c3ca4d29e0e03286671e910a`
- license: MIT (model weights are separate and remain CC BY-NC 4.0)

```
MIT License

Copyright (c) 2026 Kyutai x Mirelo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MMAudio and bundled conditioning/decoding models

- purpose: native Swift/MLX text-to-audio and synchronized video-to-audio
  runtime
- architecture source: [`hkchengrex/MMAudio`](https://github.com/hkchengrex/MMAudio)
  at commit `974010a026c731054592d8f777218bd9d85a6c24`; MIT
- MMAudio model weights: downloaded separately; CC-BY-NC-4.0
- DFN5B CLIP model: downloaded separately from
  `apple/DFN5B-CLIP-ViT-H-14-378`; Apple Machine Learning Research Model
  License Agreement (research-only)
- BigVGAN-v2 source and model: downloaded separately from
  `nvidia/bigvgan_v2_44khz_128band_512x`; MIT

Managed installs retain the exact Apple and NVIDIA license files at
`clip/LICENSE` and `bigvgan/LICENSE`. These model licenses are separate from
the mere.run source license and from the MIT license on the MMAudio graph.

## Vendored artifacts

### `vendor/llama.xcframework`

- purpose: packaged `llama.cpp` runtime used by `mere.run text code` and `mere.run api serve`
- upstream project: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- pinned upstream commit: `4988f6e866057afd130c1515ecef0c9bab9a15f8` (`Add arch support for cohere2-MoE (#24260)`, 2026-06-13)
- rebuild note: regenerated from a neutral temporary checkout using the upstream `build-xcframework.sh` script; see [`scripts/rebuild_llama_xcframework.sh`](./scripts/rebuild_llama_xcframework.sh)
- license: MIT

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### `vendor/mlx-swift_Cmlx.bundle`

- purpose: bundled MLX Metal shader resources used by MLX-backed runtime paths
- upstream project: [`ml-explore/mlx-swift`](https://github.com/ml-explore/mlx-swift)
- pinned package version: `0.31.2`
- pinned package revision: `2512c5bebfd801c817b5d07828cbfdc44c76fab4`
- license: MIT

```
MIT License

Copyright (c) 2023 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### `vendor/magentart.xcframework`

- purpose: optional native Magenta RealTime 2 runtime used by `mere.run music generate --model music-magenta-rt2-*` and `mere.run music realtime`
- upstream project: [`magenta/magenta-realtime`](https://github.com/magenta/magenta-realtime)
- pinned upstream commit: `51836beddf5fbb33636830efd919884f40ef56c5`
- rebuild note: generated from a temporary upstream checkout plus the C ABI shim in [`scripts/rebuild_magentart_xcframework.sh`](./scripts/rebuild_magentart_xcframework.sh)
- code license: Apache License 2.0
- model weights: [`google/magenta-realtime-2`](https://huggingface.co/google/magenta-realtime-2), revision `010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc`, Creative Commons Attribution 4.0 International

```
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### `vendor/ds4`

- purpose: packaged DeepSeek V4 Flash runtime used by the premier setup-agent tier
- upstream project: [DwarfStar (`antirez/ds4`)](https://github.com/antirez/ds4)
- pinned upstream commit: `efdadd41e20134af4f3381e1ed90e96fe4faef6f`
- rebuild note: regenerated with [`scripts/rebuild_ds4.sh`](./scripts/rebuild_ds4.sh)
- license: MIT

```
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
