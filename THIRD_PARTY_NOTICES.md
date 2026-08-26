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

### Sparkle for macOS updates

- purpose: secure discovery, verification, and atomic installation of signed
  MereRun macOS app updates
- upstream project: [`sparkle-project/Sparkle`](https://github.com/sparkle-project/Sparkle),
  version `2.9.5`; MIT with the bundled component notices below
- package location: `MereRun.app/Contents/Frameworks/Sparkle.framework`

```text
Copyright (c) 2006-2013 Andy Matuschak.
Copyright (c) 2009-2013 Elgato Systems GmbH.
Copyright (c) 2011-2014 Kornel Lesiński.
Copyright (c) 2015-2017 Mayur Pawashe.
Copyright (c) 2014 C.W. Betts.
Copyright (c) 2014 Petroules Corporation.
Copyright (c) 2014 Big Nerd Ranch.
All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

bspatch.c and bsdiff.c, from bsdiff 4.3:

Copyright 2003-2005 Colin Percival
All rights reserved

Redistribution and use in source and binary forms, with or without modification,
are permitted providing that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

sais.c and sais.h, from sais-lite:

Copyright (c) 2008-2010 Yuta Mori All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Portable C implementation of Ed25519, from https://github.com/orlp/ed25519:

Copyright (c) 2015 Orson Peters <orsonpeters@gmail.com>

This software is provided 'as-is', without any express or implied warranty. In
no event will the authors be held liable for any damages arising from the use
of this software.

Permission is granted to anyone to use this software for any purpose, including
commercial applications, and to alter it and redistribute it freely, subject
to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim
   that you wrote the original software. If you use this software in a product,
   an acknowledgment in the product documentation would be appreciated but is
   not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

SUSignatureVerifier.m:

Copyright (c) 2011 Mark Hamlin.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted providing that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### NVIDIA CUTLASS CUDA JIT headers

- purpose: Linux CUDA packages bundle the CUTLASS and CuTe headers required by
  MLX native quantized kernels at NVRTC compile time
- upstream project: [`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass),
  version `4.4.2`; BSD-3-Clause
- package location: `include/cute`, `include/cutlass`, and
  `include/CUTLASS-LICENSE.txt`; the Python CuTe DSL subtree is not bundled

```text
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

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
runtime code and source-level notices, not model weights. Managed models that
are access-gated or carry material non-commercial, research-only, or revenue-
limited terms require explicit user acknowledgement before a new download. A
custom license alone does not trigger the mere.run gate. Their
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

### MLX Fast Qwen3.8 decode-kernel provenance

- purpose: the macOS Metal path for serial-exact affine 4-bit, group-size-64
  matrix-vector multiplication at speculative verification widths 2 through 9,
  plus the proposal-only affine-Q2 readout, top-32 selector, and exact
  affine-Q4 reranker
- upstream project:
  [`Layr-Labs/qwen-3.8-mtp-challenge`](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge),
  revision `0863b06ac16e26e48fc06e97444095b00feb66d4`
- upstream file: `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`
- license: MIT, Copyright (c) 2026 Layr Labs, Inc.

### Qwen3.5/3.6 GDN verification prework provenance

- purpose: the macOS Metal kernel that fuses Qwen-family gated-delta
  convolution, SiLU, q/k/v preparation, RMS normalization, scaling, and
  convolution-state advance during MTP target verification
- upstream project: [`jundot/omlx`](https://github.com/jundot/omlx), tag
  `v0.6.1`, revision `b587575f3696fbc86c236b906684a48a92f8b118`
- upstream file: `omlx/patches/qwen35_gdn_prework.py`
- lineage noted by the upstream project: adapted from `mlx-serve`, itself a port of the
  `mlxfast-challenge` Qwen3.5 packed GDN prework kernel
- license: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### Poolside Laguna 2.1 native runtime

- purpose: native Swift/MLX loading, attention, routed-MoE, tokenizer/template,
  tool parsing, DFlash, and guarded acceleration support for Poolside Laguna
  2.1 checkpoints; no model weights are vendored in this repository
- managed Laguna S source: `poolside/Laguna-S-2.1-NVFP4-mlx` revision
  `e6a961c3bbebfffd8fa5b42f243e375504f41edd`
- canonical released Laguna XS model: `poolside/Laguna-XS-2.1-NVFP4`
- managed MLX-native Laguna XS source:
  `poolside/Laguna-XS-2.1-NVFP4-mlx` revision
  `841778bda563a36104dd521e37d99218e46f4f25`
- license: OpenMDW License Agreement 1.1; managed downloads retain the
  checkpoint's license and attribution files
- permitted use: OpenMDW-1.1 allows commercial and non-commercial use subject
  to its attribution and acceptable-use terms

### MiniMax-H3 serving and acceleration research

- purpose: selected MiniMax-H3 serving behavior, deterministic oracle fixtures,
  and opt-in kernel and algorithm research were translated from or validated
  against [`antirez/h3.c`](https://github.com/antirez/h3.c) at commit
  `03cb1339825feb19bcafcc60685680cb9ec6e2fe`
- distribution boundary: h3.c source code, binaries, and libraries are not
  vendored, linked, or shipped by this repository; `scripts/h3c-oracle.sh`
  fetches the pinned external source into the ignored `.build/` directory only
  when a contributor explicitly invokes it
- license: h3.c is MIT-licensed; the transferred dynamic symmetric int8
  activation-quantization design is included in the upstream BSD-3-Clause
  notice reproduced below

```text
MIT License

Copyright (c) 2026 Salvatore Sanfilippo

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

h3.c's upstream third-party notice states that its rectangular Morton decoder
and dynamic symmetric int8 quantization / Metal 4 TensorOps scheduling design
are adapted from ccv's Metal FlashAttention `NAMatMulKernel` and
`NAInt8MatMulKernel`. Of those elements, only the dynamic symmetric int8
activation-quantization design informed this repository; mere.run does not
implement h3.c's Morton decoder or TensorOps scheduler. The upstream notice is
reproduced in full:

```text
Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Sortformer speaker-diarization runtime

- purpose: native Swift/MLX implementation of the Sortformer model graph,
  feature extraction, checkpoint loading, and diarization post-processing; no
  model weights are vendored in this repository
- adapted source: [`Blaizzy/mlx-audio-swift`](https://github.com/Blaizzy/mlx-audio-swift)
  commit `4266f988d170a83017d1e82e2e4654602f277f1d`; MIT License
- converted model: [`mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16`](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16)
  revision `e23e6404bd9859e93edbf94a740eb1c7fc58f12e`; installed separately
- source model: [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
  revision `fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`; governed by the NVIDIA
  Open Model License; the public, ungated managed pull retains source
  provenance and does not add a separate mere.run acceptance gate

```text
MIT License

Copyright (c) 2025 Prince Canuma

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
  through the managed model store, including its `LICENSE.md`; OpenMDW-1.1
  acceptance occurs by exercising the licensed rights, without a separate
  mere.run acceptance gate
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

### AP-BWE

- purpose: native Swift/MLX 16 kHz to 48 kHz speech bandwidth extension
- upstream project: [`yxlu-0102/AP-BWE`](https://github.com/yxlu-0102/AP-BWE)
- upstream commit used for the port: `751710f22404c27e5bcc983248f8b856a04b8422`
- code and pretrained-weight license: MIT

The model checkpoint is downloaded separately into the user's model store.
Managed installs retain and verify the upstream code and weights license files.

```text
MIT License

Copyright (c) 2023 Ye-Xin Lu

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

### UniverSR

- purpose: native Swift/MLX general-audio super-resolution to 48 kHz
- upstream project: [`woongzip1/UniverSR`](https://github.com/woongzip1/UniverSR)
- upstream commit used for the port: `26dc21c44e11f9f19e823f02b0d4641dd5ea5af2`
- source-code license: MIT
- official checkpoint: [`woongzip1/universr-audio`](https://huggingface.co/woongzip1/universr-audio)
- checkpoint revision: `1c3294844285af851b6ffa56cbde4e43cd41fc2b`
- checkpoint license: Creative Commons Attribution 4.0 International (CC BY 4.0)

The model checkpoint is downloaded separately into the user's model store.
Managed installs retain the official model card and verify it alongside the
checkpoint and source configuration. The checkpoint license is not MIT; users
redistributing the weights or adaptations must preserve the CC BY 4.0
attribution and license notice. See the official license text at
<https://creativecommons.org/licenses/by/4.0/legalcode>.

```text
MIT License

Copyright (c) 2026 Woongjib Choi, DSPAI Lab, Yonsei University

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

### WeeTodd MiniMax-H3 MPP projection research

- purpose: the lab-only BF16 Metal Performance Primitives projection shader
  structure and measured H3 tile choices were adapted from
  [`wee-todd/WeeTodd-Nodes`](https://github.com/wee-todd/WeeTodd-Nodes) at
  commit `e5b0e014db1abe4c86fedc195d12dfcd18562042` and translated to Swift/MLX
- distribution boundary: no WeeTodd model weights, runtime package, or Python
  source files are vendored or linked; the adapted primitive remains outside
  production dispatch until mere.run's exactness and benchmark gates qualify it
- license: Apache License 2.0

```text
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

### `vendor/mlx-swift_Cmlx.bundle`

- purpose: bundled MLX Metal shader resources used by MLX-backed runtime paths
- source project: [`sawfwair/mlx-swift`](https://github.com/sawfwair/mlx-swift),
  based on upstream [`ml-explore/mlx-swift`](https://github.com/ml-explore/mlx-swift)
  0.32.1
- pinned package revision: `7558b9cff75746e3ce25802aecbdc498b240af7f`
- embedded MLX revision: `11da2b33a51772c023e2f7d7bc4ba9b3ff7e03ef`
- incorporated upstream MLX v0.32.1 revision:
  `3a6219917e4535575ce5bce2fc2ba27a483a709b`
- generated-kernel and required core AOT source SHA-256:
  `36412403ff4f0117579f0bc4471a17083411422bfc446f9f11420614ddbeb9ee`
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
- pinned upstream commit: `4893e0c40fba03dbc85555faeb035799aa04e0b6`
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
