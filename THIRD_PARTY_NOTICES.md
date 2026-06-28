# Third-Party Notices

This repository ships a small number of vendored third-party runtime artifacts
so a clean checkout can build and run the public `mere.run` package without an
extra bootstrap step.

It also includes small third-party evaluation fixtures where noted below.

When any vendored artifact changes, update this file in the same pull request
with the new upstream source, version or commit when known, and license data.

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
- upstream project: [`antirez/ds4`](https://github.com/antirez/ds4)
- pinned upstream commit: `be434773fe1c0335d76896ea07f62a376cd629e5`
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
