# Third-Party Notices

This repository ships a small number of vendored third-party runtime artifacts
so a clean checkout can build and run the public `mere.run` package without an
extra bootstrap step.

When any vendored artifact changes, update this file in the same pull request
with the new upstream source, version or commit when known, and license data.

## Vendored artifacts

### `vendor/llama.xcframework`

- purpose: packaged `llama.cpp` runtime used by `mere.run text code` and `mere.run api serve`
- upstream project: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- provenance note: source paths embedded in the packaged framework identify a local build from `ggml-org/llama.cpp`, but the exact upstream commit used for the current artifact is not recorded in this repo history
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
