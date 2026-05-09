# SAM3

SAM 3.1 segmentation, prompting, tracking, video, and camera support.

- `SAM31Config.swift`: typed model configuration.
- `SAM31Tokenizer.swift`: tokenizer compatibility boundary.
- `SAM31ImageSegmenter.swift`: image segmentation pipeline.
- `SAM31VideoTracker.swift`: video tracking pipeline.
- `SAM31Prompts.swift`: prompt data structures.
- `SAM31VideoIO.swift`: video read/write boundary.

Keep video/graphics dictionaries at Apple framework boundaries and pass typed
prompt/result structures through the runtime.
