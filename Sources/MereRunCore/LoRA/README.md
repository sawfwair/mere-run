# LoRA Module

This directory owns LoRA checkpoint loading, artifact management, and compatibility layers for training and inference.

- resolvers and loaders map checkpoint layouts into runtime structures
- metrics and manifest files preserve resumable training state
- compatibility helpers adapt external or legacy formats into mere.run's canonical shape

This area is boundary-heavy. Prefer typed compatibility structs and narrow shims over pushing raw JSON dictionaries deeper into the runtime.
