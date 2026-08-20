# Multimodal embeddings

This directory owns generic image/text embedding runtimes. It must not contain
application-specific search policy, identity claims, alert thresholds, or
tracking behavior.

`Qwen3VLEmbeddingModel` formats text and images with the upstream Qwen3-VL
embedding chat template, runs the shared native MLX vision-language encoder,
pools the final valid token, and returns an L2-normalized vector. Consumers own
indexing, similarity thresholds, reranking, and any downstream tracking.
