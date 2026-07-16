# FaceAnalysis

This module owns deterministic local inference for the managed
`vision-face-buffalo-l` model. It intentionally stops at raw, reusable face
primitives; library traversal, databases, identity labels, clustering, and
review/export workflows belong in `mere-run-plugins`.

## Data flow

1. `FaceAnalysisResources` validates the pinned detector and recognizer files.
2. `FaceImageProcessing` converts `MediaImage` pixels into InsightFace detector
   input and aligns a selected face from five landmarks for ArcFace.
3. `FaceDetector` decodes the pinned SCRFD/RetinaFace-style ONNX outputs and
   applies non-maximum suppression.
4. `FaceEmbedder` returns an L2-normalized 512-dimensional identity embedding.
5. `FaceAnalyzer` exposes detection and selected-face embedding as typed,
   codable results.

`FaceONNXSession` is the only ONNX Runtime boundary. The macOS implementation
supports CPU and explicit Core ML execution providers; `auto` currently uses
CPU because the pinned Buffalo-L graphs benchmark faster there on Apple
Silicon. Unsupported package platforms fail with a typed runtime error while
the rest of core remains buildable.

## Model contract

The implementation is coupled to the exact pinned input/output names, shapes,
preprocessing constants, and file sizes in `FaceAnalysisResources`. A different
InsightFace export needs a new managed model contract rather than silent shape
guessing.

These models provide face boxes, five-point landmarks, alignment, and identity
similarity. They do not infer emotion, demographics, liveness, deepfakes, or
general visual semantics.

## Model license

The ONNX Runtime dependency is MIT-licensed and is bundled with its complete
notice. Buffalo-L model weights are downloaded separately and remain limited
by InsightFace to non-commercial research use. `model pull` requires an
explicit `--accept-model-license` acknowledgment and prints the upstream
license URL before downloading them.
