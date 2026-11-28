# MereRunVision

Vision-specialist runtimes carved out of `MereRunCore`:

- `SAM3/`: SAM 3.1 image segmentation and video tracking.
- `FalconPerception/`: Falcon-Perception grounding and detection.
- `LightOnOCR/` and `UnlimitedOCR/`: document OCR pipelines (`MereRunUnlimitedOCRModel` is scoped to this target).
- `Clairic/`: structured document extraction.
- `FaceAnalysis/`: ONNX/CoreML face analysis (`CONNXRuntime` and `COnnxRuntimeCoreML` are scoped to this target).
- `Pose/` and `OpticalFlow/`: pose estimation and optical flow.

Resource pins and catalog metadata remain in `MereRunCore` so on-device consumers (such as iOS Studio) do not compile or link desktop vision runtimes or ONNX Runtime.
