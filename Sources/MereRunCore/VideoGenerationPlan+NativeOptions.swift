import Foundation

extension VideoGenerationPlan {
    public func miniMaxH3Options(
        prompt: String? = nil,
        adapterURL: URL?,
        firstFrameURL: URL?,
        lastFrameURL: URL?,
        frames: [MiniMaxH3FrameInput],
        references: [MiniMaxH3ReferenceInput]
    ) throws -> MiniMaxH3GenerationOptions {
        let options = self.options
        let weightMode: MiniMaxH3TransformerWeightMode
        switch options.h3WeightMode {
        case "auto": weightMode = .automatic
        case "quantized": weightMode = .quantized
        case "resident-bf16": weightMode = .residentBF16
        default:
            throw VideoGenerationIssue(id: "h3_weight_mode_invalid", title: "H3 weight mode is invalid", message: "Unknown MiniMax-H3 weight mode.")
        }
        guard let acceleration = MiniMaxH3AccelerationMode(rawValue: options.h3AccelerationMode) else {
            throw VideoGenerationIssue(id: "h3_acceleration_invalid", title: "H3 acceleration is invalid", message: "Unknown MiniMax-H3 acceleration mode.")
        }
        return try MiniMaxH3GenerationOptions(
            prompt: prompt ?? options.prompt,
            width: width,
            height: height,
            renderWidth: options.h3RenderWidth,
            renderHeight: options.h3RenderHeight,
            numFrames: numFrames,
            steps: options.steps,
            seed: UInt64(bitPattern: Int64(seed)),
            transformerWeightMode: weightMode,
            accelerationMode: acceleration,
            adapterURL: adapterURL,
            adapterStrength: options.h3AdapterStrength,
            firstFrameURL: firstFrameURL,
            lastFrameURL: lastFrameURL,
            frameInputs: frames,
            references: references
        )
    }

    public func wanOptions(
        prompt: String? = nil,
        sourceImageURL: URL
    ) throws -> Wan2GenerationOptions {
        let options = self.options
        return try Wan2GenerationOptions(
            prompt: prompt ?? options.prompt,
            negativePrompt: options.negativePrompt ?? Wan2Resources.defaultNegativePrompt,
            sourceImageURL: sourceImageURL,
            outputURL: options.outputURL,
            width: width,
            height: height,
            numFrames: numFrames,
            steps: options.steps ?? 40,
            guidanceScale: options.guidanceScale,
            shift: options.shift,
            seed: UInt64(bitPattern: Int64(seed)),
            fps: Int(fps)
        )
    }
}
