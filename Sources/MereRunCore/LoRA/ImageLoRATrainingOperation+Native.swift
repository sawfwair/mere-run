import Foundation

extension ImageLoRATrainingOperation {
    public static func train(
        _ plan: ImageLoRATrainingPlan,
        progress: (@Sendable (ImageLoRATrainingProgress) -> Void)?
    ) async throws {
        try Task.checkCancellation()
        switch plan.training {
        case .krea(let examples, let configuration):
            try await Krea2LoRATrainer.train(
                modelPath: plan.modelRoot.path, examples: examples,
                outputURL: plan.outputURL, config: configuration,
                progressHandler: { progress?(.krea($0)) }
            )
        case .klein(let examples, let configuration, let resumeFrom, let sample):
            let sampleHandler = sample.map {
                makeSampleHandler($0, outputURL: plan.outputURL, progress: progress)
            }
            try await Flux2KleinLoRATrainer.train(
                modelPath: plan.modelRoot.path, examples: examples,
                outputURL: plan.outputURL, config: configuration,
                resumeFromLoRA: resumeFrom,
                progressHandler: { progress?(.klein($0)) }, sampleHandler: sampleHandler
            )
        }
    }

    private static func makeSampleHandler(
        _ sample: ImageLoRATrainingPlan.Sample, outputURL: URL,
        progress: (@Sendable (ImageLoRATrainingProgress) -> Void)?
    ) -> @Sendable (Int, URL) async -> Void {
        let generator = Flux2KleinGenerator()
        let outputBaseName = outputURL.deletingPathExtension().lastPathComponent
        let sampleDirectory = outputURL.deletingLastPathComponent().appendingPathComponent("samples", isDirectory: true)
        return { step, checkpointURL in
            do {
                try Task.checkCancellation()
                try FileManager.default.createDirectory(at: sampleDirectory, withIntermediateDirectories: true)
                let sampleURL = sampleDirectory.appendingPathComponent("\(outputBaseName)-step\(step)-sample.png")
                let request = GenerationRequest(
                    prompt: sample.prompt, width: sample.width, height: sample.height,
                    steps: sample.steps, guidanceScale: sample.guidanceScale,
                    seed: sample.seed, outputURL: sampleURL, model: sample.modelPath,
                    lora: .local(path: checkpointURL.path, scale: sample.loraScale)
                )
                _ = try await generator.generate(request, progressHandler: nil)
                try Task.checkCancellation()
                progress?(.sampleSaved(step: step, url: sampleURL, checkpoint: checkpointURL))
            } catch {
                progress?(.sampleFailed(step: step, message: error.localizedDescription, checkpoint: checkpointURL))
            }
        }
    }
}
