import Foundation

extension ACEStepAdapterTrainingOperation {
    public static func train(
        _ plan: ACEStepAdapterTrainingPlan,
        progress: (@Sendable (ACEStepAdapterTrainingProgress) -> Void)?
    ) async throws -> ACEStepAdapterTrainingReport {
        try Task.checkCancellation()
        let options = plan.options
        let examples = try prepareExamples(plan)
        try Task.checkCancellation()
        let root = try await ACEStepRuntimePreparation.resolveCheckpointsRoot(
            model: options.model, checkpointsRoot: options.checkpointsRoot,
            turboSubdirectory: options.decoderSubdirectory,
            vaeSubdirectory: options.vaeSubdirectory,
            lmSubdirectory: nil, textSubdirectory: options.textSubdirectory
        )
        let decoder = try ACEStepRuntimePreparation.resolveTurboSubdirectory(
            at: root, explicit: options.decoderSubdirectory
        )
        guard let text = try ACEStepRuntimePreparation.resolveTextSubdirectory(
            at: root, explicit: options.textSubdirectory
        ) else {
            throw ACEStepPreparationIssue("ACE-Step text encoder not found.")
        }
        try Task.checkCancellation()
        let container = ACEStepModelContainer(
            checkpointsRootURL: root, turboSubdirectory: decoder,
            vaeSubdirectory: options.vaeSubdirectory, textEncoderSubdirectory: text
        )
        let resources = try await container.resources()
        try Task.checkCancellation()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            textEncoderResources: resources.textEncoderResources
        )
        return try pipeline.trainAdapter(
            examples: examples, configuration: options.configuration,
            outputURL: plan.outputURL, progress: progress
        )
    }

    static func prepareExamples(_ plan: ACEStepAdapterTrainingPlan) throws -> [ACEStepAdapterTrainingExample] {
        let options = plan.options
        return try plan.records.enumerated().map { index, record in
            try Task.checkCancellation()
            let decoded = try ACEStepRuntimePreparation.loadAudio48kHz(
                plan.audioURL(for: record).path, label: "Dataset audio \(index + 1)"
            )
            let maxFrames = options.maximumAudioFrames
            let audio = decoded.dim(1) > maxFrames
                ? decoded[0..., 0..<maxFrames, 0...]
                : decoded
            return ACEStepAdapterTrainingExample(
                audio48kHz: audio, caption: record.caption, lyrics: record.lyrics ?? ""
            )
        }
    }
}
