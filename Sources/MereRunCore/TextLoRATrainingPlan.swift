import Foundation

/// A validated dataset snapshot. Tensor loading, tokenization and optimizer
/// checkpoint compatibility remain owned by each native training pipeline.
public struct TextLoRATrainingPlan: Sendable {
    public let options: TextLoRATrainingOptions
    public let family: TextLoRATrainingFamily
    public let dataURL: URL
    public let outputURL: URL
    public let dataset: TextSFTPreparedDataset
    public let evaluationDataset: TextSFTPreparedDataset?

    public var evalPromptCount: Int? { evaluationDataset?.examples.count }

    public static func resolve(_ options: TextLoRATrainingOptions) throws -> Self {
        try options.validate()
        let family = try options.resolvedTrainingFamily()
        let dataURL = URL(fileURLWithPath: options.data).standardizedFileURL
        let outputURL = URL(fileURLWithPath: options.output).standardizedFileURL
        let policy: TextSFTMediaPolicy = family == .gemma4VLM ? .requireSingleLocalImage : .forbid
        let dataset = try TextSFTDataset.loadForTraining(from: dataURL, mediaPolicy: policy)
        let evaluation = try options.eval.map {
            try TextSFTDataset.loadForTraining(from: URL(fileURLWithPath: $0).standardizedFileURL, mediaPolicy: policy)
        }
        return Self(options: options, family: family, dataURL: dataURL, outputURL: outputURL,
                    dataset: dataset, evaluationDataset: evaluation)
    }
}
