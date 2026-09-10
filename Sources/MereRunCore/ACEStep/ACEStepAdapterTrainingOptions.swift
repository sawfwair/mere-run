import Foundation

public struct ACEStepAdapterTrainingOptions: Sendable {
    public let dataset: String
    public let output: String
    public let model: String
    public let configuration: ACEStepAdapterTrainingConfiguration
    public let maxDurationSeconds: Float
    public let checkpointsRoot: String?
    public let decoderSubdirectory: String
    public let vaeSubdirectory: String
    public let textSubdirectory: String?

    public init(
        dataset: String, output: String,
        model: String = ModelResolver.ModelID.aceStep.rawValue,
        configuration: ACEStepAdapterTrainingConfiguration = .init(),
        maxDurationSeconds: Float = 30,
        checkpointsRoot: String? = nil,
        decoderSubdirectory: String = "acestep-v15-turbo",
        vaeSubdirectory: String = "vae",
        textSubdirectory: String? = nil
    ) {
        self.dataset = dataset
        self.output = output
        self.model = model
        self.configuration = configuration
        self.maxDurationSeconds = maxDurationSeconds
        self.checkpointsRoot = checkpointsRoot
        self.decoderSubdirectory = decoderSubdirectory
        self.vaeSubdirectory = vaeSubdirectory
        self.textSubdirectory = textSubdirectory
    }

    public func validate() throws {
        guard configuration.kind == .lora || configuration.kind == .lokr else {
            throw ACEStepPreparationIssue("--kind must be lora or lokr.")
        }
        guard maxDurationSeconds > 0, maxDurationSeconds <= 600 else {
            throw ACEStepPreparationIssue("--max-duration must be in (0, 600].")
        }
    }

    var maximumAudioFrames: Int {
        max(1, Int((maxDurationSeconds * 48_000).rounded()))
    }
}
