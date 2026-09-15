import Foundation

public enum ImageLoRATrainingIssue: LocalizedError, Sendable {
    public enum ModelRole: Sendable { case defaultTraining, training, sample }

    case invalid(String)
    case modelUnavailable(id: String, role: ModelRole)

    public init(_ message: String) { self = .invalid(message) }

    public var errorDescription: String? {
        message { id in
            let acknowledgement = ManagedModelCatalog.spec(for: id)?.usageRestriction == nil
                ? "" : " --accept-model-license"
            return "mere.run model pull \(id)\(acknowledgement)"
        }
    }

    public func message(modelPullCommand: (String) -> String) -> String {
        switch self {
        case .invalid(let message):
            return message
        case .modelUnavailable(let id, let role):
            let command = modelPullCommand(id)
            switch role {
            case .defaultTraining:
                return "Krea 2 Raw model \(id) not found. Pull it with `\(command)` or point --model at a local Raw model path."
            case .training:
                return "Model \(id) not found. Pull it with `\(command)` or point --model at a local path."
            case .sample:
                return "Sample model \(id) not found. Pull it with `\(command)` or pass --sample-model."
            }
        }
    }
}
