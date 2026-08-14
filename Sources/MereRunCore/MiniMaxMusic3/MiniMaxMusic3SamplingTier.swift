import Foundation

public enum MiniMaxMusic3SamplingTier: String, CaseIterable, Codable, Sendable {
    case quality
    case fast
    case draft

    public var inferenceSteps: Int {
        switch self {
        case .quality:
            30
        case .fast:
            20
        case .draft:
            16
        }
    }
}
