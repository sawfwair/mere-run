import Foundation
import MLX
import MLXFast
import MLXNN

public enum LTXTransformerExecution: String, Codable, CaseIterable, Sendable {
    case eager
    case compiled
}
