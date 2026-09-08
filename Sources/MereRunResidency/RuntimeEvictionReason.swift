import Foundation

public enum RuntimeEvictionReason: String, Codable, Equatable, Sendable {
    case ttl
    case memoryPressure = "memory_pressure"
}
