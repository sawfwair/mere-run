/// Durable execution states shared by operation families.
public enum RunState: String, Codable, Sendable {
    case preparing, running, succeeded, failed, cancelled, interrupted

    public var isTerminal: Bool { self != .preparing && self != .running }
}
