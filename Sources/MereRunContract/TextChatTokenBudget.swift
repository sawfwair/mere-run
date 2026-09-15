import Foundation

/// Runtime token limits shared with shells that prepare chat commands.
/// Capability ranges remain presentation hints, not validation limits.
public enum TextChatTokenBudget {
    public struct Issue: Equatable, Sendable {
        public let field: String
        public let message: String
    }

    public static func issue(maxTokens: Int?, contextSize: Int?) -> Issue? {
        let upperBound = min(contextSize ?? Int(Int32.max), Int(Int32.max))
        guard upperBound > 0 else {
            return Issue(field: "context_size", message: "must be greater than zero")
        }
        if let maxTokens, !(1...upperBound).contains(maxTokens) {
            return Issue(field: "max_tokens", message: "must be between 1 and \(upperBound)")
        }
        return nil
    }
}
