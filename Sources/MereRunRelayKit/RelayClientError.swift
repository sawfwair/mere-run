import Foundation

/// Error thrown by MereRunRelayKit operations.
///
/// The message text is the user-facing diagnostic. The CLI maps this type onto
/// its own argument-parsing error so command output and exit codes stay
/// identical to the pre-extraction behavior; app clients present `message`
/// directly.
public struct RelayClientError: Error, Equatable, Sendable, CustomStringConvertible {
    public let message: String

    public init(_ message: String) {
        self.message = message
    }

    public var description: String { message }
}
