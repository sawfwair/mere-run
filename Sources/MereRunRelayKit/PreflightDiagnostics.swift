import Foundation

/// Typed diagnostics shared by graph validation, preflight, and their
/// clients. Part of the published preflight contract; the declarative action
/// system that references these by id stays with the CLI.
public enum PreflightDiagnosticSeverity: String, Codable, Equatable, Sendable {
    case blocker
    case warning
    case note
    case estimate
}

public struct PreflightDiagnostic: Codable, Equatable, Sendable {
    public let id: String
    public let severity: PreflightDiagnosticSeverity
    public let title: String
    public let message: String
    public let locations: [DiagnosticLocation]
    public let suggestedActionIDs: [String]

    public init(
        id: String,
        severity: PreflightDiagnosticSeverity,
        title: String,
        message: String,
        locations: [DiagnosticLocation] = [],
        suggestedActionIDs: [String] = []
    ) {
        self.id = id
        self.severity = severity
        self.title = title
        self.message = message
        self.locations = locations
        self.suggestedActionIDs = suggestedActionIDs
    }

    enum CodingKeys: String, CodingKey {
        case id
        case severity
        case title
        case message
        case locations
        case suggestedActionIDs = "suggested_action_ids"
    }

    public func withSuggestedActionIDs(_ actionIDs: [String]) -> PreflightDiagnostic {
        PreflightDiagnostic(
            id: id,
            severity: severity,
            title: title,
            message: message,
            locations: locations,
            suggestedActionIDs: actionIDs
        )
    }
}

public struct DiagnosticLocation: Codable, Equatable, Sendable {
    public let kind: String
    public let path: String?
    public let line: Int?

    public init(kind: String, path: String? = nil, line: Int? = nil) {
        self.kind = kind
        self.path = path
        self.line = line
    }
}

public enum StructuredRunStatus: String, Codable, Equatable, Sendable {
    case ok
    case warning
    case blocked
    case planned
    case queued
    case assigned
    case running
    case finished
    case failed
    case cancelled

    public static func forDiagnostics(_ diagnostics: [PreflightDiagnostic]) -> StructuredRunStatus {
        if diagnostics.contains(where: { $0.severity == .blocker }) {
            return .blocked
        }
        if diagnostics.contains(where: { $0.severity == .warning }) {
            return .warning
        }
        return .ok
    }
}
