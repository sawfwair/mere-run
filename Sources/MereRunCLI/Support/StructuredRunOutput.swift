import ArgumentParser
import MereRunRelayKit
import Foundation

enum StructuredRunMode: String, Codable, Equatable, Sendable {
    case preflight
    case run
    case dryRun = "dry_run"
    case status
    case inspection
    case materialize
}
struct StructuredRunEnvelope<Request: Codable & Equatable, Result: Codable & Equatable>: Codable, Equatable {
    let schemaVersion: Int
    let mereRunVersion: String
    let command: [String]
    let mode: StructuredRunMode
    let status: StructuredRunStatus
    let createdAt: Date
    let cwd: String
    let summary: String
    let request: Request
    let result: Result
    let diagnostics: [PreflightDiagnostic]
    let actions: [DeclarativeAction]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case mereRunVersion = "mere_run_version"
        case command
        case mode
        case status
        case createdAt = "created_at"
        case cwd
        case summary
        case request
        case result
        case diagnostics
        case actions
    }
}

enum DeclarativeActionKind: String, Codable, Equatable, Sendable {
    case command
    case openFile = "open_file"
    case revealFile = "reveal_file"
    case openDirectory = "open_directory"
    case openURL = "open_url"
    case copyText = "copy_text"
    case select
    case none
}

enum DeclarativeActionStyle: String, Codable, Equatable, Sendable {
    case primary
    case secondary
    case danger
    case link
}

struct DeclarativeCommand: Codable, Equatable, Sendable {
    let argv: [String]
    let cwd: String
    let commandPath: [String]

    enum CodingKeys: String, CodingKey {
        case argv
        case cwd
        case commandPath = "command_path"
    }
}

struct DeclarativeActionCandidate: Codable, Equatable, Sendable {
    let id: String
    let label: String
    let value: String
    let path: String?
    let status: String?
    let description: String?
    let command: DeclarativeCommand?
    let patches: [DeclarativePatch]

    init(
        id: String,
        label: String,
        value: String,
        path: String? = nil,
        status: String? = nil,
        description: String? = nil,
        command: DeclarativeCommand? = nil,
        patches: [DeclarativePatch] = []
    ) {
        self.id = id
        self.label = label
        self.value = value
        self.path = path
        self.status = status
        self.description = description
        self.command = command
        self.patches = patches
    }
}

enum DeclarativePatchOperation: String, Codable, Equatable, Sendable {
    case replace
}

struct DeclarativePatch: Codable, Equatable, Sendable {
    let op: DeclarativePatchOperation
    let path: String
    let value: String
}

struct DeclarativeAction: Codable, Equatable, Sendable {
    let id: String
    let label: String
    let kind: DeclarativeActionKind
    let style: DeclarativeActionStyle
    let enabled: Bool
    let disabledReason: String?
    let command: DeclarativeCommand?
    let path: String?
    let url: String?
    let text: String?
    let candidates: [DeclarativeActionCandidate]
    let requires: [String]
    let confirmation: String?
    let destructive: Bool
    let secretsMasked: Bool

    init(
        id: String,
        label: String,
        kind: DeclarativeActionKind,
        style: DeclarativeActionStyle = .secondary,
        enabled: Bool = true,
        disabledReason: String? = nil,
        command: DeclarativeCommand? = nil,
        path: String? = nil,
        url: String? = nil,
        text: String? = nil,
        candidates: [DeclarativeActionCandidate] = [],
        requires: [String] = [],
        confirmation: String? = nil,
        destructive: Bool = false,
        secretsMasked: Bool = true
    ) {
        self.id = id
        self.label = label
        self.kind = kind
        self.style = style
        self.enabled = enabled
        self.disabledReason = disabledReason
        self.command = command
        self.path = path
        self.url = url
        self.text = text
        self.candidates = candidates
        self.requires = requires
        self.confirmation = confirmation
        self.destructive = destructive
        self.secretsMasked = secretsMasked
    }

    enum CodingKeys: String, CodingKey {
        case id
        case label
        case kind
        case style
        case enabled
        case disabledReason = "disabled_reason"
        case command
        case path
        case url
        case text
        case candidates
        case requires
        case confirmation
        case destructive
        case secretsMasked = "secrets_masked"
    }
}

enum StructuredRunOutput {
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    static func encode<T: Encodable>(_ value: T) throws -> String {
        let data = try encoder().encode(value)
        guard let output = String(data: data, encoding: .utf8) else {
            throw ValidationError("Could not encode structured output as UTF-8.")
        }
        return output
    }

    static func status(for diagnostics: [PreflightDiagnostic]) -> StructuredRunStatus {
        .forDiagnostics(diagnostics)
    }
}
