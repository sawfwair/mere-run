import Foundation

public enum WorkflowExecutorKind: String, Codable, Equatable, Sendable {
    case ssh
    case relay
}

public struct WorkflowExecutorProfile: Codable, Equatable, Sendable {
    public let name: String
    public let kind: WorkflowExecutorKind
    public let destination: String?
    public let remoteRoot: String?
    public let port: Int?
    public let identityFile: String?
    public let mereRunPath: String?
    public let url: String?
    public let tokenFile: String?

    enum CodingKeys: String, CodingKey {
        case name
        case kind
        case destination
        case remoteRoot = "remote_root"
        case port
        case identityFile = "identity_file"
        case mereRunPath = "mere_run_path"
        case url
        case tokenFile = "token_file"
    }

    public init(
        name: String,
        kind: WorkflowExecutorKind,
        destination: String?,
        remoteRoot: String?,
        port: Int?,
        identityFile: String?,
        mereRunPath: String?,
        url: String?,
        tokenFile: String?
    ) {
        self.name = name
        self.kind = kind
        self.destination = destination
        self.remoteRoot = remoteRoot
        self.port = port
        self.identityFile = identityFile
        self.mereRunPath = mereRunPath
        self.url = url
        self.tokenFile = tokenFile
    }
}

public struct WorkflowExecutorProfiles: Codable, Equatable {
    public let schemaVersion: Int
    public var profiles: [WorkflowExecutorProfile]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case profiles
    }

    public init(schemaVersion: Int, profiles: [WorkflowExecutorProfile]) {
        self.schemaVersion = schemaVersion
        self.profiles = profiles
    }
}

/// Loads and persists executor profiles. The storage location is supplied by
/// the caller: the CLI passes its application-support path and app clients
/// pass their own container, so this module never owns platform paths.
public enum WorkflowExecutorProfileStore {
    public static func load(
        from url: URL,
        fileManager: FileManager = .default
    ) throws -> WorkflowExecutorProfiles {
        guard fileManager.fileExists(atPath: url.path) else {
            return WorkflowExecutorProfiles(schemaVersion: 1, profiles: [])
        }
        let profiles = try JSONDecoder().decode(WorkflowExecutorProfiles.self, from: Data(contentsOf: url))
        guard profiles.schemaVersion == 1 else {
            throw RelayClientError("Unsupported executor profile schema_version \(profiles.schemaVersion).")
        }
        return profiles
    }

    public static func save(
        _ value: WorkflowExecutorProfiles,
        to url: URL,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try WorkflowBundleCodec.encoder().encode(value).write(to: url, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    public static func require(_ reference: String, profilesURL: URL) throws -> WorkflowExecutorProfile {
        let parsed = try WorkflowExecutorReference(reference)
        guard parsed.kind != nil else {
            throw RelayClientError("Executor profile references must include ssh: or relay:.")
        }
        guard let profile = try load(from: profilesURL).profiles.first(where: { $0.name == parsed.name }) else {
            throw RelayClientError("Executor profile not found: \(parsed.name)")
        }
        guard profile.kind == parsed.kind else {
            throw RelayClientError("Executor profile '\(parsed.name)' is not a \(parsed.kind!.rawValue) profile.")
        }
        return profile
    }
}

public struct WorkflowExecutorReference: Equatable, Sendable {
    public let kind: WorkflowExecutorKind?
    public let name: String

    public init(_ value: String) throws {
        if value == "local" {
            kind = nil
            name = "local"
            return
        }
        let parts = value.split(separator: ":", maxSplits: 1).map(String.init)
        guard parts.count == 2,
              let parsedKind = WorkflowExecutorKind(rawValue: parts[0]),
              isValidExecutorName(parts[1]) else {
            throw RelayClientError("Executor must be local, ssh:<profile>, or relay:<profile>.")
        }
        kind = parsedKind
        name = parts[1]
    }
}

public struct WorkflowRemoteReference: Equatable, Sendable {
    public let kind: WorkflowExecutorKind
    public let profile: String
    public let jobID: String

    public init(_ value: String) throws {
        guard let url = URL(string: value),
              let scheme = url.scheme,
              let kind = WorkflowExecutorKind(rawValue: scheme),
              let profile = url.host,
              isValidExecutorName(profile),
              url.query == nil,
              url.fragment == nil else {
            throw RelayClientError("Run reference must be ssh://<profile>/<job-id> or relay://<profile>/<job-id>.")
        }
        let jobID = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !jobID.isEmpty,
              !jobID.contains("/"),
              jobID.range(of: "^[A-Za-z0-9][A-Za-z0-9-]{0,127}$", options: .regularExpression) != nil else {
            throw RelayClientError("Remote job id is invalid.")
        }
        self.kind = kind
        self.profile = profile
        self.jobID = jobID
    }

    public var executorReference: String { "\(kind.rawValue):\(profile)" }
    public var rawValue: String { "\(kind.rawValue)://\(profile)/\(jobID)" }
}

package func isValidExecutorName(_ name: String) -> Bool {
    name.range(of: "^[a-z][a-z0-9-]{0,63}$", options: .regularExpression) != nil
}
