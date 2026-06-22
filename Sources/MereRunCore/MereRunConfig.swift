import Foundation

/// Persisted user configuration for mere.run, stored as JSON at
/// `~/Library/Application Support/MereRun/config.json` (next to `models/`).
///
/// Holds credentials and endpoints that should survive across shells without relying on
/// environment variables — most importantly the Hugging Face access token used to pull
/// gated models. The file is written with `0600` permissions because it carries secrets.
public struct MereRunConfig: Codable, Equatable, Sendable {
    public var hfToken: String?
    public var hfEndpoint: String?

    public init(hfToken: String? = nil, hfEndpoint: String? = nil) {
        self.hfToken = hfToken
        self.hfEndpoint = hfEndpoint
    }

    enum CodingKeys: String, CodingKey {
        case hfToken = "hf_token"
        case hfEndpoint = "hf_endpoint"
    }

    // MARK: - Location

    public static var fileURL: URL {
        MereRunModelPaths.applicationSupportBase.appendingPathComponent("config.json", isDirectory: false)
    }

    // MARK: - Load / Save

    /// Loads the config, returning an empty config if the file is absent or unreadable.
    public static func load() -> MereRunConfig {
        let url = fileURL
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            return MereRunConfig()
        }
        return (try? JSONDecoder().decode(MereRunConfig.self, from: data)) ?? MereRunConfig()
    }

    /// Writes the config to disk with secret-safe (0600) permissions, creating the
    /// application-support directory if needed.
    public func save() throws {
        let fm = FileManager.default
        let url = Self.fileURL
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        try data.write(to: url, options: .atomic)
        // Re-assert restrictive permissions (atomic write can reset them).
        try? fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    // MARK: - Generic key access (for the `config` CLI)

    public enum Key: String, CaseIterable, Sendable {
        case hfToken = "hf-token"
        case hfEndpoint = "hf-endpoint"

        public var isSecret: Bool { self == .hfToken }
    }

    public func value(for key: Key) -> String? {
        switch key {
        case .hfToken: return hfToken
        case .hfEndpoint: return hfEndpoint
        }
    }

    public mutating func set(_ key: Key, _ value: String?) {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
        let stored = (trimmed?.isEmpty == false) ? trimmed : nil
        switch key {
        case .hfToken: hfToken = stored
        case .hfEndpoint: hfEndpoint = stored
        }
    }

    /// Masks a secret for display, e.g. `hf_abcd…WXYZ` (never prints the whole token).
    public static func masked(_ secret: String) -> String {
        let s = secret.trimmingCharacters(in: .whitespacesAndNewlines)
        guard s.count > 8 else { return String(repeating: "•", count: max(s.count, 4)) }
        return "\(s.prefix(4))…\(s.suffix(4))"
    }
}
