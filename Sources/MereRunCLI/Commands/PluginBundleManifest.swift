import ArgumentParser
import Crypto
import Foundation

/// The wire format is owned by mere-run-plugins/contracts/plugin-bundle.v1.schema.json.
struct PluginBundleManifest: Codable, Equatable {
    let contractVersion: String
    let package: String
    let version: String
    let sequence: Int
    let sourceCommit: String
    let platform: String
    let minimumOSVersion: String
    let expiresAt: String
    let appBundle: String
    let entrypoints: [String: String]
    let artifact: Artifact

    struct Artifact: Codable, Equatable {
        let url: String
        let sha256: String
        let size: Int64
    }

    var bundleIdentifier: String { "run.mere.plugins.\(package)" }

    func validate(now: Date? = Date()) throws {
        guard contractVersion == "mere.run/plugin-bundle.v1",
              Self.matches(package, "^mere-[a-z0-9-]+$"),
              Self.matches(version, "^[0-9]+\\.[0-9]+\\.[0-9]+$"), sequence > 0,
              Self.matches(sourceCommit, "^[0-9a-f]{40}$"),
              platform == "macos-arm64", minimumOSVersion == "15.0",
              Self.matches(appBundle, "^[A-Za-z0-9-]+\\.app$"),
              !entrypoints.isEmpty, entrypoints.count <= 32,
              entrypoints.allSatisfy({ Self.matches($0.key, "^mere-[a-z0-9-]+$") && $0.key == $0.value }),
              Set(entrypoints.values).count == entrypoints.count,
              Self.matches(artifact.sha256, "^[0-9a-f]{64}$"),
              artifact.size > 0, artifact.size <= PluginBundleIO.maximumArchiveSize,
              let url = URL(string: artifact.url), url.scheme == "https", url.host != nil,
              url.user == nil, url.password == nil, url.fragment == nil,
              Self.matches(expiresAt, "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"),
              let expiry = ISO8601DateFormatter().date(from: expiresAt) else {
            throw ValidationError("Invalid or unsupported plugin bundle manifest.")
        }
        if let now, expiry <= now {
            throw ValidationError("Plugin bundle release metadata has expired. Request a current signed release.")
        }
    }

    static func matches(_ value: String, _ pattern: String) -> Bool {
        value.range(of: pattern, options: .regularExpression) == value.startIndex..<value.endIndex
    }
}

struct PluginBundleEnvelope: Codable {
    let contractVersion: String
    let keyID: String
    let payload: String
    let signature: String

    // The existing Mere release key is also used by the signed macOS app feed.
    // Rotation requires a reviewed CLI trust-root update; catalogs cannot add keys.
    static let trustedKeys = ["mere-release-1": "6sFs+7UqYcE7rThPAovzMDsZtKyf/h4/d8rUmPSH2rw="]
    static let developerTeam = "S5JDPCT8RC"

    func verified(
        trustedKeys: [String: String] = Self.trustedKeys,
        now: Date? = Date()
    ) throws -> PluginBundleManifest {
        guard contractVersion == "mere.run/plugin-bundle-envelope.v1",
              let encodedKey = trustedKeys[keyID], let key = Data(base64Encoded: encodedKey),
              let bytes = Data(base64Encoded: payload), bytes.count <= 32_768,
              let signatureBytes = Data(base64Encoded: signature),
              try Curve25519.Signing.PublicKey(rawRepresentation: key).isValidSignature(signatureBytes, for: bytes) else {
            throw ValidationError("Plugin bundle signature is invalid or its publisher key is not trusted. Installation stopped.")
        }
        let manifest = try JSONDecoder().decode(PluginBundleManifest.self, from: bytes)
        try manifest.validate(now: now)
        return manifest
    }
}
