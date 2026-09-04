import CryptoKit
import Foundation

package enum CLIInstallationReceiptError: LocalizedError, Equatable {
    case unsupportedSchemaVersion(Int)
    case payloadOutsideManagedRoot(String)

    package var errorDescription: String? {
        switch self {
        case .unsupportedSchemaVersion(let version):
            return "Unsupported Studio CLI installation receipt schema version \(version)."
        case .payloadOutsideManagedRoot(let path):
            return "The Studio CLI receipt points outside the managed payload directory: \(path)."
        }
    }
}

package struct CLIInstallationReceipt: Codable, Equatable, Sendable {
    package static let currentSchemaVersion = 1

    package let schemaVersion: Int
    package let destinationPath: String
    package let payloadPath: String
    package let studioVersion: String
    package let studioBuild: String
    package let payloadFingerprint: String
    package let installedAssetNames: [String]
    package let installationTimestamp: Date

    package init(
        schemaVersion: Int = Self.currentSchemaVersion,
        destinationPath: String,
        payloadPath: String,
        studioVersion: String,
        studioBuild: String,
        payloadFingerprint: String,
        installedAssetNames: [String],
        installationTimestamp: Date
    ) {
        self.schemaVersion = schemaVersion
        self.destinationPath = destinationPath
        self.payloadPath = payloadPath
        self.studioVersion = studioVersion
        self.studioBuild = studioBuild
        self.payloadFingerprint = payloadFingerprint
        self.installedAssetNames = installedAssetNames
        self.installationTimestamp = installationTimestamp
    }

    package static func read(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        let receipt = try decoder.decode(Self.self, from: data)
        guard receipt.schemaVersion == currentSchemaVersion else {
            throw CLIInstallationReceiptError.unsupportedSchemaVersion(receipt.schemaVersion)
        }
        return receipt
    }

    package func write(to url: URL, fileManager fm: FileManager = .default) throws {
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.encoder.encode(self).write(to: url, options: .atomic)
    }

    private static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    private static var decoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

package struct CLIStudioBuild: Equatable, Sendable {
    package let version: String
    package let build: String

    package static func current(bundle: Bundle = .main) -> Self {
        let version = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        let build = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        return Self(version: version ?? "dev", build: build ?? "dev")
    }
}

package struct CLIInstallationPaths: Equatable, Sendable {
    package let applicationSupportRoot: URL
    package let payloadRoot: URL
    package let receiptURL: URL
    package let destinationCandidates: [URL]

    package init(applicationSupportRoot: URL, destinationCandidates: [URL]) {
        self.applicationSupportRoot = applicationSupportRoot
        payloadRoot = applicationSupportRoot.appendingPathComponent("cli", isDirectory: true)
        receiptURL = applicationSupportRoot.appendingPathComponent(
            "studio-cli-install.json",
            isDirectory: false
        )
        self.destinationCandidates = destinationCandidates
    }

    package static func current(fileManager fm: FileManager = .default) -> Self {
        let supportRoot = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("mere.run", isDirectory: true)
        return Self(
            applicationSupportRoot: supportRoot,
            destinationCandidates: CLIResolver.installedCandidates(fileManager: fm)
        )
    }
}

package struct CLIPayloadManifest: Equatable, Sendable {
    package let fingerprint: String
    package let assetNames: [String]
}

package enum CLIPayloadManifestBuilder {
    package static func manifest(at rootURL: URL, fileManager fm: FileManager = .default) throws -> CLIPayloadManifest {
        let assetURLs = try fm.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey],
            options: []
        ).sorted { $0.lastPathComponent < $1.lastPathComponent }

        var entries = assetURLs
        if let enumerator = fm.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey],
            options: []
        ) {
            for case let entryURL as URL in enumerator {
                if !assetURLs.contains(entryURL) {
                    entries.append(entryURL)
                }
            }
        }
        entries.sort { relativePath(for: $0, rootURL: rootURL) < relativePath(for: $1, rootURL: rootURL) }

        var hasher = SHA256()
        for entryURL in entries {
            let relativePath = relativePath(for: entryURL, rootURL: rootURL)
            let values = try entryURL.resourceValues(forKeys: [
                .fileSizeKey,
                .isDirectoryKey,
                .isRegularFileKey,
                .isSymbolicLinkKey,
            ])
            if values.isSymbolicLink == true {
                update(&hasher, marker: "link", path: relativePath)
                let destination = try fm.destinationOfSymbolicLink(atPath: entryURL.path)
                updateLength(destination.utf8.count, hasher: &hasher)
                hasher.update(data: Data(destination.utf8))
            } else if values.isDirectory == true {
                update(&hasher, marker: "directory", path: relativePath)
            } else if values.isRegularFile == true {
                update(&hasher, marker: "file", path: relativePath)
                updateLength(values.fileSize ?? 0, hasher: &hasher)
                let handle = try FileHandle(forReadingFrom: entryURL)
                defer { try? handle.close() }
                while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
                    hasher.update(data: chunk)
                }
            }
        }

        return CLIPayloadManifest(
            fingerprint: hasher.finalize().map { String(format: "%02x", $0) }.joined(),
            assetNames: assetURLs.map(\.lastPathComponent)
        )
    }

    private static func relativePath(for url: URL, rootURL: URL) -> String {
        let rootPath = rootURL.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(rootPath + "/") else { return url.lastPathComponent }
        return String(path.dropFirst(rootPath.count + 1))
    }

    private static func update(_ hasher: inout SHA256, marker: String, path: String) {
        hasher.update(data: Data(marker.utf8))
        hasher.update(data: Data([0]))
        hasher.update(data: Data(path.utf8))
        hasher.update(data: Data([0]))
    }

    private static func updateLength(_ length: Int, hasher: inout SHA256) {
        hasher.update(data: Data(String(length).utf8))
        hasher.update(data: Data([0]))
    }
}

extension URL {
    package func isContained(in rootURL: URL) -> Bool {
        let rootPath = rootURL.standardizedFileURL.path
        let candidatePath = standardizedFileURL.path
        return candidatePath == rootPath || candidatePath.hasPrefix(rootPath + "/")
    }
}
