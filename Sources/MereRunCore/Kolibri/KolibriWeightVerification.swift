import Foundation
import Crypto

enum KolibriWeightVerification {
    struct Manifest: Decodable {
        struct File: Decodable { let sha256: String; let bytes: Int }
        let sourceRepository: String
        let sourceRevision: String
        let architectureRevision: String
        let outputFiles: [String: File]
        enum CodingKeys: String, CodingKey {
            case sourceRepository = "source_repository"
            case sourceRevision = "source_revision"
            case architectureRevision = "architecture_revision"
            case outputFiles = "output_files"
        }
    }

    static func verifyIfPresent(root: URL) throws {
        let manifestURL = root.appendingPathComponent("KOLIBRI_CONVERSION.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let index = try JSONDecoder().decode(KolibriLoader.Index.self,
                                            from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        guard manifest.sourceRepository == "Aleph-Alpha/Kolibri-1-BF16",
              manifest.sourceRevision == KolibriResources.sourceRevision,
              manifest.architectureRevision == "049a6a7bd2405b27d6d280d256bd3d585191c7ae",
              Set(manifest.outputFiles.keys) == Set(index.weightMap.values) else {
            throw ChatRequestIssue("checkpoint", "conversion provenance does not match the pinned Kolibri source")
        }
        for (name, expected) in manifest.outputFiles.sorted(by: { $0.key < $1.key }) {
            guard name == URL(fileURLWithPath: name).lastPathComponent else {
                throw ChatRequestIssue("checkpoint", "invalid conversion file name")
            }
            let url = root.appendingPathComponent(name)
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            var count = 0
            while let data = try handle.read(upToCount: 8 * 1_024 * 1_024), !data.isEmpty {
                try Task.checkCancellation()
                hasher.update(data: data)
                count += data.count
            }
            let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            guard count == expected.bytes, digest == expected.sha256 else {
                throw ChatRequestIssue("checkpoint", "conversion checksum mismatch for \(name)")
            }
        }
    }
}
