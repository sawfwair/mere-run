import Foundation

package enum StudioLibrarySource: String, Codable, Equatable {
    case raycast

    package var title: String {
        switch self {
        case .raycast: return "Raycast"
        }
    }
}

package enum StudioLibraryImportKind: String, Codable, Equatable {
    case image
    case video
    case music
    case speech

    package var mode: StudioMode {
        switch self {
        case .image: return .createImage
        case .video: return .video
        case .music: return .music
        case .speech: return .speak
        }
    }

    package var expectedOutputKind: StudioOutputFileKind {
        switch self {
        case .image: return .image
        case .video: return .video
        case .music, .speech: return .audio
        }
    }

    package var commandPreview: String {
        switch self {
        case .image: return "mere.run image generate"
        case .video: return "mere.run video generate"
        case .music: return "mere.run music generate"
        case .speech: return "mere.run speech synthesize"
        }
    }
}

package struct StudioLibraryImportReceipt: Codable, Equatable {
    package static let currentVersion = 1
    package static let maximumByteCount = 256 * 1_024

    package let version: Int
    package let id: UUID
    package let source: StudioLibrarySource
    package let kind: StudioLibraryImportKind
    package let prompt: String
    package let artifactPath: String
    package let createdAt: Date

    package static func load(
        from receiptURL: URL,
        fileManager: FileManager = .default
    ) throws -> StudioLibraryImportReceipt {
        let attributes: [FileAttributeKey: Any]
        do {
            attributes = try fileManager.attributesOfItem(atPath: receiptURL.path)
        } catch {
            throw StudioLibraryImportError.invalidReceipt
        }
        guard attributes[.type] as? FileAttributeType == .typeRegular else {
            throw StudioLibraryImportError.invalidReceipt
        }
        let byteCount = (attributes[.size] as? NSNumber)?.intValue ?? 0
        guard byteCount <= maximumByteCount else {
            throw StudioLibraryImportError.receiptTooLarge
        }

        let data: Data
        do {
            data = try Data(contentsOf: receiptURL)
        } catch {
            throw StudioLibraryImportError.invalidReceipt
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let receipt: StudioLibraryImportReceipt
        do {
            receipt = try decoder.decode(StudioLibraryImportReceipt.self, from: data)
        } catch {
            throw StudioLibraryImportError.invalidReceipt
        }

        guard receipt.version == currentVersion else {
            throw StudioLibraryImportError.unsupportedVersion(receipt.version)
        }
        guard !receipt.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw StudioLibraryImportError.emptyPrompt
        }
        return receipt
    }

    package func artifactURL(fileManager: FileManager = .default) throws -> URL {
        guard NSString(string: artifactPath).isAbsolutePath else {
            throw StudioLibraryImportError.artifactPathMustBeAbsolute
        }

        let url = URL(fileURLWithPath: artifactPath).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw StudioLibraryImportError.artifactNotFound(url.path)
        }
        guard !isDirectory.boolValue else {
            throw StudioLibraryImportError.artifactNotAFile(url.path)
        }
        guard fileManager.isReadableFile(atPath: url.path) else {
            throw StudioLibraryImportError.artifactNotReadable(url.path)
        }
        guard StudioOutputFileKind.classify(url) == kind.expectedOutputKind else {
            throw StudioLibraryImportError.artifactKindMismatch(kind.rawValue)
        }
        return url
    }
}

package enum StudioLibraryImportError: LocalizedError, Equatable {
    case invalidReceipt
    case receiptTooLarge
    case unsupportedVersion(Int)
    case emptyPrompt
    case artifactPathMustBeAbsolute
    case artifactNotFound(String)
    case artifactNotAFile(String)
    case artifactNotReadable(String)
    case artifactKindMismatch(String)
    case receiptIDConflict(UUID)

    package var errorDescription: String? {
        switch self {
        case .invalidReceipt:
            return "The Library import receipt is not valid JSON or is missing required fields."
        case .receiptTooLarge:
            return "The Library import receipt is larger than 256 KiB."
        case .unsupportedVersion(let version):
            return "Library import receipt version \(version) is not supported."
        case .emptyPrompt:
            return "The Library import receipt has an empty prompt."
        case .artifactPathMustBeAbsolute:
            return "The imported artifact path must be absolute."
        case .artifactNotFound(let path):
            return "The imported artifact does not exist: \(path)"
        case .artifactNotAFile(let path):
            return "The imported artifact is not a file: \(path)"
        case .artifactNotReadable(let path):
            return "The imported artifact is not readable: \(path)"
        case .artifactKindMismatch(let kind):
            return "The imported artifact does not match the declared \(kind) kind."
        case .receiptIDConflict(let id):
            return "Library item \(id.uuidString) already refers to another artifact."
        }
    }
}
