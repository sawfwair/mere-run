import Foundation

enum MereRunDeepLink: Equatable {
    static let scheme = "mererun"

    case preview(URL)

    static func parse(
        _ url: URL,
        fileManager: FileManager = .default
    ) throws -> MereRunDeepLink {
        guard url.scheme?.lowercased() == scheme,
              url.host?.lowercased() == "preview",
              url.path.isEmpty else {
            throw MereRunDeepLinkError.unsupportedRoute
        }

        guard url.user == nil, url.password == nil, url.port == nil,
              url.fragment == nil else {
            throw MereRunDeepLinkError.unsupportedRoute
        }

        let queryItems = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard !queryItems.isEmpty else {
            throw MereRunDeepLinkError.missingPath
        }
        guard queryItems.count == 1, queryItems[0].name == "path" else {
            throw MereRunDeepLinkError.unsupportedParameters
        }
        guard let path = queryItems[0].value, !path.isEmpty else {
            throw MereRunDeepLinkError.missingPath
        }

        guard NSString(string: path).isAbsolutePath else {
            throw MereRunDeepLinkError.pathMustBeAbsolute
        }

        let fileURL = URL(fileURLWithPath: path).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: fileURL.path, isDirectory: &isDirectory) else {
            throw MereRunDeepLinkError.fileNotFound(fileURL.path)
        }
        guard !isDirectory.boolValue else {
            throw MereRunDeepLinkError.notAFile(fileURL.path)
        }
        guard fileManager.isReadableFile(atPath: fileURL.path) else {
            throw MereRunDeepLinkError.fileNotReadable(fileURL.path)
        }

        return .preview(fileURL)
    }
}

enum MereRunDeepLinkError: LocalizedError, Equatable {
    case unsupportedRoute
    case missingPath
    case unsupportedParameters
    case pathMustBeAbsolute
    case fileNotFound(String)
    case notAFile(String)
    case fileNotReadable(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedRoute:
            return "Use mererun://preview with one percent-encoded path parameter."
        case .missingPath:
            return "The preview link is missing its artifact path."
        case .unsupportedParameters:
            return "The preview link must contain only one path parameter."
        case .pathMustBeAbsolute:
            return "The preview artifact path must be absolute."
        case .fileNotFound(let path):
            return "No preview artifact exists at \(path)."
        case .notAFile(let path):
            return "The preview target is not a file: \(path)."
        case .fileNotReadable(let path):
            return "The preview artifact is not readable: \(path)."
        }
    }
}
