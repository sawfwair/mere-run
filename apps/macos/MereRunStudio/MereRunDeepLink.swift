import Foundation

enum MereRunDeepLink: Equatable {
    static let scheme = "mererun"

    case preview(URL)
    case libraryImport(URL)

    static func parse(
        _ url: URL,
        fileManager: FileManager = .default
    ) throws -> MereRunDeepLink {
        guard url.scheme?.lowercased() == scheme else {
            throw MereRunDeepLinkError.unsupportedRoute
        }

        guard url.user == nil, url.password == nil, url.port == nil,
              url.fragment == nil else {
            throw MereRunDeepLinkError.unsupportedRoute
        }

        let route: Route
        switch (url.host?.lowercased(), url.path) {
        case ("preview", ""):
            route = .preview
        case ("library", "/import"):
            route = .libraryImport
        default:
            throw MereRunDeepLinkError.unsupportedRoute
        }

        let queryItems = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard !queryItems.isEmpty else {
            throw route.missingPathError
        }
        guard queryItems.count == 1, queryItems[0].name == route.parameterName else {
            throw MereRunDeepLinkError.unsupportedParameters
        }
        guard let path = queryItems[0].value, !path.isEmpty else {
            throw route.missingPathError
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

        switch route {
        case .preview: return .preview(fileURL)
        case .libraryImport: return .libraryImport(fileURL)
        }
    }

    private enum Route {
        case preview
        case libraryImport

        var parameterName: String {
            switch self {
            case .preview: return "path"
            case .libraryImport: return "receipt"
            }
        }

        var missingPathError: MereRunDeepLinkError {
            switch self {
            case .preview: return .missingPath
            case .libraryImport: return .missingReceipt
            }
        }
    }
}

enum MereRunDeepLinkError: LocalizedError, Equatable {
    case unsupportedRoute
    case missingPath
    case missingReceipt
    case unsupportedParameters
    case pathMustBeAbsolute
    case fileNotFound(String)
    case notAFile(String)
    case fileNotReadable(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedRoute:
            return "Use mererun://preview or mererun://library/import with one percent-encoded file path."
        case .missingPath:
            return "The preview link is missing its artifact path."
        case .missingReceipt:
            return "The Library import link is missing its receipt path."
        case .unsupportedParameters:
            return "The link has unsupported or duplicate parameters."
        case .pathMustBeAbsolute:
            return "The linked file path must be absolute."
        case .fileNotFound(let path):
            return "No linked file exists at \(path)."
        case .notAFile(let path):
            return "The linked target is not a file: \(path)."
        case .fileNotReadable(let path):
            return "The linked file is not readable: \(path)."
        }
    }
}
