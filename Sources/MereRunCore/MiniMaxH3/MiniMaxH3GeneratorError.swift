import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

@inline(__always)
func withMiniMaxH3AutoreleasePool<T>(_ body: () throws -> T) rethrows -> T {
    #if canImport(ObjectiveC)
    return try autoreleasepool(invoking: body)
    #else
    return try body()
    #endif
}

public enum MiniMaxH3GeneratorError: LocalizedError {
    case missingModelFiles([URL])
    case invalidOptions(String)
    case imageDecodeFailed(URL)
    case mediaDecodeFailed(URL, String)

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let files):
            return "MiniMax-H3 model root is missing: \(files.map(\.lastPathComponent).joined(separator: ", "))"
        case .invalidOptions(let reason): return "Invalid MiniMax-H3 request: \(reason)"
        case .imageDecodeFailed(let url): return "MiniMax-H3 could not decode image: \(url.path)"
        case .mediaDecodeFailed(let url, let reason):
            return "MiniMax-H3 could not decode reference \(url.path): \(reason)"
        }
    }
}
