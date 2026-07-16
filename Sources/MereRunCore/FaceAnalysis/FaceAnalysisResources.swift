import Foundation

public struct FaceAnalysisResources: Sendable {
    public static let modelID = "vision-face-buffalo-l"
    public static let detectorRelativePath = "buffalo_l/det_10g.onnx"
    public static let recognizerRelativePath = "buffalo_l/w600k_r50.onnx"
    public static let detectorByteCount: Int64 = 16_923_827
    public static let recognizerByteCount: Int64 = 174_383_860

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var detectorURL: URL {
        rootURL.appendingPathComponent(Self.detectorRelativePath)
    }

    public var recognizerURL: URL {
        rootURL.appendingPathComponent(Self.recognizerRelativePath)
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        [
            (detectorURL, Self.detectorByteCount),
            (recognizerURL, Self.recognizerByteCount),
        ].compactMap { url, expectedBytes in
            guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
                  let byteCount = (attributes[.size] as? NSNumber)?.int64Value,
                  byteCount == expectedBytes else {
                return url
            }
            return nil
        }
    }
}
