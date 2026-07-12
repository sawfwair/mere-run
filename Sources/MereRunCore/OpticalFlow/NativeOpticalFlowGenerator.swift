import Foundation

#if canImport(CoreGraphics) && canImport(CoreVideo) && canImport(ImageIO) && canImport(Vision)
import CoreGraphics
import CoreVideo
import ImageIO
import Vision
#endif

public enum NativeOpticalFlowError: Error, Equatable, LocalizedError, Sendable {
    case imageNotFound(String)
    case imageDecodeFailed(String)
    case dimensionMismatch(fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int)
    case noObservation
    case unsupportedPixelFormat(UInt32)
    case unavailablePixelData
    case unsupportedPlatform

    public var errorDescription: String? {
        switch self {
        case .imageNotFound(let path): "Image not found: \(path)"
        case .imageDecodeFailed(let path): "Could not decode image: \(path)"
        case .dimensionMismatch(let fromWidth, let fromHeight, let toWidth, let toHeight):
            "Optical-flow images must have equal dimensions; received \(fromWidth)x\(fromHeight) and \(toWidth)x\(toHeight)."
        case .noObservation: "Native optical flow produced no pixel-buffer observation."
        case .unsupportedPixelFormat(let format): "Native optical flow returned unsupported pixel format \(format)."
        case .unavailablePixelData: "Native optical-flow pixel data was unavailable."
        case .unsupportedPlatform: "Native optical flow requires an Apple platform with the Vision framework."
        }
    }
}

public enum NativeOpticalFlowAccuracy: String, Codable, CaseIterable, Sendable {
    case low
    case medium
    case high
    case veryHigh = "very-high"
}

public struct NativeOpticalFlowResult: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let fromPath: String
    public let toPath: String
    public let outputPath: String
    public let format: String
    public let width: Int
    public let height: Int
    public let vectorCount: Int
    public let accuracy: NativeOpticalFlowAccuracy
    public let meanMagnitude: Double
    public let maximumMagnitude: Double

    public init(
        schemaVersion: Int = 1,
        fromPath: String,
        toPath: String,
        outputPath: String,
        format: String = "middlebury-flo",
        width: Int,
        height: Int,
        vectorCount: Int,
        accuracy: NativeOpticalFlowAccuracy,
        meanMagnitude: Double,
        maximumMagnitude: Double
    ) {
        self.schemaVersion = schemaVersion
        self.fromPath = fromPath
        self.toPath = toPath
        self.outputPath = outputPath
        self.format = format
        self.width = width
        self.height = height
        self.vectorCount = vectorCount
        self.accuracy = accuracy
        self.meanMagnitude = meanMagnitude
        self.maximumMagnitude = maximumMagnitude
    }
}

public struct NativeOpticalFlowGenerator: Sendable {
    public init() {}

    public func generate(
        from fromURL: URL,
        to toURL: URL,
        outputURL: URL,
        accuracy: NativeOpticalFlowAccuracy = .high
    ) throws -> NativeOpticalFlowResult {
        let standardizedFrom = fromURL.standardizedFileURL
        let standardizedTo = toURL.standardizedFileURL
        guard FileManager.default.fileExists(atPath: standardizedFrom.path) else {
            throw NativeOpticalFlowError.imageNotFound(standardizedFrom.path)
        }
        guard FileManager.default.fileExists(atPath: standardizedTo.path) else {
            throw NativeOpticalFlowError.imageNotFound(standardizedTo.path)
        }

        #if canImport(CoreGraphics) && canImport(CoreVideo) && canImport(ImageIO) && canImport(Vision)
        let fromImage = try Self.loadImage(standardizedFrom)
        let toImage = try Self.loadImage(standardizedTo)
        guard fromImage.width == toImage.width, fromImage.height == toImage.height else {
            throw NativeOpticalFlowError.dimensionMismatch(
                fromWidth: fromImage.width,
                fromHeight: fromImage.height,
                toWidth: toImage.width,
                toHeight: toImage.height
            )
        }

        let request = VNGenerateOpticalFlowRequest(targetedCGImage: toImage, options: [:])
        request.computationAccuracy = Self.visionAccuracy(accuracy)
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        let handler = VNImageRequestHandler(cgImage: fromImage, options: [:])
        try handler.perform([request])
        guard let observation = request.results?.first else {
            throw NativeOpticalFlowError.noObservation
        }
        let pixelBuffer = observation.pixelBuffer
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard pixelFormat == kCVPixelFormatType_TwoComponent32Float else {
            throw NativeOpticalFlowError.unsupportedPixelFormat(pixelFormat)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let vectors = try Self.readVectors(pixelBuffer, width: width, height: height)
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.writeMiddleburyFlow(vectors, width: width, height: height, to: outputURL)
        var magnitudeSum = 0.0
        var maximumMagnitude = 0.0
        for index in stride(from: 0, to: vectors.count, by: 2) {
            let magnitude = hypot(Double(vectors[index]), Double(vectors[index + 1]))
            magnitudeSum += magnitude
            maximumMagnitude = max(maximumMagnitude, magnitude)
        }
        let count = width * height
        return NativeOpticalFlowResult(
            fromPath: standardizedFrom.path,
            toPath: standardizedTo.path,
            outputPath: outputURL.standardizedFileURL.path,
            width: width,
            height: height,
            vectorCount: count,
            accuracy: accuracy,
            meanMagnitude: count > 0 ? magnitudeSum / Double(count) : 0,
            maximumMagnitude: maximumMagnitude
        )
        #else
        throw NativeOpticalFlowError.unsupportedPlatform
        #endif
    }

    #if canImport(CoreGraphics) && canImport(CoreVideo) && canImport(ImageIO) && canImport(Vision)
    private static func loadImage(_ url: URL) throws -> CGImage {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw NativeOpticalFlowError.imageDecodeFailed(url.path)
        }
        return image
    }

    private static func visionAccuracy(
        _ accuracy: NativeOpticalFlowAccuracy
    ) -> VNGenerateOpticalFlowRequest.ComputationAccuracy {
        switch accuracy {
        case .low: .low
        case .medium: .medium
        case .high: .high
        case .veryHigh: .veryHigh
        }
    }

    private static func readVectors(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) throws -> [Float] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NativeOpticalFlowError.unavailablePixelData
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var vectors = [Float](repeating: 0, count: width * height * 2)
        for y in 0..<height {
            let row = baseAddress.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float.self)
            let destination = y * width * 2
            for x in 0..<width {
                vectors[destination + x * 2] = row[x * 2]
                vectors[destination + x * 2 + 1] = row[x * 2 + 1]
            }
        }
        return vectors
    }

    private static func writeMiddleburyFlow(_ vectors: [Float], width: Int, height: Int, to url: URL) throws {
        var data = Data()
        appendFloat(202_021.25, to: &data)
        appendInt32(Int32(width), to: &data)
        appendInt32(Int32(height), to: &data)
        for value in vectors {
            appendFloat(value, to: &data)
        }
        try data.write(to: url, options: .atomic)
    }

    private static func appendInt32(_ value: Int32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func appendFloat(_ value: Float, to data: inout Data) {
        var littleEndian = value.bitPattern.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
    #endif
}
