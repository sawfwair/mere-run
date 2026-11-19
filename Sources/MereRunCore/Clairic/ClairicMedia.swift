import Foundation
#if !os(iOS)
import MediaIO
import MLX
import MereRunQwenModel
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#endif

enum ClairicMediaInput: Decodable, Sendable {
    struct Embedded: Decodable, Sendable {
        let content_type: String
        let base64: String
        let rotation: Int?
    }

    case pathOrDataURI(String), embedded(Embedded)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .pathOrDataURI(value)
        } else {
            self = .embedded(try container.decode(Embedded.self))
        }
    }
}

#if !os(iOS)
struct ClairicMedia {
    private static let maximumBytes = 4 * 1_024 * 1_024
    private static let mimeTypes: Set<String> = ["image/png", "image/jpeg", "image/webp", "image/gif"]

    struct Item {
        let pixels: [Float]
        let width: Int
        let height: Int

        var tokenCount: Int { width * height / 1_024 }

        func encode(using tower: Q35VisionTower) throws -> Q35VisionReplacement {
            let grid = (1, height / 16, width / 16)
            let embeddings = try tower.encodeImage(pixelValues: MLXArray(pixels, [1, 3, height, width]), gridTHW: grid)
            MLX.eval(embeddings)
            return Q35VisionReplacement(embeddings: embeddings, gridTHW: grid)
        }
    }

    static func prepare(_ request: ClairicRequest, processor: ClairicConfiguration.Processor) throws -> [Item] {
        var images: [MediaImage] = []
        var usedBytes = 0
        for input in request.images ?? [] {
            try Task.checkCancellation()
            let (data, rotation) = try load(input, mimeTypes: mimeTypes, usedBytes: &usedBytes)
            let image = try MediaImageIO.decode(data: data)
            let opaque = try MediaImageIO.compositeOnWhite(image)
            images.append(try rotate(opaque, degrees: rotation))
        }
        for input in request.documents ?? [] {
            try Task.checkCancellation()
            let (data, _) = try load(input, mimeTypes: ["application/pdf"], usedBytes: &usedBytes)
            images += try pdfImages(data, remaining: 4 - images.count)
        }
        return try images.map { image in
            try Task.checkCancellation()
            let size = try Q35Generator.qwen3VLTargetSize(
                originalWidth: image.width, originalHeight: image.height, patchSize: 16, spatialMergeSize: 2,
                minPixels: processor.size.shortest_edge, maxPixels: min(processor.size.longest_edge, 1_600_000))
            let resized = try MediaImageIO.bicubicResizedRGB(image, width: size.width, height: size.height)
            let pixels = MediaImageIO.rescaledRGBCHWFloat(resized, rescaleFactor: 1 / 255.0, normalizedToMinusOneToOne: true)
            return Item(pixels: pixels, width: size.width, height: size.height)
        }
    }

    private static func load(
        _ input: ClairicMediaInput, mimeTypes: Set<String>, usedBytes: inout Int
    ) throws -> (Data, Int) {
        let data: Data
        let rotation: Int
        switch input {
        case .embedded(let embedded):
            data = try decode(mime: embedded.content_type, base64: embedded.base64, allowed: mimeTypes)
            rotation = embedded.rotation ?? 0
        case .pathOrDataURI(let value) where value.hasPrefix("data:"):
            guard let comma = value.firstIndex(of: ","), value[..<comma].count > 12,
                  value[..<comma].hasSuffix(";base64") else {
                throw ClairicError.invalid("Clairic media data URIs must use base64.")
            }
            let mime = String(value[value.index(value.startIndex, offsetBy: 5)..<value.index(comma, offsetBy: -7)])
            data = try decode(mime: mime, base64: String(value[value.index(after: comma)...]), allowed: mimeTypes)
            rotation = 0
        case .pathOrDataURI(let path):
            guard URL(string: path)?.scheme == nil else {
                throw ClairicError.invalid("Clairic media requires a local path or embedded data.")
            }
            let url = URL(fileURLWithPath: path)
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard values.isRegularFile == true, let size = values.fileSize, size <= maximumBytes else {
                throw ClairicError.invalid("Clairic media must be a regular file of at most 4 MiB.")
            }
            data = try Data(contentsOf: url)
            rotation = 0
        }
        usedBytes += data.count
        guard !data.isEmpty, data.count <= maximumBytes, usedBytes <= 2 * maximumBytes else {
            throw ClairicError.invalid("Clairic media is limited to 4 MiB per input and 8 MiB total.")
        }
        guard [0, 90, 180, 270].contains(rotation), rotation == 0 || !mimeTypes.contains("application/pdf") else {
            throw ClairicError.invalid("Image rotation must be 0, 90, 180, or 270 degrees.")
        }
        return (data, rotation)
    }

    private static func decode(mime: String, base64: String, allowed: Set<String>) throws -> Data {
        guard allowed.contains(mime.lowercased()), base64.utf8.count <= 4 * ((maximumBytes + 2) / 3),
              let data = Data(base64Encoded: base64) else {
            throw ClairicError.invalid("Clairic received unsupported or invalid embedded media.")
        }
        return data
    }

    private static func rotate(_ image: MediaImage, degrees: Int) throws -> MediaImage {
        guard degrees != 0 else { return image }
        let sideways = degrees == 90 || degrees == 270
        let width = sideways ? image.height : image.width
        let height = sideways ? image.width : image.height
        var pixels = [UInt8](repeating: 0, count: image.rgba8.count)
        for y in 0..<image.height {
            for x in 0..<image.width {
                let point = switch degrees {
                case 90: (image.height - 1 - y, x)
                case 180: (image.width - 1 - x, image.height - 1 - y)
                default: (y, image.width - 1 - x)
                }
                let source = (y * image.width + x) * 4
                let target = (point.1 * width + point.0) * 4
                for channel in 0..<4 { pixels[target + channel] = image.rgba8[source + channel] }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: pixels)
    }

    private static func pdfImages(_ data: Data, remaining: Int) throws -> [MediaImage] {
#if canImport(CoreGraphics)
        guard let provider = CGDataProvider(data: data as CFData), let pdf = CGPDFDocument(provider), !pdf.isEncrypted else {
            throw ClairicError.invalid("Clairic cannot open this PDF or it is encrypted.")
        }
        guard pdf.numberOfPages > 0, pdf.numberOfPages <= remaining else {
            throw ClairicError.invalid("Clairic accepts at most four images or PDF pages per request.")
        }
        return try (1...pdf.numberOfPages).map { index in
            try Task.checkCancellation()
            guard let page = pdf.page(at: index) else { throw ClairicError.invalid("Clairic could not read PDF page \(index).") }
            return try rasterize(page)
        }
#else
        throw ClairicError.unsupported("Clairic PDF rendering requires macOS.")
#endif
    }

#if canImport(CoreGraphics)
    private static func rasterize(_ page: CGPDFPage) throws -> MediaImage {
        let bounds = page.getBoxRect(.cropBox)
        let area = bounds.width * bounds.height
        guard area.isFinite, bounds.width > 0, bounds.height > 0,
              max(bounds.width, bounds.height) / min(bounds.width, bounds.height) <= 200 else {
            throw ClairicError.invalid("Clairic received invalid PDF page dimensions.")
        }
        let sideways = page.rotationAngle % 180 != 0
        let scale = min(2, sqrt(4_000_000 / area))
        let width = max(1, Int(ceil((sideways ? bounds.height : bounds.width) * scale)))
        let height = max(1, Int(ceil((sideways ? bounds.width : bounds.height) * scale)))
        let space = CGColorSpaceCreateDeviceRGB()
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        let rendered = bytes.withUnsafeMutableBytes { buffer -> Bool in
            let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space, bitmapInfo: bitmapInfo) else { return false }
            let rect = CGRect(x: 0, y: 0, width: width, height: height)
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(rect)
            context.concatenate(page.getDrawingTransform(.cropBox, rect: rect, rotate: 0, preserveAspectRatio: true))
            context.drawPDFPage(page)
            return true
        }
        guard rendered else { throw ClairicError.inference("Clairic could not rasterize a PDF page.") }
        return try MediaImage(width: width, height: height, rgba8: bytes)
    }
#endif
}
#endif
