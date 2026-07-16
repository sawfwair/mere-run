import Foundation
import MediaIO

struct FaceDetectorInput {
    let values: [Float]
    let scale: Double
}

enum FaceImageProcessing {
    static let detectorSize = 640
    static let recognizerSize = 112
    static let arcFaceTemplate = [
        FacePoint(x: 38.2946, y: 51.6963),
        FacePoint(x: 73.5318, y: 51.5014),
        FacePoint(x: 56.0252, y: 71.7366),
        FacePoint(x: 41.5493, y: 92.3655),
        FacePoint(x: 70.7299, y: 92.2041),
    ]

    static func detectorInput(from image: MediaImage) throws -> FaceDetectorInput {
        let scale = min(
            Double(detectorSize) / Double(image.width),
            Double(detectorSize) / Double(image.height)
        )
        let resizedWidth = max(1, Int((Double(image.width) * scale).rounded()))
        let resizedHeight = max(1, Int((Double(image.height) * scale).rounded()))
        let resized = try resizedBilinear(image, width: resizedWidth, height: resizedHeight)
        var canvas = try MediaImage(
            width: detectorSize,
            height: detectorSize,
            rgba8: [UInt8](repeating: 0, count: detectorSize * detectorSize * 4)
        )
        for pixel in 0..<(detectorSize * detectorSize) {
            canvas.rgba8[(pixel * 4) + 3] = 255
        }
        for y in 0..<resizedHeight {
            for x in 0..<resizedWidth {
                let source = ((y * resizedWidth) + x) * 4
                let destination = ((y * detectorSize) + x) * 4
                canvas.rgba8[destination..<(destination + 4)] = resized.rgba8[source..<(source + 4)]
            }
        }
        return FaceDetectorInput(values: normalizedCHW(canvas, scale: 128), scale: scale)
    }

    static func alignedFace(from image: MediaImage, landmarks: [FacePoint]) throws -> MediaImage {
        guard landmarks.count == arcFaceTemplate.count,
              let transform = similarityTransform(from: landmarks, to: arcFaceTemplate) else {
            throw FaceAnalysisError.invalidLandmarks
        }
        return try warpedBilinear(
            image,
            width: recognizerSize,
            height: recognizerSize,
            sourceToDestination: transform
        )
    }

    static func recognizerInput(from alignedFace: MediaImage) -> [Float] {
        normalizedCHW(alignedFace, scale: 127.5)
    }

    private static func normalizedCHW(_ image: MediaImage, scale: Float) -> [Float] {
        let pixelCount = image.width * image.height
        var values = [Float](repeating: 0, count: pixelCount * 3)
        for pixel in 0..<pixelCount {
            let source = pixel * 4
            values[pixel] = (Float(image.rgba8[source]) - 127.5) / scale
            values[pixel + pixelCount] = (Float(image.rgba8[source + 1]) - 127.5) / scale
            values[pixel + (2 * pixelCount)] = (Float(image.rgba8[source + 2]) - 127.5) / scale
        }
        return values
    }

    private static func resizedBilinear(_ image: MediaImage, width: Int, height: Int) throws -> MediaImage {
        var output = [UInt8](repeating: 0, count: width * height * 4)
        let xScale = Double(image.width) / Double(width)
        let yScale = Double(image.height) / Double(height)
        for y in 0..<height {
            let sourceY = min(
                Double(image.height - 1),
                max(0, ((Double(y) + 0.5) * yScale) - 0.5)
            )
            for x in 0..<width {
                let sourceX = min(
                    Double(image.width - 1),
                    max(0, ((Double(x) + 0.5) * xScale) - 0.5)
                )
                sampleBilinear(image, x: sourceX, y: sourceY, into: &output, at: ((y * width) + x) * 4)
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    private static func warpedBilinear(
        _ image: MediaImage,
        width: Int,
        height: Int,
        sourceToDestination transform: [Double]
    ) throws -> MediaImage {
        let a = transform[0]
        let b = transform[1]
        let tx = transform[2]
        let ty = transform[3]
        let determinant = (a * a) + (b * b)
        guard determinant > 1e-12 else { throw FaceAnalysisError.invalidLandmarks }
        var output = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let dx = Double(x) - tx
                let dy = Double(y) - ty
                let sourceX = ((a * dx) + (b * dy)) / determinant
                let sourceY = ((-b * dx) + (a * dy)) / determinant
                sampleBilinear(image, x: sourceX, y: sourceY, into: &output, at: ((y * width) + x) * 4)
            }
        }
        return try MediaImage(width: width, height: height, rgba8: output)
    }

    private static func sampleBilinear(
        _ image: MediaImage,
        x: Double,
        y: Double,
        into output: inout [UInt8],
        at destination: Int
    ) {
        guard x >= 0, y >= 0, x <= Double(image.width - 1), y <= Double(image.height - 1) else {
            output[destination + 3] = 255
            return
        }
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(image.width - 1, x0 + 1)
        let y1 = min(image.height - 1, y0 + 1)
        let fx = x - Double(x0)
        let fy = y - Double(y0)
        for channel in 0..<4 {
            let topLeft = Double(image.rgba8[((y0 * image.width) + x0) * 4 + channel])
            let topRight = Double(image.rgba8[((y0 * image.width) + x1) * 4 + channel])
            let bottomLeft = Double(image.rgba8[((y1 * image.width) + x0) * 4 + channel])
            let bottomRight = Double(image.rgba8[((y1 * image.width) + x1) * 4 + channel])
            let top = topLeft + ((topRight - topLeft) * fx)
            let bottom = bottomLeft + ((bottomRight - bottomLeft) * fx)
            output[destination + channel] = UInt8(clamping: Int((top + ((bottom - top) * fy)).rounded()))
        }
    }

    private static func similarityTransform(from source: [FacePoint], to destination: [FacePoint]) -> [Double]? {
        guard source.count == destination.count, source.count >= 2 else { return nil }
        var normal = Array(repeating: Array(repeating: 0.0, count: 4), count: 4)
        var target = Array(repeating: 0.0, count: 4)

        func accumulate(_ row: [Double], value: Double) {
            for i in 0..<4 {
                target[i] += row[i] * value
                for j in 0..<4 {
                    normal[i][j] += row[i] * row[j]
                }
            }
        }

        for index in source.indices {
            let x = source[index].x
            let y = source[index].y
            accumulate([x, -y, 1, 0], value: destination[index].x)
            accumulate([y, x, 0, 1], value: destination[index].y)
        }
        return solveLinearSystem(normal, target)
    }

    private static func solveLinearSystem(_ matrix: [[Double]], _ vector: [Double]) -> [Double]? {
        var augmented = zip(matrix, vector).map { $0 + [$1] }
        for column in 0..<4 {
            guard let pivot = (column..<4).max(by: {
                abs(augmented[$0][column]) < abs(augmented[$1][column])
            }), abs(augmented[pivot][column]) > 1e-12 else { return nil }
            augmented.swapAt(column, pivot)
            let divisor = augmented[column][column]
            for index in column...4 { augmented[column][index] /= divisor }
            for row in 0..<4 where row != column {
                let factor = augmented[row][column]
                for index in column...4 { augmented[row][index] -= factor * augmented[column][index] }
            }
        }
        return augmented.map { $0[4] }
    }
}
