import Foundation
import MediaIO

struct FaceDetector {
    private static let strides = [8, 16, 32]
    private static let anchorsPerCell = 2
    private static let outputNames = ["448", "471", "494", "451", "474", "497", "454", "477", "500"]

    let session: FaceONNXSession

    init(modelURL: URL, executionProvider: FaceExecutionProvider) throws {
        session = try FaceONNXSession(modelURL: modelURL, executionProvider: executionProvider)
    }

    func detect(
        image: MediaImage,
        scoreThreshold: Float,
        nmsThreshold: Double,
        maxFaces: Int?
    ) throws -> [FaceDetection] {
        let input = try FaceImageProcessing.detectorInput(from: image)
        let outputs = try session.run(
            input: input.values,
            shape: [1, 3, 640, 640],
            inputName: "input.1",
            outputNames: Self.outputNames
        )
        guard outputs.count == Self.outputNames.count else {
            throw FaceAnalysisError.invalidModelOutput("detector returned \(outputs.count) tensors")
        }

        var candidates: [FaceDetection] = []
        for level in Self.strides.indices {
            let stride = Self.strides[level]
            let scores = outputs[level]
            let boxes = outputs[level + 3]
            let landmarks = outputs[level + 6]
            let grid = 640 / stride
            let anchorCount = grid * grid * Self.anchorsPerCell
            guard scores.count == anchorCount,
                  boxes.count == anchorCount * 4,
                  landmarks.count == anchorCount * 10 else {
                throw FaceAnalysisError.invalidModelOutput(
                    "unexpected detector shapes at stride \(stride): scores=\(scores.count) boxes=\(boxes.count) landmarks=\(landmarks.count)"
                )
            }

            for anchor in 0..<anchorCount where scores[anchor] >= scoreThreshold {
                let cell = anchor / Self.anchorsPerCell
                let centerX = Float(cell % grid) * Float(stride)
                let centerY = Float(cell / grid) * Float(stride)
                let boxOffset = anchor * 4
                let x1 = Double(centerX - boxes[boxOffset] * Float(stride)) / input.scale
                let y1 = Double(centerY - boxes[boxOffset + 1] * Float(stride)) / input.scale
                let x2 = Double(centerX + boxes[boxOffset + 2] * Float(stride)) / input.scale
                let y2 = Double(centerY + boxes[boxOffset + 3] * Float(stride)) / input.scale
                let clippedX1 = min(max(0, x1), Double(image.width))
                let clippedY1 = min(max(0, y1), Double(image.height))
                let clippedX2 = min(max(0, x2), Double(image.width))
                let clippedY2 = min(max(0, y2), Double(image.height))

                var points: [FacePoint] = []
                let landmarkOffset = anchor * 10
                for point in 0..<5 {
                    points.append(FacePoint(
                        x: Double(centerX + landmarks[landmarkOffset + (point * 2)] * Float(stride)) / input.scale,
                        y: Double(centerY + landmarks[landmarkOffset + (point * 2) + 1] * Float(stride)) / input.scale
                    ))
                }
                candidates.append(FaceDetection(
                    score: Double(scores[anchor]),
                    boundingBox: FaceBoundingBox(
                        x: clippedX1,
                        y: clippedY1,
                        width: max(0, clippedX2 - clippedX1),
                        height: max(0, clippedY2 - clippedY1)
                    ),
                    landmarks: points
                ))
            }
        }

        let selected = nonMaximumSuppression(candidates, threshold: nmsThreshold)
        if let maxFaces { return Array(selected.prefix(max(0, maxFaces))) }
        return selected
    }

    private func nonMaximumSuppression(_ detections: [FaceDetection], threshold: Double) -> [FaceDetection] {
        var remaining = detections.sorted { $0.score > $1.score }
        var selected: [FaceDetection] = []
        while let best = remaining.first {
            selected.append(best)
            remaining.removeFirst()
            remaining.removeAll { intersectionOverUnion(best.boundingBox, $0.boundingBox) > threshold }
        }
        return selected
    }

    private func intersectionOverUnion(_ lhs: FaceBoundingBox, _ rhs: FaceBoundingBox) -> Double {
        let x1 = max(lhs.x, rhs.x)
        let y1 = max(lhs.y, rhs.y)
        let x2 = min(lhs.x + lhs.width, rhs.x + rhs.width)
        let y2 = min(lhs.y + lhs.height, rhs.y + rhs.height)
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let union = lhs.area + rhs.area - intersection
        return union > 0 ? intersection / union : 0
    }
}
