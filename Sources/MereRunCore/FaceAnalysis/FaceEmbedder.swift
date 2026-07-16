import Foundation
import MediaIO

struct FaceEmbedder {
    let session: FaceONNXSession

    init(modelURL: URL, executionProvider: FaceExecutionProvider) throws {
        session = try FaceONNXSession(modelURL: modelURL, executionProvider: executionProvider)
    }

    func embed(image: MediaImage, landmarks: [FacePoint]) throws -> [Float] {
        let aligned = try FaceImageProcessing.alignedFace(from: image, landmarks: landmarks)
        let outputs = try session.run(
            input: FaceImageProcessing.recognizerInput(from: aligned),
            shape: [1, 3, 112, 112],
            inputName: "input.1",
            outputNames: ["683"]
        )
        guard let values = outputs.first, values.count == 512 else {
            throw FaceAnalysisError.invalidModelOutput(
                "recognizer returned \(outputs.first?.count ?? 0) values; expected 512"
            )
        }
        return FaceAnalysisMath.l2Normalized(values)
    }
}
