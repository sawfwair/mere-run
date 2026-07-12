import MLX
import MereRunCore
import XCTest

final class MoGe2ModelTests: MereRunCoreTestCase {
    func testPinnedONNXMapsEveryParameterAndRunsNativeRawForward() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_MOGE_ONNX"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path), "Set MERERUN_TEST_MOGE_ONNX to the pinned model.onnx fixture.")
        let model = MoGe2Model()
        try MoGe2ONNXWeights.load(
            model: model,
            archive: ONNXInitializerArchive(url: URL(fileURLWithPath: path))
        )
        let image = MLX.ones([1, 28, 28, 3], dtype: .float32) * MLXArray(Float(0.5))
        let output = try model(image, tokenCount: 4)
        MLX.eval(output.points, output.normals, output.maskProbability, output.metricScale)
        XCTAssertEqual(output.points.shape, [1, 28, 28, 3])
        XCTAssertEqual(output.normals.shape, [1, 28, 28, 3])
        XCTAssertEqual(output.maskProbability.shape, [1, 28, 28])
        XCTAssertEqual(output.metricScale.shape, [1])
        XCTAssertTrue(output.points.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(output.normals.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(output.maskProbability.asArray(Float.self).allSatisfy { (0...1).contains($0) })
        XCTAssertGreaterThan(output.metricScale.item(Float.self), 0)

        if let parityRoot = ProcessInfo.processInfo.environment["MERERUN_TEST_MOGE_PARITY"], !parityRoot.isEmpty {
            try assertParity(output.points.asArray(Float.self), file: "points.f32", root: parityRoot)
            try assertParity(output.normals.asArray(Float.self), file: "normal.f32", root: parityRoot)
            try assertParity(output.maskProbability.asArray(Float.self), file: "mask.f32", root: parityRoot)
            try assertParity(output.metricScale.asArray(Float.self), file: "scale.f32", root: parityRoot)
        }
    }

    private func assertParity(_ actual: [Float], file: String, root: String) throws {
        let expected = try readFloat32(URL(fileURLWithPath: root).appendingPathComponent(file))
        XCTAssertEqual(actual.count, expected.count, "\(file) element count")
        guard actual.count == expected.count else { return }
        let differences = zip(actual, expected).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(max(1, differences.count))
        let maximum = differences.max() ?? 0
        XCTAssertLessThanOrEqual(mean, 5e-4, "\(file) mean absolute error \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, 5e-3, "\(file) mean absolute error \(mean), max \(maximum)")
    }

    private func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        precondition(data.count.isMultiple(of: 4))
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }
}
