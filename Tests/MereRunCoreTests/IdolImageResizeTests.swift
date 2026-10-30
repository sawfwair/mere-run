import Foundation
import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class IdolImageResizeTests: MereRunCoreTestCase {
    func testLanczosRGBAAndRGBMatchPillowBytes() throws {
        let fixture = try arrays()
        for label in ["rgba", "rgb"] {
            let input = fixture[label + "_input"]!
            let image = try MediaImage(width: input.dim(1), height: input.dim(0), rgba8: input.asArray(UInt8.self))
            for (width, height) in [(17, 11), (5, 7), (5, 19)] {
                let output = try IdolImageResize.resize(image, width: width, height: height)
                XCTAssertEqual(output.rgba8, fixture["\(label)_\(width)_\(height)"]!.asArray(UInt8.self))
            }
        }
    }
    func testOpaquePreprocessingInvokesMaskAfterShortSideResizeAndMatchesPillowFraming() throws {
        let fixture = try arrays(), input = fixture["framing_input"]!, mask = fixture["framing_mask"]!
        let image = try MediaImage(width: input.dim(1), height: input.dim(0), rgba8: input.asArray(UInt8.self))
        for mode in [IdolInputMode.synthetic, .real] {
            var calls = 0
            let result = try IdolPreprocessor.prepare(image, size: 64, mode: mode) { source in
                calls += 1
                XCTAssertEqual(source.width, mask.dim(1)); XCTAssertEqual(source.height, mask.dim(0))
                var bytes = source.rgba8
                let alpha = mask.asArray(UInt8.self)
                for index in alpha.indices { bytes[index * 4 + 3] = alpha[index] }
                return try MediaImage(width: source.width, height: source.height, rgba8: bytes)
            }
            XCTAssertEqual(calls, 1)
            XCTAssertEqual(result.rgba8, fixture["framing_" + mode.rawValue]!.asArray(UInt8.self))
        }
    }
    private func arrays() throws -> [String: MLXArray] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try MLX.loadArrays(url: root.appending(path: "ImageRuntimeTests/Fixtures/Idol/preprocess.safetensors"))
    }
}
