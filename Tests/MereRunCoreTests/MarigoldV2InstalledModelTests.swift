import Foundation
import XCTest
@testable import MereRunCore

final class MarigoldV2InstalledModelTests: XCTestCase {
    func testInstalledDepthExamplesWhenRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_MARIGOLD_ROOT"],
              let examples = environment["MERERUN_TEST_MARIGOLD_EXAMPLES"],
              let output = environment["MERERUN_TEST_MARIGOLD_EXAMPLES_OUTPUT"] else {
            throw XCTSkip("Set the Marigold root, upstream examples directory, and output directory.")
        }
        let generator = MarigoldV2Generator()
        for name in ["church.jpg", "dogs.png", "squirrel.jpg", "train.jpg"] {
            let imageURL = URL(fileURLWithPath: examples).appendingPathComponent(name)
            let directory = URL(fileURLWithPath: output).appendingPathComponent(imageURL.deletingPathExtension().lastPathComponent)
            let result = try await generator.generate(
                imageURL: imageURL, outputDirectory: directory, model: root,
                progress: { FileHandle.standardError.write(Data("marigold_example \($0)\n".utf8)) }
            )
            XCTAssertEqual(result.adapterPairCount, 723)
            XCTAssertEqual(result.vaeDecoderTensorCount, 108)
            XCTAssertEqual(max(result.inferenceWidth, result.inferenceHeight), 1_024)
            FileHandle.standardError.write(Data(
                "marigold_example \(name) load=\(result.modelLoadSeconds) inference=\(result.inferenceSeconds)\n".utf8
            ))
        }
        await generator.unload()
    }
}
