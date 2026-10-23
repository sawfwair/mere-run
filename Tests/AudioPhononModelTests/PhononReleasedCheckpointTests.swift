import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioPhononModel

final class PhononReleasedCheckpointTests: MLXTestCase {
    /// This oracle was recorded from the pinned upstream F32 CPU implementation,
    /// with identical input noise. It catches graph errors without comparing RNGs.
    func testReleasedCheckpointAgainstUpstreamSteps() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_PHONON_CHECKPOINT"] else {
            throw XCTSkip("Set MERERUN_PHONON_CHECKPOINT to the pinned original checkpoint directory.")
        }
        struct Step: Decodable { let latent: [Float]; let eos: Float }
        struct Case: Decodable { let text: String; let steps: [Step] }
        struct Oracle: Decodable { let noise: [Float]; let cases: [Case] }
        enum Stop: Error { case traced }
        let fixture = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures/released-step-oracle.json")
        let oracle = try JSONDecoder().decode(Oracle.self, from: Data(contentsOf: fixture))
        let directory = URL(fileURLWithPath: path)
        let model = try PhononModel(directory: directory)
        let voice = try model.loadVoice(url: directory.appendingPathComponent("default-voice.safetensors"))
        for item in oracle.cases {
            var cursor = 0, step = 0
            do {
                try model.generate(text: item.text, voice: voice, noiseSampler: {
                    let values = Array(oracle.noise[cursor..<(cursor + 32)])
                    cursor += 32
                    return MLXArray(values, [1, 32])
                }, onStep: { latent, eos in
                    let target = item.steps[step]
                    let actual = latent.asArray(Float.self)
                    XCTAssertEqual(actual.count, target.latent.count)
                    // The first steps isolate kernel parity. Later autoregressive steps
                    // include feedback from previous floating-point differences.
                    let tolerance: Float = step < 3 ? 2e-5 : 1e-3
                    for (value, expected) in zip(actual, target.latent) {
                        XCTAssertEqual(value, expected, accuracy: tolerance, "\(item.text), step \(step)")
                    }
                    XCTAssertEqual(eos.item(Float.self), target.eos, accuracy: 2e-4)
                    step += 1
                }, onFrame: { frame in
                    XCTAssertEqual(frame.count, 1920)
                    XCTAssertTrue(frame.allSatisfy(\.isFinite))
                    if step == item.steps.count { throw Stop.traced }
                })
                XCTFail("Expected the bounded trace to stop generation")
            } catch Stop.traced { /* Only the first six steps are part of this oracle. */ }
            XCTAssertEqual(step, item.steps.count)
        }
    }
}
