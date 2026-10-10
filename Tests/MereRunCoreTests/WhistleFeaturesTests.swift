import Foundation
import AudioCore
import MLX
import XCTest
@testable import AudioSTT

final class WhistleFeaturesTests: MereRunCoreTestCase {
    func testCrossAttentionDTWTracksMonotonicAcousticPeaks() {
        let attention: [[Float]] = [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
        XCTAssertEqual(WhistleAlignment.frames(attention: attention, frames: 6), [0, 2, 4])
        XCTAssertEqual(WhistleAlignment.frames(attention: [], frames: 6), [])
    }

    func testWhistleControlsPersistAndOldRequestsDecode() throws {
        let old = Data(#"{"audioURL":"file:///tmp/audio.wav","task":"transcribe","maxTokens":448}"#.utf8)
        XCTAssertNil(try JSONDecoder().decode(ASRRequest.self, from: old).whistle)
        let options = WhistleOptions(weights: .fp32, beamSize: 3, decoderDepth: 4, keywords: ["Mere Run"], wordTimestamps: false)
        let request = ASRRequest(audioURL: URL(fileURLWithPath: "/tmp/audio.wav"), whistle: options)
        XCTAssertEqual(try JSONDecoder().decode(ASRRequest.self, from: JSONEncoder().encode(request)), request)
        XCTAssertThrowsError(try WhistleOptions(beamSize: 9).validate())
    }

    func testFrontendMatchesScalarDFTIncludingQuietMelFloor() throws {
        let samples = (0..<960).map { index in
            Float(sin(Double(index * index) * 0.0007) * (0.02 + Double(index) * 0.0001))
        }
        var filter = [Float](repeating: 0, count: 257 * 80)
        for channel in 0..<80 { filter[(channel + 3) * 80 + channel] = 1e-6 }
        let actual = try WhistleFeatures(filterbank: filter).extract(samples).asArray(Float.self)
        let levels = stride(from: 0, to: 960, by: 320).map { start in
            sqrt(samples[start..<(start + 320)].reduce(Float(0)) { $0 + $1 * $1 } / 320)
        }.sorted()
        let gain = 0.1 / levels[1]
        var expected = [Float](repeating: 0, count: 6 * 80)
        for frame in 0..<6 {
            for channel in 0..<80 {
                var real = 0.0
                var imaginary = 0.0
                for index in 0..<400 {
                    let source = frame * 160 - 200 + index
                    guard samples.indices.contains(source) else { continue }
                    let window = 0.5 - 0.5 * cos(2 * Double.pi * Double(index) / 399)
                    let value = Double(samples[source] * gain) * window
                    let angle = -2 * Double.pi * Double((channel + 3) * index) / 512
                    real += value * cos(angle)
                    imaginary += value * sin(angle)
                }
                expected[frame * 80 + channel] = Float(log((real * real + imaginary * imaginary) * 1e-6 + pow(2, -24)))
            }
        }
        for channel in 0..<80 {
            let mean = (0..<6).reduce(Float(0)) { $0 + expected[$1 * 80 + channel] } / 6
            let variance = (0..<6).reduce(Float(0)) { sum, frame in
                let delta = expected[frame * 80 + channel] - mean
                return sum + delta * delta
            } / 5
            for frame in 0..<6 { expected[frame * 80 + channel] = (expected[frame * 80 + channel] - mean) / (sqrt(variance) + 1e-5) }
        }
        XCTAssertLessThan(zip(actual, expected).map { abs($0 - $1) }.max()!, 0.002)
    }

    func testWindowMergeRemovesCommonWordsAndHandlesEmptyText() {
        XCTAssertEqual(WhistleGenerator.merge("Hello from the kitchen", "the kitchen lights are on"), "Hello from the kitchen lights are on")
        XCTAssertEqual(WhistleGenerator.merge("", "Hello"), "Hello")
        XCTAssertEqual(WhistleGenerator.merge("Hello", ""), "Hello")
    }
}
