import XCTest
@testable import MereRunCore
import AudioCodecs

final class PCMStreamConverterTests: XCTestCase {
    func testDownmixInterleavedStereoToMono() {
        let stereo: [Float] = [0.2, -0.2, 0.6, 0.2, -0.4, -0.6]
        let mono = PCMStreamConverter.downmixInterleavedToMono(stereo, channelCount: 2)
        XCTAssertEqual(mono.count, 3)
        XCTAssertEqual(mono[0], 0.0, accuracy: 0.0001)
        XCTAssertEqual(mono[1], 0.4, accuracy: 0.0001)
        XCTAssertEqual(mono[2], -0.5, accuracy: 0.0001)
    }

    func testResampleLinearIdentityWhenRatesMatch() {
        let input: [Float] = [0.1, 0.3, 0.5, 0.7]
        let output = PCMStreamConverter.resampleLinear(input, from: 24_000, to: 24_000)
        XCTAssertEqual(output, input)
    }

    func testConvertInterleavedToMonoAndResample() {
        let stereo: [Float] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        let converted = PCMStreamConverter.convertInterleavedToMono(
            stereo,
            sourceSampleRate: 4,
            sourceChannelCount: 2,
            targetSampleRate: 2
        )
        XCTAssertEqual(converted.count, 2)
        XCTAssertEqual(converted[0], 0.5, accuracy: 0.0001)
        XCTAssertEqual(converted[1], 0.5, accuracy: 0.0001)
    }
}
