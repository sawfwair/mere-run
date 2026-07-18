import XCTest
import AudioCore
@testable import MereRunCLI

final class LiveASRCLITests: XCTestCase {
    func testPCMDecoderCarriesFragmentedOddByte() throws {
        var decoder = PCM16LittleEndianDecoder()
        XCTAssertTrue(decoder.decode(Data([0x00])).isEmpty)
        let samples = decoder.decode(Data([0x80, 0xFF, 0x7F]))
        XCTAssertEqual(samples.count, 2)
        XCTAssertEqual(samples[0], -1, accuracy: 0.000_01)
        XCTAssertEqual(samples[1], Float(Int16.max) / 32_768, accuracy: 0.000_01)
        XCTAssertNoThrow(try decoder.validateEOF())
    }

    func testPCMDecoderRejectsOddByteAtEOF() {
        var decoder = PCM16LittleEndianDecoder()
        _ = decoder.decode(Data([0x01]))
        XCTAssertThrowsError(try decoder.validateEOF()) { error in
            XCTAssertTrue(error.localizedDescription.contains("pcm_odd_byte_eof"))
        }
    }

    func testProtocolEventsEncodeOnlyDeclaredFields() throws {
        let encoder = JSONEncoder()
        let ready = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoder.encode(LiveASRProtocolEvent.ready())) as? [String: Any]
        )
        XCTAssertEqual(ready["protocol"] as? Int, 1)
        XCTAssertEqual(ready["type"] as? String, "ready")
        XCTAssertEqual(ready["sampleRate"] as? Int, 16_000)
        XCTAssertEqual(ready["inputFormat"] as? String, "pcm-s16le")
        XCTAssertNil(ready["text"])
    }
}
