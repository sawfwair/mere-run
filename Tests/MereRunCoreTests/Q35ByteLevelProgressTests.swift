import XCTest
@testable import MereRunCore

final class Q35ByteLevelProgressTests: XCTestCase {
    func testEveryPrefixMatchesByteLevelDecode() {
        let bytes: [[UInt8]] = [
            Array("hello ".utf8), [0xf0,0x9f], [0xa7], [0xa0], Array(" e".utf8),
            [0xcc], [0x81], [0xef,0xbf,0xbd], [0xc2], Array("<think>".utf8), [0x9f], [], [10]
        ]
        func decode(_ tokens: [Int]) -> String {
            String(decoding: tokens.flatMap { bytes[$0] }, as: UTF8.self)
        }
        var progress = Q35ByteLevelProgress(), counts: [Int] = []
        for id in bytes.indices {
            progress.append(token: id, piece: decode([id])) { suffix in
                counts.append(suffix.count)
                return decode(suffix)
            }
            XCTAssertEqual(Array(progress.text.utf8), Array(decode(Array(0...id)).utf8))
        }
        XCTAssertLessThan(counts.max()!, bytes.count)
    }
}
