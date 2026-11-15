import Foundation
import XCTest
@testable import AudioSTT

final class SOVExpertRecipeTests: XCTestCase {
    func testTypedRecipeValidationAndRoundTrip() throws {
        let recipe = SOVExpertRecipe(layerBits: [4, 6, 8])
        let decoded = try JSONDecoder().decode(SOVExpertRecipe.self, from: JSONEncoder().encode(recipe))
        try decoded.validate(layerCount: 3)
        XCTAssertEqual(decoded.layerBits, [4, 6, 8])
        XCTAssertThrowsError(try decoded.validate(layerCount: 2))
        XCTAssertThrowsError(try SOVExpertRecipe(layerBits: [3]).validate(layerCount: 1))
        XCTAssertThrowsError(try SOVExpertRecipe(groupSize: 16, layerBits: [4]).validate(layerCount: 1))
        let unsupported = Data(#"{"schemaVersion":2,"groupSize":64,"layerBits":[4]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(SOVExpertRecipe.self, from: unsupported).validate(layerCount: 1))
    }
}
