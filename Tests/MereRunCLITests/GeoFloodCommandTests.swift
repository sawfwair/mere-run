import XCTest
@testable import MereRunCLI

final class GeoFloodCommandTests: XCTestCase {
    func testParsesNativeFloodOptions() throws {
        let command = try GeoFlood.parse([
            "/tmp/input.safetensors",
            "--output", "/tmp/logits.safetensors",
            "--model", "/tmp/terramind",
            "--preflight",
            "--json",
        ])

        XCTAssertEqual(command.input, "/tmp/input.safetensors")
        XCTAssertEqual(command.output, "/tmp/logits.safetensors")
        XCTAssertEqual(command.model, "/tmp/terramind")
        XCTAssertTrue(command.preflight)
        XCTAssertTrue(command.json)
    }

    func testGeoCommandRegistersFlood() {
        XCTAssertTrue(Geo.configuration.subcommands.contains { $0 == GeoFlood.self })
    }
}
