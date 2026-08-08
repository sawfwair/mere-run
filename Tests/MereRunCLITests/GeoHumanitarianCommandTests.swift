import XCTest
@testable import MereRunCLI

final class GeoHumanitarianCommandTests: XCTestCase {
    func testParsesTerraMindFireOptions() throws {
        let command = try GeoFire.parse([
            "/tmp/input.safetensors",
            "--output", "/tmp/fire.safetensors",
            "--model", "vision-fire-terramind-base",
            "--preflight",
            "--json",
        ])

        XCTAssertEqual(command.model, "vision-fire-terramind-base")
        XCTAssertTrue(command.preflight)
        XCTAssertTrue(command.json)
    }

    func testParsesTESSERATierAndMatryoshkaDimensions() throws {
        let command = try GeoTESSERA.parse([
            "/tmp/input.safetensors",
            "--output", "/tmp/embedding.safetensors",
            "--model", "vision-embed-tessera-v2-large",
            "--dimensions", "64",
        ])

        XCTAssertEqual(command.model, "vision-embed-tessera-v2-large")
        XCTAssertEqual(command.dimensions, 64)
    }

    func testParsesOlmoEarthSpatialControls() throws {
        let command = try GeoOlmoEarth.parse([
            "/tmp/input.safetensors",
            "--output", "/tmp/embedding.safetensors",
            "--model", "vision-embed-olmoearth-v12-base",
            "--patch-size", "8",
            "--input-resolution", "30",
            "--include-tokens",
        ])

        XCTAssertEqual(command.model, "vision-embed-olmoearth-v12-base")
        XCTAssertEqual(command.patchSize, 8)
        XCTAssertEqual(command.inputResolution, 30)
        XCTAssertTrue(command.includeTokens)
    }

    func testGeoRegistersEveryNativeHumanitarianRoute() {
        XCTAssertTrue(Geo.configuration.subcommands.contains { $0 == GeoFire.self })
        XCTAssertTrue(Geo.configuration.subcommands.contains { $0 == GeoTESSERA.self })
        XCTAssertTrue(Geo.configuration.subcommands.contains { $0 == GeoOlmoEarth.self })
    }
}
