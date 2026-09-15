import ArgumentParser
import XCTest
@testable import MereRunCLI

final class AudioExportFadeTests: XCTestCase {
    func testCLIRejectsInvalidExportSettingsBeforeOpeningModel() async throws {
        for flag in ["--fade-in-ms", "--fade-out-ms", "--target-peak-db"] {
            let values = flag == "--target-peak-db" ? ["nan", "inf", "-inf"] : ["nan", "inf", "-inf", "-1"]
            for value in values {
                let command = try MusicGenerate.parse([
                    "instrumental guitar", "--model", "/missing/export-fade-regression-model",
                    "\(flag)=\(value)",
                ])
                do {
                    try await command.run()
                    XCTFail("Expected rejection for \(flag)=\(value)")
                } catch let error as ValidationError {
                    XCTAssertEqual(
                        error.message,
                        flag == "--target-peak-db"
                            ? "--target-peak-db must be finite and <= 0"
                            : "Output fades must be finite and >= 0"
                    )
                }
            }
        }
    }

}
