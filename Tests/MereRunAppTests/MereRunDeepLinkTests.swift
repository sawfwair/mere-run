@testable import MereRunApp
import XCTest

final class MereRunDeepLinkTests: XCTestCase {
    func testPreviewLinkResolvesPercentEncodedAbsolutePath() throws {
        let artifactURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere preview \(UUID().uuidString)")
            .appendingPathExtension("png")
        try Data("preview".utf8).write(to: artifactURL)
        defer { try? FileManager.default.removeItem(at: artifactURL) }

        var components = URLComponents()
        components.scheme = MereRunDeepLink.scheme
        components.host = "preview"
        components.queryItems = [URLQueryItem(name: "path", value: artifactURL.path)]
        let link = try XCTUnwrap(components.url)

        XCTAssertEqual(
            try MereRunDeepLink.parse(link),
            .preview(artifactURL.standardizedFileURL)
        )
    }

    func testPreviewLinkRejectsUnsupportedRoute() throws {
        let link = try XCTUnwrap(URL(string: "mererun://generate?path=/tmp/output.png"))

        XCTAssertThrowsError(try MereRunDeepLink.parse(link)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .unsupportedRoute)
        }
    }

    func testPreviewLinkRequiresExactlyOnePathParameter() throws {
        let missing = try XCTUnwrap(URL(string: "mererun://preview"))
        let duplicate = try XCTUnwrap(URL(string: "mererun://preview?path=/tmp/a&path=/tmp/b"))
        let extra = try XCTUnwrap(URL(string: "mererun://preview?path=/tmp/a&title=Result"))

        XCTAssertThrowsError(try MereRunDeepLink.parse(missing)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .missingPath)
        }
        XCTAssertThrowsError(try MereRunDeepLink.parse(duplicate)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .unsupportedParameters)
        }
        XCTAssertThrowsError(try MereRunDeepLink.parse(extra)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .unsupportedParameters)
        }
    }

    func testPreviewLinkRequiresAbsoluteExistingFile() throws {
        let relative = try XCTUnwrap(URL(string: "mererun://preview?path=output.png"))
        let missingPath = "/tmp/mere-run-missing-\(UUID().uuidString).png"
        var missingComponents = URLComponents()
        missingComponents.scheme = MereRunDeepLink.scheme
        missingComponents.host = "preview"
        missingComponents.queryItems = [URLQueryItem(name: "path", value: missingPath)]
        let missing = try XCTUnwrap(missingComponents.url)

        XCTAssertThrowsError(try MereRunDeepLink.parse(relative)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .pathMustBeAbsolute)
        }
        XCTAssertThrowsError(try MereRunDeepLink.parse(missing)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .fileNotFound(missingPath))
        }
    }

    func testPreviewLinkRejectsDirectory() throws {
        let directoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-preview-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directoryURL) }

        var components = URLComponents()
        components.scheme = MereRunDeepLink.scheme
        components.host = "preview"
        components.queryItems = [URLQueryItem(name: "path", value: directoryURL.path)]
        let link = try XCTUnwrap(components.url)

        XCTAssertThrowsError(try MereRunDeepLink.parse(link)) { error in
            XCTAssertEqual(error as? MereRunDeepLinkError, .notAFile(directoryURL.path))
        }
    }
}
