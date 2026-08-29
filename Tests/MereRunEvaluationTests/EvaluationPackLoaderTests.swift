import Foundation
import XCTest
@testable import MereRunEvaluation

final class EvaluationPackLoaderTests: XCTestCase {
    func testLoadsSyntheticPackAndPinsOnlyDeclaredFiles() throws {
        let pack = try EvaluationPackLoader.load(from: fixturePackURL())

        XCTAssertEqual(pack.manifest.schemaVersion, 1)
        XCTAssertEqual(pack.manifest.id, "synthetic-color-check")
        XCTAssertEqual(pack.cases.count, 2)
        XCTAssertEqual(pack.imageURLs.count, 0)
        XCTAssertEqual(pack.promptSets.count, 1)
        XCTAssertEqual(pack.manifest.arms.count, 4)
        XCTAssertEqual(pack.manifest.samplingProfiles.count, 2)
        XCTAssertEqual(pack.files.map(\.relativePath), [
            "cases.jsonl",
            "eval-pack.json",
            "prompts/concise.txt",
        ])
        XCTAssertEqual(pack.packSHA256.count, 64)
        XCTAssertEqual(pack.manifestSHA256.count, 64)
        XCTAssertTrue(pack.cases.allSatisfy { $0.contentSHA256.count == 64 })
        XCTAssertEqual(
            pack.promptSet(id: "concise")?.systemPrompt,
            "Answer directly and concisely.\n"
        )
    }

    func testPackHashChangesWhenDeclaredContentChanges() throws {
        let source = try fixturePackURL()
        let copy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: copy) }
        let original = try EvaluationPackLoader.load(from: copy)
        let promptURL = copy.appendingPathComponent("prompts/concise.txt")
        try Data("Use one sentence.\n".utf8).write(to: promptURL, options: .atomic)
        let changed = try EvaluationPackLoader.load(from: copy)

        XCTAssertNotEqual(original.packSHA256, changed.packSHA256)
        XCTAssertNotEqual(
            original.promptSet(id: "concise")?.contentSHA256,
            changed.promptSet(id: "concise")?.contentSHA256
        )
    }

    func testLoadsAndPinsManifestDeclaredVisionInput() throws {
        let copy = try visionPackCopy()
        defer { try? FileManager.default.removeItem(at: copy) }

        let pack = try EvaluationPackLoader.load(from: copy)
        let imagePath = "images/synthetic.png"

        XCTAssertEqual(pack.imageURLs.keys.sorted(), [imagePath])
        XCTAssertEqual(
            pack.imageURL(relativePath: imagePath)?.path,
            copy.appendingPathComponent(imagePath).path
        )
        XCTAssertEqual(pack.cases[0].specification.messages[0].imageFile, imagePath)
        XCTAssertEqual(pack.files.map(\.relativePath), [
            "cases.jsonl",
            "eval-pack.json",
            imagePath,
            "prompts/concise.txt",
        ])
    }

    func testPackHashChangesWhenDeclaredImageChanges() throws {
        let copy = try visionPackCopy()
        defer { try? FileManager.default.removeItem(at: copy) }
        let original = try EvaluationPackLoader.load(from: copy)
        let imageURL = copy.appendingPathComponent("images/synthetic.png")
        try Data("changed synthetic image bytes".utf8).write(to: imageURL, options: .atomic)
        let changed = try EvaluationPackLoader.load(from: copy)

        XCTAssertNotEqual(original.packSHA256, changed.packSHA256)
        XCTAssertNotEqual(
            original.files.first { $0.relativePath == "images/synthetic.png" }?.sha256,
            changed.files.first { $0.relativePath == "images/synthetic.png" }?.sha256
        )
    }

    func testRejectsVisionInputThatIsNotDeclaredByManifest() throws {
        let copy = try visionPackCopy()
        defer { try? FileManager.default.removeItem(at: copy) }
        let manifestURL = copy.appendingPathComponent("eval-pack.json")
        var text = try String(contentsOf: manifestURL, encoding: .utf8)
        text = text.replacingOccurrences(
            of: "  \"image_files\": [\n    \"images/synthetic.png\"\n  ],\n",
            with: ""
        )
        try Data(text.utf8).write(to: manifestURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: copy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("not declared in image_files"))
        }
    }

    func testRejectsUnsupportedFieldsInsteadOfSilentlyIgnoringThem() throws {
        let source = try fixturePackURL()
        let manifestCopy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: manifestCopy) }
        let manifestURL = manifestCopy.appendingPathComponent("eval-pack.json")
        var manifest = try String(contentsOf: manifestURL, encoding: .utf8)
        manifest = manifest.replacingOccurrences(
            of: "  \"schema_version\": 1,",
            with: "  \"schema_version\": 1,\n  \"silent_manifest_field\": true,"
        )
        try Data(manifest.utf8).write(to: manifestURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: manifestCopy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("silent_manifest_field"))
        }

        let caseCopy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: caseCopy) }
        let casesURL = caseCopy.appendingPathComponent("cases.jsonl")
        var cases = try String(contentsOf: casesURL, encoding: .utf8)
        cases = cases.replacingOccurrences(
            of: "\"content\":\"What color is a clear daytime sky?\"}",
            with: "\"content\":\"What color is a clear daytime sky?\",\"image\":\"ignored.png\"}"
        )
        try Data(cases.utf8).write(to: casesURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: caseCopy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("messages[0]"))
            XCTAssertTrue(error.localizedDescription.contains("image"))
        }
    }

    func testRejectsTraversalBeforeReadingExternalFiles() throws {
        let source = try fixturePackURL()
        let copy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: copy) }
        let manifestURL = copy.appendingPathComponent("eval-pack.json")
        var text = try String(contentsOf: manifestURL, encoding: .utf8)
        text = text.replacingOccurrences(of: "cases.jsonl", with: "../outside.jsonl")
        try Data(text.utf8).write(to: manifestURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: copy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("normalized relative path"))
        }
    }

    func testRejectsDuplicateCaseIDsAcrossFiles() throws {
        let source = try fixturePackURL()
        let copy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: copy) }
        let casesURL = copy.appendingPathComponent("cases.jsonl")
        let first = try String(contentsOf: casesURL, encoding: .utf8)
            .split(separator: "\n")
            .first
            .map(String.init)
        XCTAssertNotNil(first)
        try Data("\(first!)\n\(first!)\n".utf8).write(to: casesURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: copy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("duplicate case id"))
        }
    }

    func testRejectsCaseFilesThatAreNotUTF8() throws {
        let source = try fixturePackURL()
        let copy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: copy) }
        let casesURL = copy.appendingPathComponent("cases.jsonl")
        try Data([0x7B, 0xFF, 0x7D, 0x0A]).write(to: casesURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: copy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("not UTF-8"))
        }
    }

    func testRejectsDuplicateProfileIDsWithinAnArm() throws {
        let source = try fixturePackURL()
        let copy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: copy) }
        let manifestURL = copy.appendingPathComponent("eval-pack.json")
        var text = try String(contentsOf: manifestURL, encoding: .utf8)
        text = text.replacingOccurrences(
            of: "\"adapter_scale\": 1",
            with: "\"adapter_scale\": 1, \"profile_ids\": [\"sampled-quality\", \"sampled-quality\"]"
        )
        try Data(text.utf8).write(to: manifestURL, options: .atomic)

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: copy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("profile_ids must be unique"))
        }
    }

    func testRejectsDeclaredSymlink() throws {
        let source = try fixturePackURL()
        let copy = try temporaryPackCopy(source)
        defer { try? FileManager.default.removeItem(at: copy) }
        let promptURL = copy.appendingPathComponent("prompts/concise.txt")
        try FileManager.default.removeItem(at: promptURL)
        try FileManager.default.createSymbolicLink(
            at: promptURL,
            withDestinationURL: copy.appendingPathComponent("cases.jsonl")
        )

        XCTAssertThrowsError(try EvaluationPackLoader.load(from: copy)) { error in
            XCTAssertTrue(error.localizedDescription.contains("symbolic link"))
        }
    }

    func testScorerProtocolRoundTripsWithoutDomainTypes() throws {
        let request = EvaluationScorerRequest(
            packID: "synthetic-pack",
            packVersion: "1.0.0",
            packSHA256: String(repeating: "a", count: 64),
            caseID: "synthetic-case",
            caseSHA256: String(repeating: "b", count: 64),
            armID: "candidate",
            profileID: "sampled",
            trial: 2,
            modelID: "synthetic-model",
            adapterSHA256: String(repeating: "c", count: 64),
            response: "synthetic response",
            responseSHA256: String(repeating: "d", count: 64)
        )
        let encoded = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(EvaluationScorerRequest.self, from: encoded)

        XCTAssertEqual(decoded, request)
    }

    private func fixturePackURL() throws -> URL {
        try XCTUnwrap(
            Bundle.module.url(
                forResource: "SyntheticEvaluationPack",
                withExtension: nil,
                subdirectory: "Fixtures"
            )
        )
    }

    private func temporaryPackCopy(_ source: URL) throws -> URL {
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("evaluation-pack-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.copyItem(at: source, to: destination)
        return destination
    }

    private func visionPackCopy() throws -> URL {
        let copy = try temporaryPackCopy(fixturePackURL())
        let images = copy.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        try Data("synthetic image bytes".utf8).write(
            to: images.appendingPathComponent("synthetic.png"),
            options: .atomic
        )

        let manifestURL = copy.appendingPathComponent("eval-pack.json")
        var manifest = try String(contentsOf: manifestURL, encoding: .utf8)
        manifest = manifest.replacingOccurrences(
            of: "  \"case_files\": [\n    \"cases.jsonl\"\n  ],\n",
            with: "  \"case_files\": [\n    \"cases.jsonl\"\n  ],\n"
                + "  \"image_files\": [\n    \"images/synthetic.png\"\n  ],\n"
        )
        try Data(manifest.utf8).write(to: manifestURL, options: .atomic)

        let casesURL = copy.appendingPathComponent("cases.jsonl")
        var cases = try String(contentsOf: casesURL, encoding: .utf8)
        cases = cases.replacingOccurrences(
            of: "\"content\":\"What color is a clear daytime sky?\"}",
            with: "\"content\":\"What color is a clear daytime sky?\","
                + "\"image_file\":\"images/synthetic.png\"}"
        )
        try Data(cases.utf8).write(to: casesURL, options: .atomic)
        return copy
    }
}
