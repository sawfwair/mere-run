import AudioTTS
import Foundation
import XCTest

final class Qwen3TTSResourcesSymlinkTests: XCTestCase {
    func testValidationAndWeightDiscoveryTraverseSymlinkedComponents() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let resources = Qwen3TTSResources(rootURL: fixture.model)

        XCTAssertTrue(resources.validate().isEmpty)
        XCTAssertEqual(
            resources.speechTokenizerWeightsURLs.map(\.lastPathComponent),
            ["speech-tokenizer.safetensors"]
        )
        XCTAssertEqual(
            resources.speakerEncoderWeightsURLs.map(\.lastPathComponent),
            ["speaker-encoder.safetensors"]
        )
    }

    private func makeFixture() throws -> (root: URL, model: URL) {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let model = root.appendingPathComponent("model", isDirectory: true)
        let snapshot = root.appendingPathComponent("snapshot", isDirectory: true)
        let speechTokenizer = snapshot.appendingPathComponent("speech_tokenizer", isDirectory: true)
        let speakerEncoder = snapshot.appendingPathComponent("speaker_encoder", isDirectory: true)

        try fileManager.createDirectory(at: model, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: speechTokenizer, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: speakerEncoder, withIntermediateDirectories: true)

        try Data("{}".utf8).write(to: model.appendingPathComponent("config.json"))
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: model.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: model.appendingPathComponent("tokenizer_config.json"))
        try Data("{}".utf8).write(to: speechTokenizer.appendingPathComponent("config.json"))
        try Data().write(to: speechTokenizer.appendingPathComponent("speech-tokenizer.safetensors"))
        try Data().write(to: speakerEncoder.appendingPathComponent("speaker-encoder.safetensors"))

        try fileManager.createSymbolicLink(
            at: model.appendingPathComponent("speech_tokenizer", isDirectory: true),
            withDestinationURL: speechTokenizer
        )
        try fileManager.createSymbolicLink(
            at: model.appendingPathComponent("speaker_encoder", isDirectory: true),
            withDestinationURL: speakerEncoder
        )

        return (root, model)
    }
}
