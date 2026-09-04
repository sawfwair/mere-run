import Foundation
import MereRunCore
import XCTest
@testable import AudioSTT

final class ParakeetCoreMLManifestTests: XCTestCase {
    func testLoadsExactPinnedArtifactClosure() throws {
        let fixture = try makeFixture()
        let loaded = try ParakeetCoreMLManifest.load(
            artifactURL: fixture.root,
            config: makeConfig()
        )

        XCTAssertEqual(loaded.manifest.schemaVersion, 1)
        XCTAssertEqual(
            loaded.manifest.source.revision,
            ParakeetCoreMLManifest.sourceRevision
        )
        XCTAssertEqual(loaded.manifest.encoder.inputFrames, 1_501)
        XCTAssertEqual(loaded.modelURL.lastPathComponent, "encoder.mlmodelc")
    }

    func testRejectsMutableOrDifferentSourceRevision() throws {
        let fixture = try makeFixture(sourceRevision: "main")

        XCTAssertThrowsError(
            try ParakeetCoreMLManifest.load(
                artifactURL: fixture.root,
                config: makeConfig()
            )
        ) { error in
            guard case ParakeetCoreMLError.untrustedSource = error else {
                return XCTFail("Expected untrustedSource, found \(error)")
            }
        }
    }

    func testRejectsUnpinnedConversionEnvironment() throws {
        let fixture = try makeFixture(converterVersion: 2)

        XCTAssertThrowsError(
            try ParakeetCoreMLManifest.load(
                artifactURL: fixture.root,
                config: makeConfig()
            )
        ) { error in
            guard case ParakeetCoreMLError.untrustedConversion = error else {
                return XCTFail("Expected untrustedConversion, found \(error)")
            }
        }
    }

    func testRejectsCompiledFilesMissingFromArtifactClosure() throws {
        let fixture = try makeFixture(includeUnlistedFile: true)

        XCTAssertThrowsError(
            try ParakeetCoreMLManifest.load(
                artifactURL: fixture.root,
                config: makeConfig()
            )
        ) { error in
            guard case ParakeetCoreMLError.artifactClosureMismatch = error else {
                return XCTFail("Expected artifactClosureMismatch, found \(error)")
            }
        }
    }

    func testRejectsArtifactPathTraversalBeforeReadingFiles() throws {
        let fixture = try makeFixture(artifactFilename: "../escape")

        XCTAssertThrowsError(
            try ParakeetCoreMLManifest.load(
                artifactURL: fixture.root,
                config: makeConfig()
            )
        ) { error in
            guard case ParakeetCoreMLError.unsafeArtifactPath("../escape") = error else {
                return XCTFail("Expected unsafeArtifactPath, found \(error)")
            }
        }
    }

    func testLoadsStandaloneHybridArtifactClosure() throws {
        let fixture = try makeHybridFixture()
        let loaded = try ParakeetCoreMLManifest.load(
            artifactURL: fixture.root,
            config: makeConfig(packaging: .coreMLHybrid)
        )

        XCTAssertEqual(loaded.manifest.schemaVersion, 2)
        XCTAssertEqual(loaded.manifest.decoder?.tensorCount, 13)
    }

    func testRejectsUnlistedFileInStandaloneHybridArtifact() throws {
        let fixture = try makeHybridFixture()
        try Data("extra".utf8).write(to: fixture.root.appendingPathComponent("extra.bin"))

        XCTAssertThrowsError(
            try ParakeetCoreMLManifest.load(
                artifactURL: fixture.root,
                config: makeConfig(packaging: .coreMLHybrid)
            )
        ) { error in
            guard case ParakeetCoreMLError.artifactClosureMismatch = error else {
                return XCTFail("Expected artifactClosureMismatch, found \(error)")
            }
        }
    }

    func testLoadsLaneBatchedCoreMLDecoderArtifact() throws {
        let fixture = try makeHybridFixture(includeCoreMLDecoder: true)
        let loaded = try ParakeetCoreMLManifest.load(
            artifactURL: fixture.root,
            config: makeConfig(packaging: .coreMLHybrid)
        )

        XCTAssertEqual(loaded.manifest.schemaVersion, 3)
        XCTAssertEqual(loaded.manifest.coreMLDecoder?.lanes, 16)
        XCTAssertEqual(loaded.manifest.coreMLDecoder?.windowFrames, 8)
    }

    private func makeFixture(
        sourceRevision: String = ParakeetCoreMLManifest.sourceRevision,
        artifactFilename: String = "encoder.mlmodelc/model.mil",
        converterVersion: Int = 1,
        includeUnlistedFile: Bool = false
    ) throws -> (root: URL, directory: URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "parakeet-coreml-manifest-\(UUID().uuidString)",
            isDirectory: true
        )
        let root = directory.appendingPathComponent("artifact", isDirectory: true)
        let model = root.appendingPathComponent("encoder.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let modelFile = model.appendingPathComponent("model.mil")
        try Data("fixture".utf8).write(to: modelFile)
        if includeUnlistedFile {
            try Data("unlisted".utf8).write(
                to: model.appendingPathComponent("unlisted.bin")
            )
        }
        let sha256 = try ModelArtifactPin.fileSHA256(modelFile)
        let manifest = """
        {
          "schemaVersion": 1,
          "source": {
            "repository": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "\(sourceRevision)",
            "license": "CC-BY-4.0"
          },
          "conversion": {
            "converter": "convert_parakeet_coreml.py",
            "converterVersion": \(converterVersion),
            "python": "3.12.12",
            "torch": "2.7.0",
            "transformers": "5.16.1",
            "coremltools": "9.0",
            "xcode": "Xcode 26.4 / Build version 17E192"
          },
          "encoder": {
            "compiledModelDirectory": "encoder.mlmodelc",
            "inputName": "input_features",
            "attentionMaskInputName": "attention_mask",
            "outputName": "encoded_features",
            "outputMaskName": "encoded_attention_mask",
            "inputFrames": 1501,
            "inputFeatures": 128,
            "outputFeatures": 1024,
            "sampleRate": 16000,
            "windowSeconds": 15.0
          },
          "artifacts": [
            {
              "filename": "\(artifactFilename)",
              "byteCount": 7,
              "sha256": "\(sha256)"
            }
          ]
        }
        """
        try Data(manifest.utf8).write(
            to: root.appendingPathComponent(ParakeetCoreMLManifest.filename)
        )
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return (root, directory)
    }

    private func makeHybridFixture(
        includeCoreMLDecoder: Bool = false
    ) throws -> (root: URL, directory: URL) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "parakeet-coreml-hybrid-manifest-\(UUID().uuidString)",
            isDirectory: true
        )
        let root = directory.appendingPathComponent("artifact", isDirectory: true)
        let model = root.appendingPathComponent("encoder.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        var files: [(String, Data)] = [
            ("encoder.mlmodelc/model.mil", Data("encoder".utf8)),
            ("model.safetensors", Data("decoder".utf8)),
            ("config.json", Data("config".utf8)),
            ("vocab.txt", Data("token\n".utf8)),
        ]
        if includeCoreMLDecoder {
            let decoder = root.appendingPathComponent("decoder.mlmodelc", isDirectory: true)
            try FileManager.default.createDirectory(at: decoder, withIntermediateDirectories: true)
            files += [
                ("decoder.mlmodelc/model.mil", Data("decoder-coreml".utf8)),
                ("embedding.f16", Data(repeating: 0, count: 16)),
            ]
        }
        var artifactJSON: [String] = []
        for (filename, data) in files {
            let url = root.appendingPathComponent(filename)
            try data.write(to: url)
            artifactJSON.append(
                """
                {"filename":"\(filename)","byteCount":\(data.count),"sha256":"\(try ModelArtifactPin.fileSHA256(url))"}
                """
            )
        }
        let schemaVersion = includeCoreMLDecoder ? 3 : 2
        let converterVersion = includeCoreMLDecoder ? 3 : 2
        let coreMLDecoder = includeCoreMLDecoder
            ? """
              ,"coreMLDecoder": {
                "compiledModelDirectory": "decoder.mlmodelc",
                "embeddingFile": "embedding.f16",
                "encoderInputName": "encoder_window",
                "embeddingInputName": "token_embedding",
                "hiddenInputName": "hidden_state",
                "cellInputName": "cell_state",
                "tokenOutputName": "token",
                "durationOutputName": "duration",
                "hiddenOutputName": "next_hidden",
                "cellOutputName": "next_cell",
                "lanes": 16,
                "windowFrames": 8,
                "hiddenSize": 640,
                "layers": 2,
                "vocabularySize": 0
              }
              """
            : ""
        let manifest = """
        {
          "schemaVersion": \(schemaVersion),
          "source": {
            "repository": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "\(ParakeetCoreMLManifest.sourceRevision)",
            "license": "CC-BY-4.0"
          },
          "conversion": {
            "converter": "convert_parakeet_coreml.py",
            "converterVersion": \(converterVersion),
            "python": "3.12.12",
            "torch": "2.7.0",
            "transformers": "5.16.1",
            "coremltools": "9.0",
            "xcode": "Xcode 26.4 / Build version 17E192"
          },
          "encoder": {
            "compiledModelDirectory": "encoder.mlmodelc",
            "inputName": "input_features",
            "attentionMaskInputName": "attention_mask",
            "outputName": "encoded_features",
            "outputMaskName": "encoded_attention_mask",
            "inputFrames": 1501,
            "inputFeatures": 128,
            "outputFeatures": 1024,
            "sampleRate": 16000,
            "windowSeconds": 15.0
          },
          "decoder": {
            "format": "coreml-hybrid-v1",
            "weightsFile": "model.safetensors",
            "configFile": "config.json",
            "vocabularyFile": "vocab.txt",
            "tensorCount": 13
          }
          \(coreMLDecoder),
          "artifacts": [\(artifactJSON.joined(separator: ","))]
        }
        """
        try Data(manifest.utf8).write(
            to: root.appendingPathComponent(ParakeetCoreMLManifest.filename)
        )
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return (root, directory)
    }

    private func makeConfig(
        packaging: ParakeetPackaging = .completeMLX
    ) -> ParakeetModelConfig {
        ParakeetModelConfig(
            packaging: packaging,
            variant: .tdt,
            target: "test",
            preprocessor: ParakeetPreprocessorConfig(
                sampleRate: 16_000,
                normalize: "per_feature",
                windowSize: 0.025,
                windowStride: 0.01,
                window: "hann",
                features: 128,
                nFFT: 512,
                dither: 0,
                padTo: 0,
                padValue: 0,
                preemph: 0
            ),
            encoder: ParakeetEncoderConfig(
                featIn: 128,
                layers: 24,
                modelDim: 1_024,
                heads: 8,
                ffExpansionFactor: 4,
                subsamplingFactor: 8,
                selfAttentionModel: "rel_pos",
                subsampling: "dw_striding",
                convKernelSize: 9,
                subsamplingConvChannels: 256,
                posEmbMaxLen: 5_000,
                causalDownsampling: false,
                useBias: false,
                xScaling: false,
                subsamplingConvChunkingFactor: 1
            ),
            rnntDecoder: nil,
            ctcDecoder: nil,
            joint: nil,
            tdtDurations: [0, 1, 2, 3, 4],
            maxSymbols: 10,
            quantizationBits: nil,
            quantizationGroupSize: nil,
            supportedLanguageCodes: ["en"]
        )
    }
}
