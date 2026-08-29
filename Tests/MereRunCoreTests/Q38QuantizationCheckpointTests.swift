#if os(macOS) && DEBUG
import Foundation
import CryptoKit
import MLX
import XCTest
@testable import MereRunCore

final class Q38QuantizationCheckpointTests: MereRunCoreTestCase {
    func testInstalledCandidateManifest() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_QUANTIZATION"] == "1",
                          "Full-checkpoint quantization experiments are opt-in.")
        let root = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        let path = try XCTUnwrap(environment["MERERUN_TEST_Q38_QUANTIZATION_MANIFEST"])
        let recipe = try XCTUnwrap(Q38ExpertRequantization(
            rawValue: environment["MERERUN_TEST_Q38_QUANTIZATION_RECIPE"] ?? "q4"
        ))
        var manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertFalse(manifest.cases.isEmpty)
        XCTAssertEqual(Set(manifest.cases.map(\.id)).count, manifest.cases.count)
        var observedIDs = Set(manifest.cases.map(\.id))
        let profileOutput = environment["MERERUN_TEST_Q38_ACTIVATION_PROFILE_OUTPUT"].map(URL.init(fileURLWithPath:))
        let profileInput = environment["MERERUN_TEST_Q38_ACTIVATION_PROFILE_INPUT"].map(URL.init(fileURLWithPath:))
        let profiler = profileOutput.map { _ in Q38ExpertActivationProfile() }
        if profiler != nil {
            XCTAssertEqual(recipe, .q4)
            XCTAssertEqual(environment["MERERUN_Q35_MTP_SPECULATION"], "0")
        }
        if recipe == .q3ActivationRefit {
            let metadata = try SafetensorsStreamingLoader.fileMetadata(url: XCTUnwrap(profileInput))
            XCTAssertEqual(metadata["method"], "q4-expert-input-second-moments-v1")
            XCTAssertEqual(metadata["source_revision"], "6cc9bbc0fae9ce26b7670b3ed1e26d557c154506")
            XCTAssertEqual(metadata["calibration_sha256"],
                           "467ae6c9002d63a201edbaba050a344fb275a1ae01b4116fdd19efa85cb718eb")
        }
        let generator = Q35Generator(modelId: Q35Resources.q38FlashNext4BitModelId,
                                     prefixKVCacheEnabled: false, continuousBatchingEnabled: false)
        await generator.setCheckpointTransformForTesting { model in
            try recipe.transform(model, profileURL: profileInput)
            profiler?.install(on: model)
        }
        do {
            try await generator.prepare(modelPath: root)
            Memory.clearCache()
            FileHandle.standardError.write(Data(
                "[q38-quantization-ready] recipe=\(recipe.rawValue) active_bytes=\(Memory.activeMemory)\n".utf8
            ))
            var followedUp = false
            while true {
                var phasePassed = true
                for item in manifest.cases {
                    profiler?.modality = item.image == nil ? .text : .image
                    let response = try await generator.chat(
                        ChatRequest(messages: [.init(role: .user, content: item.prompt, imageUrl: item.image)],
                                    maxTokens: item.maxTokens ?? 96, temperature: 0, topP: 1,
                                    showThinking: false, maxContextTokens: 8_192),
                        modelPath: root, progressHandler: nil
                    )
                    let actual = response.response.trimmingCharacters(in: .whitespacesAndNewlines)
                    let record = Record(id: item.id, recipe: recipe.rawValue, expected: item.expected,
                                        actual: actual, exactMatch: item.expected.map { $0 == actual },
                                        outputTokens: response.tokensGenerated, acceleration: response.acceleration,
                                        prefillSeconds: response.timing?.prefillSeconds,
                                        decodeSeconds: response.timing?.decodeSeconds,
                                        activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory)
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.sortedKeys]
                    FileHandle.standardError.write(Data("[q38-quantization] ".utf8)
                        + (try encoder.encode(record)) + Data("\n".utf8))
                    if let expected = item.expected, actual != expected {
                        phasePassed = false
                        XCTAssertEqual(actual, expected, item.id)
                    }
                    let mtp = environment["MERERUN_Q35_MTP_SPECULATION"] == "1" && item.image == nil
                    let expectedRoute = mtp ? "mtp-speculative" : "final-target-pipelined"
                    if response.acceleration?.route != expectedRoute {
                        phasePassed = false
                        XCTAssertEqual(response.acceleration?.route, expectedRoute)
                    }
                }
                guard phasePassed, !followedUp,
                      let path = environment["MERERUN_TEST_Q38_QUANTIZATION_FOLLOWUP_MANIFEST"] else { break }
                followedUp = true
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                manifest = try JSONDecoder().decode(Manifest.self, from: data)
                XCTAssertFalse(manifest.cases.isEmpty)
                let followupIDs = Set(manifest.cases.map(\.id))
                XCTAssertEqual(followupIDs.count, manifest.cases.count)
                XCTAssertTrue(observedIDs.isDisjoint(with: followupIDs))
                observedIDs.formUnion(followupIDs)
                let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                FileHandle.standardError.write(Data(
                    "[q38-quantization-followup] cases=\(manifest.cases.count) sha256=\(digest)\n".utf8
                ))
            }
            if let profiler, let profileOutput {
                let digest = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: path)))
                    .map { String(format: "%02x", $0) }.joined()
                XCTAssertEqual(digest, "467ae6c9002d63a201edbaba050a344fb275a1ae01b4116fdd19efa85cb718eb")
                try profiler.save(to: profileOutput, metadata: [
                    "method": "q4-expert-input-second-moments-v1", "calibration_sha256": digest,
                    "source_revision": "6cc9bbc0fae9ce26b7670b3ed1e26d557c154506",
                ])
            }
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private struct Manifest: Decodable {
        let cases: [Case]
    }

    private struct Case: Decodable {
        let id: String
        let image: String?
        let prompt: String
        let expected: String?
        let maxTokens: Int?
    }

    private struct Record: Encodable {
        let id: String
        let recipe: String
        let expected: String?
        let actual: String
        let exactMatch: Bool?
        let outputTokens: Int
        let acceleration: ChatAccelerationDiagnostics?
        let prefillSeconds: Double?
        let decodeSeconds: Double?
        let activeBytes: Int
        let cacheBytes: Int
    }
}
#endif
