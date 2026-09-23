import Foundation
import AudioCodecs
import MLX
import XCTest

@testable import AudioSTT
@testable import AudioSortformer
@testable import MereRunCore

final class SortformerDiarizationTests: XCTestCase {
    func testManagedSortformerRootDoesNotRequireASRTextComponents() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        try MereRunModelManifest.template(
            for: .sortformerDiarization,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"))

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.sortformerDiarization.rawValue
        )

        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertEqual(report.manifest?.engine, .sortformer)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), [.speakerDiarization])
    }

    func testManagedSortformerSpecIsPinnedAndPubliclyPullable() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.sortformerDiarization.rawValue)
        )

        XCTAssertEqual(spec.category, .speechDiarization)
        XCTAssertEqual(spec.validationKind, .sortformer)
        XCTAssertEqual(spec.upstreamRevision, "e23e6404bd9859e93edbf94a740eb1c7fc58f12e")
        XCTAssertEqual(spec.hubFallback?.patterns, ["README.md", "config.json", "model.safetensors"])
        XCTAssertNil(spec.usageRestriction)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
    }

    func testNemotron3ManagedSpecPinsReleasedCheckpointAndEightSpeakerManifest() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Nemotron3DiarizationResources.modelID))
        XCTAssertEqual(spec.category, .speechDiarization)
        XCTAssertEqual(spec.validationKind, .sortformer)
        XCTAssertEqual(spec.hubFallback?.repoId, "nvidia/Nemotron-3-Diarization")
        XCTAssertEqual(spec.hubFallback?.revision, Nemotron3DiarizationResources.revision)
        XCTAssertEqual(spec.hubFallback?.patterns, ["README.md", "Nemotron-3-Diarization.nemo"])
        XCTAssertEqual(spec.usageRestriction?.terms.first?.license, "OpenMDW-1.1")
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(Nemotron3DiarizationResources.archivePin.byteCount, 198_676_480)
        XCTAssertEqual(
            MereRunModelManifest.template(for: .nemotron3Diarization).supports,
            [.speakerDiarization]
        )
    }

    func testNemotron3ValidationRejectsUnpinnedArchive() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try TestFileSystem.writeFile(root.appendingPathComponent("Nemotron-3-Diarization.nemo"))
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Nemotron3DiarizationResources.modelID))
        XCTAssertFalse(spec.validationMessages(in: root).isEmpty)
    }

    func testNemotron3RealCheckpointDiarizesTwoVoicesWhenConfigured() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MERERUN_NEMOTRON3_MODEL_DIR"],
              let audioPath = environment["MERERUN_NEMOTRON3_AUDIO"] else {
            throw XCTSkip("Set MERERUN_NEMOTRON3_MODEL_DIR and MERERUN_NEMOTRON3_AUDIO for checkpoint smoke")
        }
        let audio = try AudioReader.readAudioBuffer(
            from: URL(fileURLWithPath: audioPath), sampleRate: 16_000, channels: 1
        )
        let diarizer = try Nemotron3Diarizer(modelDirectory: URL(fileURLWithPath: modelPath))
        let result = try diarizer.diarize(
            samples: audio.samples,
            sampleRate: audio.sampleRate,
            minDuration: 0.1,
            mergeGap: 0.1
        )
        XCTAssertGreaterThanOrEqual(result.numSpeakers, 2)
        XCTAssertTrue(result.segments.allSatisfy { $0.start >= 0 && $0.end > $0.start })
    }

    func testNemotron3LiveChunksMatchFileInferenceWhenConfigured() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MERERUN_NEMOTRON3_MODEL_DIR"],
              let audioPath = environment["MERERUN_NEMOTRON3_AUDIO"] else {
            throw XCTSkip("Set MERERUN_NEMOTRON3_MODEL_DIR and MERERUN_NEMOTRON3_AUDIO for live parity")
        }
        let audio = try AudioReader.readAudioBuffer(
            from: URL(fileURLWithPath: audioPath), sampleRate: 16_000, channels: 1
        )
        let diarizer = try Nemotron3Diarizer(modelDirectory: URL(fileURLWithPath: modelPath))
        let file = try diarizer.diarize(
            samples: audio.samples,
            sampleRate: audio.sampleRate,
            minDuration: 0,
            mergeGap: 0,
            chunkLength: 6,
            rightContext: 2,
            fifoLength: 264,
            cacheUpdatePeriod: 222
        )
        let live = try diarizer.makeStreamingSession(
            chunkLength: 6,
            rightContext: 2,
            fifoLength: 264,
            cacheUpdatePeriod: 222
        )
        var chunks = [Nemotron3DiarizationStreamChunk]()
        var cursor = 0
        while cursor < audio.samples.count {
            let end = min(cursor + 1_137, audio.samples.count)
            chunks += try live.feed(samples: Array(audio.samples[cursor..<end]))
            cursor = end
        }
        XCTAssertFalse(chunks.isEmpty, "Live inference must emit before EOF")
        chunks += try live.finish()
        XCTAssertLessThan(live.bufferedSampleCount, 32_000)
        XCTAssertEqual(chunks.first?.startFrame, 0)
        XCTAssertEqual(chunks.last?.endFrame, audio.samples.count / 160 + 1)

        let frames = audio.samples.count / 160 + 1
        func activity(_ segments: [DiarizationSegment]) -> [UInt8] {
            var masks = [UInt8](repeating: 0, count: frames)
            for segment in segments {
                let start = max(0, Int((segment.start * 100).rounded()))
                let end = min(frames, Int((segment.end * 100).rounded()))
                for frame in start..<end { masks[frame] |= UInt8(1 << segment.speaker) }
            }
            return masks
        }
        let reference = activity(file.segments)
        let incremental = activity(chunks.flatMap(\.segments))
        let mismatches = zip(reference, incremental).filter { $0.0 != $0.1 }.count
        XCTAssertLessThanOrEqual(mismatches, max(3, frames / 100))
    }
}
