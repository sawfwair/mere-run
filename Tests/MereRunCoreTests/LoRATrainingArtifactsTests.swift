import Foundation
import XCTest
import MLX
import MLXNN
@testable import MereRunCore

final class LoRATrainingArtifactsTests: MereRunCoreTestCase {

    func testCheckpointStateRoundTrip() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("artifact.safetensors")
        try Data().write(to: checkpointURL)

        let state = LoRATrainingCheckpointState(
            format: "mererun.flux2.lora",
            baseModel: "flux2-klein",
            checkpointFile: checkpointURL.lastPathComponent,
            step: 42,
            totalSteps: 1000,
            seed: 123,
            rngState: 987654,
            datasetFingerprint: "dataset-abc",
            configFingerprint: "config-def",
            phaseSchedule: [
                .init(width: 1024, height: 1024, steps: 30, sampleCount: 12),
                .init(width: 768, height: 768, steps: 70, sampleCount: 24),
            ],
            phaseCursor: .init(phaseIndex: 1, stepInPhase: 12, phaseSteps: 70),
            configSnapshot: ["model": "/tmp/model", "config_fingerprint": "config-def"],
            lossCSVFile: "artifact.loss.csv",
            lossHTMLFile: "artifact.loss.html",
            manifestFile: "artifact.manifest.json"
        )
        try state.write(nextTo: checkpointURL)

        let loaded = try LoRATrainingCheckpointState.load(nextTo: checkpointURL)
        XCTAssertNotNil(loaded)
        XCTAssertEqual(loaded?.format, "mererun.flux2.lora")
        XCTAssertEqual(loaded?.step, 42)
        XCTAssertEqual(loaded?.rngState, 987654)
        XCTAssertEqual(loaded?.datasetFingerprint, "dataset-abc")
        XCTAssertEqual(loaded?.configFingerprint, "config-def")
        XCTAssertEqual(loaded?.phaseSchedule?.count, 2)
        XCTAssertEqual(loaded?.phaseSchedule?.last?.width, 768)
        XCTAssertEqual(loaded?.phaseCursor?.phaseIndex, 1)
        XCTAssertEqual(loaded?.configSnapshot?["config_fingerprint"], "config-def")
        XCTAssertEqual(loaded?.lossCSVFile, "artifact.loss.csv")
        XCTAssertEqual(loaded?.lossHTMLFile, "artifact.loss.html")
        XCTAssertEqual(loaded?.manifestFile, "artifact.manifest.json")
    }

    func testCheckpointStateCursorComputation() {
        let schedule: [LoRATrainingCheckpointState.Phase] = [
            .init(width: 1024, height: 1024, steps: 5, sampleCount: 10),
            .init(width: 768, height: 768, steps: 10, sampleCount: 20),
        ]

        XCTAssertNil(LoRATrainingCheckpointState.cursor(forCompletedStep: 0, phaseSchedule: schedule))

        let first = LoRATrainingCheckpointState.cursor(forCompletedStep: 3, phaseSchedule: schedule)
        XCTAssertEqual(first?.phaseIndex, 0)
        XCTAssertEqual(first?.stepInPhase, 3)
        XCTAssertEqual(first?.phaseSteps, 5)

        let second = LoRATrainingCheckpointState.cursor(forCompletedStep: 9, phaseSchedule: schedule)
        XCTAssertEqual(second?.phaseIndex, 1)
        XCTAssertEqual(second?.stepInPhase, 4)
        XCTAssertEqual(second?.phaseSteps, 10)

        XCTAssertTrue(LoRATrainingCheckpointState.scheduleMatches(expected: schedule, actual: schedule))
    }

    func testMetricsLoggerPersistsAndResumes() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let outputURL = temp.appendingPathComponent("train-output.safetensors")

        let logger = try LoRATrainingMetricsLogger(baseOutputURL: outputURL, resumeExisting: false)
        try logger.record(step: 1, loss: 1.25)
        try logger.record(step: 2, loss: 0.75)
        try logger.writeArtifacts()

        XCTAssertTrue(FileManager.default.fileExists(atPath: logger.csvURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: logger.htmlURL.path))

        let resumed = try LoRATrainingMetricsLogger(baseOutputURL: outputURL, resumeExisting: true)
        try resumed.record(step: 3, loss: 0.5)
        try resumed.writeArtifacts()

        let csv = try String(contentsOf: resumed.csvURL, encoding: .utf8)
        XCTAssertTrue(csv.contains("step,loss"))
        XCTAssertTrue(csv.contains("1,1.25"))
        XCTAssertTrue(csv.contains("2,0.75"))
        XCTAssertTrue(csv.contains("3,0.5"))
    }

    func testCheckpointArchiveCreation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        let sidecarURL = temp.appendingPathComponent("checkpoint.checkpoint.json")
        let manifestURL = temp.appendingPathComponent("checkpoint.manifest.json")
        try Data("weights".utf8).write(to: checkpointURL)
        try Data("state".utf8).write(to: sidecarURL)
        try Data("manifest".utf8).write(to: manifestURL)

        #if os(macOS)
        let archiveURL = try XCTUnwrap(
            try LoRACheckpointArchive.createZipBundle(
                primaryFile: checkpointURL,
                additionalFiles: [sidecarURL, manifestURL]
            )
        )
        XCTAssertEqual(archiveURL.pathExtension, "zip")
        XCTAssertTrue(FileManager.default.fileExists(atPath: archiveURL.path))
        let attrs = try FileManager.default.attributesOfItem(atPath: archiveURL.path)
        let size = attrs[.size] as? NSNumber
        XCTAssertGreaterThan(size?.intValue ?? 0, 0)
        #else
        let archiveURL = try LoRACheckpointArchive.createZipBundle(
            primaryFile: checkpointURL,
            additionalFiles: [sidecarURL, manifestURL]
        )
        XCTAssertNil(archiveURL)
        #endif
    }

    func testMFluxCompatArtifactsWriterWritesStateFilesAndCheckpointManifest() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let artifacts = try MFluxCheckpointCompatArtifactsWriter.write(
            checkpointURL: checkpointURL,
            step: 12,
            seed: 321,
            batchSize: 2,
            datasetCount: 5,
            loraAdapterFileName: checkpointURL.lastPathComponent,
            optimizerFileName: checkpointURL.lastPathComponent,
            iteratorCursor: nil,
            lossPoints: [
                LoRALossPoint(step: 1, loss: 0.9),
                LoRALossPoint(step: 2, loss: 0.7),
            ],
            configSnapshot: [
                "model": "/tmp/model",
                "dataset_root": "/tmp/data",
                "scheduler_steps": "1000",
                "learning_rate": "0.0002",
                "checkpoint_interval": "25",
                "lora_target_ranks": "layers.0.attn.to_q=8;layers.0.attn.to_v=8",
            ]
        )

        XCTAssertEqual(artifacts.iteratorURL.lastPathComponent, "0000012_iterator.json")
        XCTAssertEqual(artifacts.lossURL.lastPathComponent, "0000012_loss.json")
        XCTAssertEqual(artifacts.configURL.lastPathComponent, "0000012_config.json")
        XCTAssertEqual(artifacts.checkpointManifestURL.lastPathComponent, "checkpoint.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifacts.iteratorURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifacts.lossURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifacts.configURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifacts.checkpointManifestURL.path))

        let checkpointManifest = try XCTUnwrap(loadJSONObject(from: artifacts.checkpointManifestURL) as? [String: Any])
        let files = try XCTUnwrap(checkpointManifest["files"] as? [String: String])
        XCTAssertEqual(files["lora_adapter"], checkpointURL.lastPathComponent)
        XCTAssertEqual(files["optimizer"], checkpointURL.lastPathComponent)
        XCTAssertEqual(files["iterator"], "0000012_iterator.json")
        XCTAssertEqual(files["loss"], "0000012_loss.json")
        XCTAssertEqual(files["config"], "0000012_config.json")

        let iterator = try XCTUnwrap(loadJSONObject(from: artifacts.iteratorURL) as? [String: Any])
        XCTAssertEqual(iterator["num_iterations"] as? Int, 12)
        XCTAssertEqual(iterator["seed"] as? Int, 321)
        XCTAssertEqual(iterator["batch_size"] as? Int, 2)
        XCTAssertNil(iterator["current_permutation"])
        XCTAssertNil(iterator["position"])

        let losses = try XCTUnwrap(loadJSONObject(from: artifacts.lossURL) as? [[String: Any]])
        XCTAssertEqual(losses.count, 2)
        XCTAssertEqual(losses[0]["step"] as? Int, 1)
        XCTAssertEqual(losses[1]["step"] as? Int, 2)

        let config = try XCTUnwrap(loadJSONObject(from: artifacts.configURL) as? [String: Any])
        XCTAssertEqual(config["model"] as? String, "/tmp/model")
        XCTAssertEqual(config["data"] as? String, "/tmp/data")
        XCTAssertEqual(config["steps"] as? Int, 1000)
        let optimizer = try XCTUnwrap(config["optimizer"] as? [String: Any])
        XCTAssertEqual(optimizer["name"] as? String, "adamw")
    }

    func testMFluxCompatArtifactsWriterPersistsIteratorPermutationCursor() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let cursor = MFluxResumeIteratorCompat.Cursor(
            permutation: [3, 1, 2, 0],
            position: 2,
            pythonRNG: nil
        )

        let artifacts = try MFluxCheckpointCompatArtifactsWriter.write(
            checkpointURL: checkpointURL,
            step: 20,
            seed: 42,
            batchSize: 1,
            datasetCount: 4,
            loraAdapterFileName: checkpointURL.lastPathComponent,
            optimizerFileName: checkpointURL.lastPathComponent,
            iteratorCursor: cursor,
            lossPoints: [],
            configSnapshot: [
                "model": "/tmp/model",
                "dataset_root": "/tmp/data",
            ]
        )

        let iterator = try XCTUnwrap(loadJSONObject(from: artifacts.iteratorURL) as? [String: Any])
        XCTAssertEqual(iterator["position"] as? Int, 2)
        XCTAssertEqual(iterator["current_permutation"] as? [Int], [3, 1, 2, 0])
    }

    func testCheckpointResolverPassThroughSafetensors() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let resolved = try LoRACheckpointResolver.resolve(checkpointURL)
        XCTAssertEqual(resolved.checkpointURL, checkpointURL.standardizedFileURL)
        XCTAssertNil(resolved.cleanupURL)
    }

    func testCheckpointResolverExtractsZipBundle() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)
        let state = LoRATrainingCheckpointState(
            format: "mererun.zimage.lora",
            baseModel: "z-image-turbo",
            checkpointFile: checkpointURL.lastPathComponent,
            step: 7,
            totalSteps: 100,
            seed: 123,
            rngState: 456,
            datasetFingerprint: "dataset",
            configFingerprint: "config"
        )
        try state.write(nextTo: checkpointURL)
        let sidecarURL = LoRATrainingCheckpointState.url(nextTo: checkpointURL)

        #if os(macOS)
        let archiveURL = try XCTUnwrap(
            try LoRACheckpointArchive.createZipBundle(
                primaryFile: checkpointURL,
                additionalFiles: [sidecarURL]
            )
        )
        let resolved = try LoRACheckpointResolver.resolve(archiveURL)
        defer { resolved.cleanup() }
        XCTAssertEqual(resolved.checkpointURL.pathExtension, "safetensors")
        XCTAssertTrue(FileManager.default.fileExists(atPath: resolved.checkpointURL.path))
        let loadedState = try LoRATrainingCheckpointState.load(nextTo: resolved.checkpointURL)
        XCTAssertEqual(loadedState?.step, 7)
        XCTAssertEqual(loadedState?.rngState, 456)
        #else
        XCTAssertThrowsError(try LoRACheckpointResolver.resolve(temp.appendingPathComponent("checkpoint.zip")))
        #endif
    }

    func testCheckpointResolverPrefersRunManifestAdapterInZip() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let optimizerURL = temp.appendingPathComponent("a_optimizer.safetensors")
        let adapterURL = temp.appendingPathComponent("z_adapter.safetensors")
        try Data("optimizer".utf8).write(to: optimizerURL)
        try Data("adapter".utf8).write(to: adapterURL)

        let runManifestURL = temp.appendingPathComponent(LoRATrainingRunManifest.filename)
        let runManifestJSON = """
        {
          "version": 1,
          "created_at": "2026-02-08 12:00:00",
          "format": "mererun.zimage.lora",
          "model": "z-image-turbo",
          "is_edit": false,
          "data_root": "/tmp/dataset",
          "data_root_relative": "../dataset",
          "data_fingerprint": {
            "count": 1,
            "images": ["image.png"],
            "input_images": [],
            "is_edit": false
          },
          "checkpoint_files": {
            "lora_adapter": "z_adapter.safetensors",
            "optimizer": "a_optimizer.safetensors"
          },
          "step": 10,
          "total_steps": 100,
          "seed": 123,
          "rng_state": 456
        }
        """
        try runManifestJSON.write(to: runManifestURL, atomically: true, encoding: .utf8)

        #if os(macOS)
        let archiveURL = try XCTUnwrap(
            try LoRACheckpointArchive.createZipBundle(
                primaryFile: optimizerURL,
                additionalFiles: [adapterURL, runManifestURL]
            )
        )
        let resolved = try LoRACheckpointResolver.resolve(archiveURL)
        defer { resolved.cleanup() }
        XCTAssertEqual(resolved.checkpointURL.lastPathComponent, "z_adapter.safetensors")
        #else
        XCTAssertThrowsError(try LoRACheckpointResolver.resolve(temp.appendingPathComponent("checkpoint.zip")))
        #endif
    }

    func testCheckpointResolverExtractsSplitStateFilesFromMinimalRunManifest() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("0000010_adapter.safetensors")
        let optimizerURL = temp.appendingPathComponent("0000010_optimizer.safetensors")
        let iteratorURL = temp.appendingPathComponent("0000010_iterator.json")
        let lossURL = temp.appendingPathComponent("0000010_loss.json")
        let configURL = temp.appendingPathComponent("0000010_config.json")
        try Data("adapter".utf8).write(to: adapterURL)
        try Data("optimizer".utf8).write(to: optimizerURL)
        try Data("{\"num_iterations\":10,\"seed\":7}".utf8).write(to: iteratorURL)
        try Data("[{\"step\":10,\"loss\":0.42}]".utf8).write(to: lossURL)
        try Data("{\"model\":\"z-image-turbo\",\"data\":\"/tmp/data\"}".utf8).write(to: configURL)

        let runManifestURL = temp.appendingPathComponent("run.json")
        let runManifestJSON = """
        {
          "version": 1,
          "created_at": "2026-02-08 12:00:00",
          "model": "z-image-turbo",
          "is_edit": false,
          "data_root": "/tmp/data",
          "data_root_relative": "../data",
          "data_fingerprint": {
            "count": 1,
            "images": ["img.png"],
            "input_images": [],
            "is_edit": false
          },
          "checkpoint_files": {
            "lora_adapter": "0000010_adapter.safetensors",
            "optimizer": "0000010_optimizer.safetensors",
            "iterator": "0000010_iterator.json",
            "loss": "0000010_loss.json",
            "config": "0000010_config.json"
          }
        }
        """
        try runManifestJSON.write(to: runManifestURL, atomically: true, encoding: .utf8)

        #if os(macOS)
        let archiveURL = try XCTUnwrap(
            try LoRACheckpointArchive.createZipBundle(
                primaryFile: optimizerURL,
                additionalFiles: [adapterURL, iteratorURL, lossURL, configURL, runManifestURL]
            )
        )
        let resolved = try LoRACheckpointResolver.resolve(archiveURL)
        defer { resolved.cleanup() }
        XCTAssertEqual(resolved.checkpointURL.lastPathComponent, adapterURL.lastPathComponent)
        XCTAssertEqual(resolved.optimizerStateURL?.lastPathComponent, optimizerURL.lastPathComponent)
        XCTAssertEqual(resolved.iteratorStateURL?.lastPathComponent, iteratorURL.lastPathComponent)
        XCTAssertEqual(resolved.lossStateURL?.lastPathComponent, lossURL.lastPathComponent)
        XCTAssertEqual(resolved.configStateURL?.lastPathComponent, configURL.lastPathComponent)
        #else
        XCTAssertThrowsError(try LoRACheckpointResolver.resolve(temp.appendingPathComponent("checkpoint.zip")))
        #endif
    }

    func testCheckpointResolverPrefersCheckpointManifestAdapterInZip() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let optimizerURL = temp.appendingPathComponent("a_optimizer.safetensors")
        let adapterURL = temp.appendingPathComponent("z_adapter.safetensors")
        try Data("optimizer".utf8).write(to: optimizerURL)
        try Data("adapter".utf8).write(to: adapterURL)

        let checkpointManifestURL = temp.appendingPathComponent("checkpoint.json")
        let checkpointManifestJSON = """
        {
          "files": {
            "lora_adapter": "z_adapter.safetensors",
            "optimizer": "a_optimizer.safetensors"
          }
        }
        """
        try checkpointManifestJSON.write(to: checkpointManifestURL, atomically: true, encoding: .utf8)

        #if os(macOS)
        let archiveURL = try XCTUnwrap(
            try LoRACheckpointArchive.createZipBundle(
                primaryFile: optimizerURL,
                additionalFiles: [adapterURL, checkpointManifestURL]
            )
        )
        let resolved = try LoRACheckpointResolver.resolve(archiveURL)
        defer { resolved.cleanup() }
        XCTAssertEqual(resolved.checkpointURL.lastPathComponent, "z_adapter.safetensors")
        #else
        XCTAssertThrowsError(try LoRACheckpointResolver.resolve(temp.appendingPathComponent("checkpoint.zip")))
        #endif
    }

    func testResumeArtifactsRestoreFromSidecarAndManifest() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let sourceDir = temp.appendingPathComponent("resume-source", isDirectory: true)
        let outputDir = temp.appendingPathComponent("resume-output", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let checkpointURL = sourceDir.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let sourceLossCSV = sourceDir.appendingPathComponent("resume.loss.csv")
        let sourceLossHTML = sourceDir.appendingPathComponent("resume.loss.html")
        try Data("step,loss\n1,0.9\n".utf8).write(to: sourceLossCSV)
        try Data("<html>resume</html>".utf8).write(to: sourceLossHTML)

        let rootSample = sourceDir.appendingPathComponent("step-10.png")
        try Data("sample-10".utf8).write(to: rootSample)
        let nestedSamplesDir = sourceDir.appendingPathComponent("samples", isDirectory: true)
        try FileManager.default.createDirectory(at: nestedSamplesDir, withIntermediateDirectories: true)
        let nestedSample = nestedSamplesDir.appendingPathComponent("step-20.png")
        try Data("sample-20".utf8).write(to: nestedSample)

        let manifest = LoRATrainingManifest(
            format: "mererun.flux2.lora",
            baseModel: "flux2-klein",
            outputFile: checkpointURL.lastPathComponent,
            emaOutputFile: nil,
            training: .init(
                width: 1024,
                height: 1024,
                trainingSteps: 100,
                batchSize: 1,
                learningRate: 1e-4,
                seed: 1,
                datasetCount: 4,
                checkpointInterval: 25,
                sampleInterval: 10,
                samplePrompt: "test",
                emaDecay: 0
            ),
            lora: .init(
                rank: 16,
                alpha: 1.0,
                saveDType: "float16",
                includesOptimizerState: true
            ),
            extras: ["sample_files": "step-10.png,step-20.png"]
        )
        try manifest.write(nextTo: checkpointURL)

        let sidecar = LoRATrainingCheckpointState(
            format: "mererun.flux2.lora",
            baseModel: "flux2-klein",
            checkpointFile: checkpointURL.lastPathComponent,
            step: 20,
            totalSteps: 100,
            seed: 123,
            rngState: 456,
            datasetFingerprint: nil,
            configFingerprint: nil,
            phaseSchedule: nil,
            phaseCursor: nil,
            configSnapshot: nil,
            lossCSVFile: sourceLossCSV.lastPathComponent,
            lossHTMLFile: sourceLossHTML.lastPathComponent,
            manifestFile: LoRATrainingManifest.url(nextTo: checkpointURL).lastPathComponent
        )

        let outputBaseURL = outputDir.appendingPathComponent("new-run.safetensors")
        LoRATrainingResumeArtifacts.restore(
            from: checkpointURL,
            sidecar: sidecar,
            to: outputBaseURL
        )

        let restoredCSV = outputBaseURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("csv")
        let restoredHTML = outputBaseURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("html")
        XCTAssertEqual(try String(contentsOf: restoredCSV, encoding: .utf8), "step,loss\n1,0.9\n")
        XCTAssertEqual(try String(contentsOf: restoredHTML, encoding: .utf8), "<html>resume</html>")

        let restoredSamplesDir = outputDir.appendingPathComponent("samples", isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: restoredSamplesDir.appendingPathComponent("step-10.png").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: restoredSamplesDir.appendingPathComponent("step-20.png").path))
    }

    func testResumeArtifactsFallbackWithoutSidecar() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let sourceDir = temp.appendingPathComponent("fallback-source", isDirectory: true)
        let outputDir = temp.appendingPathComponent("fallback-output", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let checkpointURL = sourceDir.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let fallbackCSV = sourceDir.appendingPathComponent("legacy.loss.csv")
        let fallbackHTML = sourceDir.appendingPathComponent("legacy.loss.html")
        try Data("step,loss\n5,0.2\n".utf8).write(to: fallbackCSV)
        try Data("<html>legacy</html>".utf8).write(to: fallbackHTML)
        try Data("sample-30".utf8).write(to: sourceDir.appendingPathComponent("step-30.jpg"))
        try Data("ignore".utf8).write(to: sourceDir.appendingPathComponent("not-a-sample.txt"))

        let outputBaseURL = outputDir.appendingPathComponent("resumed.safetensors")
        LoRATrainingResumeArtifacts.restore(
            from: checkpointURL,
            sidecar: nil,
            to: outputBaseURL
        )

        let restoredCSV = outputBaseURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("csv")
        let restoredHTML = outputBaseURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("html")
        XCTAssertEqual(try String(contentsOf: restoredCSV, encoding: .utf8), "step,loss\n5,0.2\n")
        XCTAssertEqual(try String(contentsOf: restoredHTML, encoding: .utf8), "<html>legacy</html>")

        let restoredSample = outputDir
            .appendingPathComponent("samples", isDirectory: true)
            .appendingPathComponent("step-30.jpg")
        XCTAssertTrue(FileManager.default.fileExists(atPath: restoredSample.path))
        // If run.json is present, restore prefers explicit artifact filenames from the run manifest.
        let runManifestCSV = sourceDir.appendingPathComponent("checkpoint.metrics.csv")
        let runManifestHTML = sourceDir.appendingPathComponent("checkpoint.metrics.html")
        let runManifestSample = sourceDir.appendingPathComponent("preview.webp")
        try Data("step,loss\n8,0.15\n".utf8).write(to: runManifestCSV)
        try Data("<html>run-manifest</html>".utf8).write(to: runManifestHTML)
        try Data("sample-preview".utf8).write(to: runManifestSample)

        let runManifest = LoRATrainingRunManifest(
            format: "mererun.zimage.lora",
            model: "z-image-turbo",
            isEdit: false,
            dataRoot: nil,
            dataRootRelative: nil,
            dataFingerprint: .init(
                count: 1,
                images: ["image.png"],
                inputImages: [],
                isEdit: false
            ),
            checkpointFiles: [
                "lora_adapter": checkpointURL.lastPathComponent,
                "loss_csv": runManifestCSV.lastPathComponent,
                "loss_html": runManifestHTML.lastPathComponent,
                "sample_files": runManifestSample.lastPathComponent,
            ],
            step: 8,
            totalSteps: 100,
            seed: 123,
            rngState: 456,
            datasetFingerprint: "dataset-fingerprint",
            configFingerprint: "config-fingerprint",
            configSnapshot: nil
        )
        try runManifest.write(nextTo: checkpointURL)

        let outputFromRunManifest = outputDir.appendingPathComponent("resumed-runmanifest.safetensors")
        LoRATrainingResumeArtifacts.restore(
            from: checkpointURL,
            sidecar: nil,
            to: outputFromRunManifest
        )

        let runManifestRestoredCSV = outputFromRunManifest
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("csv")
        let runManifestRestoredHTML = outputFromRunManifest
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("html")
        XCTAssertEqual(try String(contentsOf: runManifestRestoredCSV, encoding: .utf8), "step,loss\n8,0.15\n")
        XCTAssertEqual(try String(contentsOf: runManifestRestoredHTML, encoding: .utf8), "<html>run-manifest</html>")

        let runManifestRestoredSample = outputDir
            .appendingPathComponent("samples", isDirectory: true)
            .appendingPathComponent("preview.webp")
        XCTAssertTrue(FileManager.default.fileExists(atPath: runManifestRestoredSample.path))

        let conflictingSidecar = LoRATrainingCheckpointState(
            format: "mererun.zimage.lora",
            baseModel: "z-image-turbo",
            checkpointFile: checkpointURL.lastPathComponent,
            step: 8,
            totalSteps: 100,
            seed: 123,
            rngState: 456,
            datasetFingerprint: nil,
            configFingerprint: nil,
            phaseSchedule: nil,
            phaseCursor: nil,
            configSnapshot: nil,
            lossCSVFile: fallbackCSV.lastPathComponent,
            lossHTMLFile: fallbackHTML.lastPathComponent,
            manifestFile: nil
        )
        let outputWithSidecar = outputDir.appendingPathComponent("resumed-runmanifest-vs-sidecar.safetensors")
        LoRATrainingResumeArtifacts.restore(
            from: checkpointURL,
            sidecar: conflictingSidecar,
            to: outputWithSidecar
        )

        let restoredWithSidecarCSV = outputWithSidecar
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("csv")
        let restoredWithSidecarHTML = outputWithSidecar
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("html")
        XCTAssertEqual(try String(contentsOf: restoredWithSidecarCSV, encoding: .utf8), "step,loss\n8,0.15\n")
        XCTAssertEqual(try String(contentsOf: restoredWithSidecarHTML, encoding: .utf8), "<html>run-manifest</html>")
    }

    func testResumeArtifactsImportsMfluxLossJSONFromCheckpointManifest() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let sourceDir = temp.appendingPathComponent("mflux-source", isDirectory: true)
        let outputDir = temp.appendingPathComponent("mflux-output", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let checkpointURL = sourceDir.appendingPathComponent("adapter.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let mfluxLossURL = sourceDir.appendingPathComponent("0000010_loss.json")
        let mfluxLossJSON = """
        [
          { "step": 1, "loss": 0.9, "time": "2026-02-08 12:00:00" },
          { "step": "bad", "loss": 42.0 },
          { "step": 2, "loss": 0.7, "time": "2026-02-08 12:00:01" }
        ]
        """
        try mfluxLossJSON.write(to: mfluxLossURL, atomically: true, encoding: .utf8)

        let checkpointManifestURL = sourceDir.appendingPathComponent("checkpoint.json")
        let checkpointManifestJSON = """
        {
          "files": {
            "loss": "0000010_loss.json"
          }
        }
        """
        try checkpointManifestJSON.write(to: checkpointManifestURL, atomically: true, encoding: .utf8)

        let outputBaseURL = outputDir.appendingPathComponent("resumed.safetensors")
        LoRATrainingResumeArtifacts.restore(
            from: checkpointURL,
            sidecar: nil,
            to: outputBaseURL
        )

        let restoredCSV = outputBaseURL
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("csv")
        let csv = try String(contentsOf: restoredCSV, encoding: .utf8)
        XCTAssertEqual(csv, "step,loss\n1,0.9\n2,0.7\n")

        let iteratorURL = sourceDir.appendingPathComponent("0000010_iterator.json")
        var mtState: [Int] = Array(0..<624)
        mtState.append(624)
        let iteratorWithRNG: [String: Any] = [
            "num_iterations": 10,
            "seed": 7,
            "batch_size": 2,
            "position": 3,
            "current_permutation": [0, 1, 2, 3],
            "rng_state": [3, mtState, NSNull()],
        ]
        let iteratorWithRNGData = try JSONSerialization.data(withJSONObject: iteratorWithRNG, options: [])
        try iteratorWithRNGData.write(to: iteratorURL)

        let iteratorState = try XCTUnwrap(MFluxResumeIteratorCompat.loadState(from: iteratorURL))
        XCTAssertEqual(iteratorState.step, 10)
        XCTAssertEqual(iteratorState.seed, 7)

        var cursor = iteratorState.cursor
        let firstBatch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursor
        )
        XCTAssertEqual(firstBatch, [3])
        MFluxResumeIteratorCompat.advanceTrainingRNG(batchSize: firstBatch?.count ?? 0, cursor: &cursor)
        let secondBatch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursor
        )
        XCTAssertNotNil(secondBatch)
        XCTAssertEqual(secondBatch?.count, 2)
        XCTAssertTrue(secondBatch?.allSatisfy({ $0 >= 0 && $0 < 4 }) ?? false)
        XCTAssertNotNil(cursor)

        let iteratorWithoutRNGURL = sourceDir.appendingPathComponent("0000010_iterator_no_rng.json")
        let iteratorWithoutRNG: [String: Any] = [
            "num_iterations": 10,
            "seed": 7,
            "batch_size": 2,
            "position": 3,
            "current_permutation": [0, 1, 2, 3],
        ]
        let iteratorWithoutRNGData = try JSONSerialization.data(withJSONObject: iteratorWithoutRNG, options: [])
        try iteratorWithoutRNGData.write(to: iteratorWithoutRNGURL)

        var cursorWithoutRNG = try XCTUnwrap(MFluxResumeIteratorCompat.loadState(from: iteratorWithoutRNGURL)).cursor
        let legacyBatch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursorWithoutRNG
        )
        XCTAssertEqual(legacyBatch, [3])
        XCTAssertNil(cursorWithoutRNG)
    }

    func testMFluxResumeIteratorCompatDecodesStringScalars() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let iteratorURL = temp.appendingPathComponent("string_iterator.json")
        let iteratorJSON = """
        {
          "num_iterations": "12",
          "seed": "99",
          "batch_size": "2",
          "position": "1",
          "current_permutation": ["0", "1", "2"],
          "zero_rng_state": "12345"
        }
        """
        try iteratorJSON.write(to: iteratorURL, atomically: true, encoding: .utf8)

        let iteratorState = try XCTUnwrap(MFluxResumeIteratorCompat.loadState(from: iteratorURL))
        XCTAssertEqual(iteratorState.step, 12)
        XCTAssertEqual(iteratorState.seed, 99)
        XCTAssertEqual(iteratorState.cursor?.position, 1)
        XCTAssertEqual(iteratorState.cursor?.permutation, [0, 1, 2])
        XCTAssertEqual(iteratorState.cursor?.localRNGState, 12_345)
    }

    func testMFluxResumeIteratorCompatLocalCursorReshufflesAcrossEpochs() throws {
        var cursor = MFluxResumeIteratorCompat.makeLocalCursor(sampleCount: 4, seed: 123)

        let firstBatch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursor
        )
        XCTAssertEqual(firstBatch?.count, 2)
        XCTAssertEqual(Set(firstBatch ?? []).count, 2)
        MFluxResumeIteratorCompat.advanceTrainingRNG(batchSize: firstBatch?.count ?? 0, cursor: &cursor)

        let secondBatch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursor
        )
        XCTAssertEqual(secondBatch?.count, 2)
        XCTAssertEqual(Set(secondBatch ?? []).count, 2)
        XCTAssertEqual(Set((firstBatch ?? []) + (secondBatch ?? [])), Set([0, 1, 2, 3]))
        MFluxResumeIteratorCompat.advanceTrainingRNG(batchSize: secondBatch?.count ?? 0, cursor: &cursor)

        // Next call must reshuffle and continue instead of dropping cursor at epoch boundary.
        let thirdBatch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursor
        )
        XCTAssertEqual(thirdBatch?.count, 2)
        XCTAssertTrue(thirdBatch?.allSatisfy({ $0 >= 0 && $0 < 4 }) ?? false)
        XCTAssertNotNil(cursor)
    }

    func testMFluxResumeIteratorCompatReturnsTrainingSeedPairsFromLocalCursor() throws {
        var cursor: MFluxResumeIteratorCompat.Cursor? = try XCTUnwrap(
            MFluxResumeIteratorCompat.makeLocalCursor(sampleCount: 5, seed: 999)
        )
        let sampled = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 3,
            sampleCount: 5,
            cursor: &cursor
        )
        let seeds = MFluxResumeIteratorCompat.nextTrainingSeedPairs(
            batchSize: sampled?.count ?? 0,
            cursor: &cursor
        )
        XCTAssertEqual(seeds?.count, sampled?.count)
        XCTAssertTrue(seeds?.allSatisfy { $0.time <= UInt64(UInt32.max) && $0.noise <= UInt64(UInt32.max) } ?? false)
        XCTAssertNotNil(cursor?.localRNGState)
    }

    func testMFluxResumeIteratorCompatMakePythonCursorMatchesMfluxSeededRNG() throws {
        var cursor: MFluxResumeIteratorCompat.Cursor? = try XCTUnwrap(
            MFluxResumeIteratorCompat.makePythonCursor(sampleCount: 10, seed: 42)
        )
        XCTAssertNotNil(cursor?.pythonRNG)
        XCTAssertNil(cursor?.localRNGState)

        let batch = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 4,
            sampleCount: 10,
            cursor: &cursor
        )
        XCTAssertEqual(batch, [7, 3, 2, 8])

        let seeds = MFluxResumeIteratorCompat.nextTrainingSeedPairs(batchSize: 4, cursor: &cursor)
        XCTAssertEqual(seeds?.count, 4)
        XCTAssertEqual(seeds?[0].time, 2_536_146_025)
        XCTAssertEqual(seeds?[0].noise, 136_505_587)
        XCTAssertEqual(seeds?[1].time, 402_418_010)
        XCTAssertEqual(seeds?[1].noise, 2_585_650_756)
        XCTAssertEqual(seeds?[2].time, 2_410_529_190)
        XCTAssertEqual(seeds?[2].noise, 1_801_823_908)
        XCTAssertEqual(seeds?[3].time, 3_733_616_459)
        XCTAssertEqual(seeds?[3].noise, 1_815_115_025)
    }

    func testMFluxResumeIteratorCompatReturnsTrainingSeedPairsFromPythonCursor() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let iteratorURL = temp.appendingPathComponent("iterator.json")
        var mtState: [Int] = Array(0..<624)
        mtState.append(624)
        let iterator: [String: Any] = [
            "num_iterations": 4,
            "seed": 7,
            "batch_size": 2,
            "position": 1,
            "current_permutation": [0, 1, 2, 3],
            "rng_state": [3, mtState, NSNull()],
        ]
        try JSONSerialization.data(withJSONObject: iterator, options: []).write(to: iteratorURL)

        var cursor: MFluxResumeIteratorCompat.Cursor? = try XCTUnwrap(
            MFluxResumeIteratorCompat.loadState(from: iteratorURL)?.cursor
        )
        _ = MFluxResumeIteratorCompat.nextBatchIndices(
            requestedBatchSize: 2,
            sampleCount: 4,
            cursor: &cursor
        )
        let seeds = MFluxResumeIteratorCompat.nextTrainingSeedPairs(batchSize: 2, cursor: &cursor)
        XCTAssertEqual(seeds?.count, 2)
        XCTAssertNotNil(cursor?.pythonRNG)
    }

    func testMFluxCompatArtifactsWriterPersistsLocalRNGStateInIteratorJSON() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let cursor = try XCTUnwrap(
            MFluxResumeIteratorCompat.makeLocalCursor(sampleCount: 5, seed: 777)
        )
        let artifacts = try MFluxCheckpointCompatArtifactsWriter.write(
            checkpointURL: checkpointURL,
            step: 9,
            seed: 777,
            batchSize: 2,
            datasetCount: 5,
            loraAdapterFileName: checkpointURL.lastPathComponent,
            optimizerFileName: checkpointURL.lastPathComponent,
            iteratorCursor: cursor,
            lossPoints: [],
            configSnapshot: [
                "model": "/tmp/model",
                "dataset_root": "/tmp/data",
            ]
        )

        let iterator = try XCTUnwrap(loadJSONObject(from: artifacts.iteratorURL) as? [String: Any])
        XCTAssertNotNil(iterator["current_permutation"] as? [Int])
        XCTAssertNotNil(iterator["position"] as? Int)
        XCTAssertNotNil(iterator["zero_rng_state"] as? String)
        XCTAssertNil(iterator["rng_state"])
    }

    func testMFluxCompatArtifactsWriterPersistsPythonRNGStateInIteratorJSON() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let checkpointURL = temp.appendingPathComponent("checkpoint.safetensors")
        try Data("weights".utf8).write(to: checkpointURL)

        let iteratorSourceURL = temp.appendingPathComponent("source_iterator.json")
        var mtState: [Int] = Array(0..<624)
        mtState.append(624)
        let iteratorSource: [String: Any] = [
            "num_iterations": 5,
            "seed": 123,
            "batch_size": 2,
            "position": 1,
            "current_permutation": [0, 1, 2, 3],
            "rng_state": [3, mtState, NSNull()],
        ]
        try JSONSerialization.data(withJSONObject: iteratorSource, options: []).write(to: iteratorSourceURL)
        let loadedCursor = try XCTUnwrap(MFluxResumeIteratorCompat.loadState(from: iteratorSourceURL)?.cursor)

        let artifacts = try MFluxCheckpointCompatArtifactsWriter.write(
            checkpointURL: checkpointURL,
            step: 5,
            seed: 123,
            batchSize: 2,
            datasetCount: 4,
            loraAdapterFileName: checkpointURL.lastPathComponent,
            optimizerFileName: checkpointURL.lastPathComponent,
            iteratorCursor: loadedCursor,
            lossPoints: [],
            configSnapshot: [
                "model": "/tmp/model",
                "dataset_root": "/tmp/data",
            ]
        )

        let iterator = try XCTUnwrap(loadJSONObject(from: artifacts.iteratorURL) as? [String: Any])
        let rngState = try XCTUnwrap(iterator["rng_state"] as? [Any])
        XCTAssertEqual(rngState.count, 3)
        XCTAssertNil(iterator["zero_rng_state"])
    }

    func testFluxInjectorLoadsMfluxStyleWeightsAndExternalOptimizerState() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("adapter.safetensors")
        let optimizerURL = temp.appendingPathComponent("optimizer.safetensors")

        let downWeight = MLXArray([0.11 as Float, 0.12 as Float], [1, 2])
        let upWeight = MLXArray([0.21 as Float, 0.22 as Float], [2, 1])
        let downM = MLXArray([1.11 as Float, 1.12 as Float], [1, 2])
        let downV = MLXArray([2.11 as Float, 2.12 as Float], [1, 2])
        let upM = MLXArray([3.11 as Float, 3.12 as Float], [2, 1])
        let upV = MLXArray([4.11 as Float, 4.12 as Float], [2, 1])
        try MLX.save(
            arrays: [
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": downWeight,
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": upWeight,
            ],
            metadata: [:],
            url: adapterURL
        )
        try MLX.save(
            arrays: [
                "transformer.transformer_blocks.0.attn.to_q.lora_A.m": downM,
                "transformer.transformer_blocks.0.attn.to_q.lora_A.v": downV,
                "transformer.transformer_blocks.0.attn.to_q.lora_B.m": upM,
                "transformer.transformer_blocks.0.attn.to_q.lora_B.v": upV,
            ],
            metadata: [:],
            url: optimizerURL
        )

        let base = Linear(weight: MLXArray.zeros([2, 2], dtype: .float32), bias: nil)
        let layer = LoRALinear(base: base, rank: 1, alpha: 1.0, zeroInitUp: true)
        let updated = try Flux2LoRAInjector.loadWeights(
            from: adapterURL,
            into: ["transformer_blocks.0.attn.to_q": layer],
            optimizerStateURL: optimizerURL
        )
        XCTAssertEqual(updated, 1)
        XCTAssertEqual(MLX.sum(layer.loraDown).item(Float.self), MLX.sum(downWeight).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraUp).item(Float.self), MLX.sum(upWeight).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraDownM!).item(Float.self), MLX.sum(downM).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraDownV!).item(Float.self), MLX.sum(downV).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraUpM!).item(Float.self), MLX.sum(upM).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraUpV!).item(Float.self), MLX.sum(upV).item(Float.self), accuracy: 1e-6)
    }

    func testZImageInjectorLoadsMfluxStyleWeightsAndExternalOptimizerState() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let adapterURL = temp.appendingPathComponent("adapter.safetensors")
        let optimizerURL = temp.appendingPathComponent("optimizer.safetensors")

        let downWeight = MLXArray([0.31 as Float, 0.32 as Float], [1, 2])
        let upWeight = MLXArray([0.41 as Float, 0.42 as Float], [2, 1])
        let downM = MLXArray([5.11 as Float, 5.12 as Float], [1, 2])
        let downV = MLXArray([6.11 as Float, 6.12 as Float], [1, 2])
        let upM = MLXArray([7.11 as Float, 7.12 as Float], [2, 1])
        let upV = MLXArray([8.11 as Float, 8.12 as Float], [2, 1])
        try MLX.save(
            arrays: [
                "diffusion_model.layers.0.attention.to_q.lora_A.weight": downWeight,
                "diffusion_model.layers.0.attention.to_q.lora_B.weight": upWeight,
            ],
            metadata: [:],
            url: adapterURL
        )
        try MLX.save(
            arrays: [
                "diffusion_model.layers.0.attention.to_q.lora_A.m": downM,
                "diffusion_model.layers.0.attention.to_q.lora_A.v": downV,
                "diffusion_model.layers.0.attention.to_q.lora_B.m": upM,
                "diffusion_model.layers.0.attention.to_q.lora_B.v": upV,
            ],
            metadata: [:],
            url: optimizerURL
        )

        let base = Linear(weight: MLXArray.zeros([2, 2], dtype: .float32), bias: nil)
        let layer = LoRALinear(base: base, rank: 1, alpha: 1.0, zeroInitUp: true)
        let updated = try ZImageLoRAInjector.loadWeights(
            from: adapterURL,
            into: ["layers.0.attention.to_q": layer],
            optimizerStateURL: optimizerURL
        )
        XCTAssertEqual(updated, 1)
        XCTAssertEqual(MLX.sum(layer.loraDown).item(Float.self), MLX.sum(downWeight).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraUp).item(Float.self), MLX.sum(upWeight).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraDownM!).item(Float.self), MLX.sum(downM).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraDownV!).item(Float.self), MLX.sum(downV).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraUpM!).item(Float.self), MLX.sum(upM).item(Float.self), accuracy: 1e-6)
        XCTAssertEqual(MLX.sum(layer.loraUpV!).item(Float.self), MLX.sum(upV).item(Float.self), accuracy: 1e-6)
    }

    private func loadJSONObject(from url: URL) throws -> Any {
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data)
    }
}
