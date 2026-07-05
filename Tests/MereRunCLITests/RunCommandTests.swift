import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class RunCommandTests: XCTestCase {
    func testRootCommandExposesRunSubcommand() {
        let names = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(names.contains("run"))

        let runNames = Set(Run.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(runNames, Set(["list", "inspect"]))
    }

    func testRunListReportsWorkspaceArtifacts() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let runDir = temp.appendingPathComponent("runs", isDirectory: true)
            .appendingPathComponent("render", isDirectory: true)
        try writeMinimalRunDirectory(at: runDir)

        let planURL = temp.appendingPathComponent("render.plan.json")
        try writePlanHeader(to: planURL)

        let reportURL = temp.appendingPathComponent("report.json")
        let report = RunInspectionEnvelope(
            schemaVersion: 1,
            mereRunVersion: "test",
            command: ["image", "generate"],
            mode: .preflight,
            status: .ok,
            createdAt: Date(timeIntervalSince1970: 0),
            cwd: temp.path,
            summary: "fixture report",
            request: RunInspectionRequest(path: temp.path),
            result: RunInspectionResult(kind: "fixture", path: temp.path, runDirectory: nil, report: nil, plan: nil),
            diagnostics: [],
            actions: []
        )
        try StructuredRunOutput.encode(report).write(to: reportURL, atomically: true, encoding: .utf8)
        try "{}".write(to: temp.appendingPathComponent("ignored.json"), atomically: true, encoding: .utf8)

        let envelope = try RunList.parse(["--root", temp.path, "--json"])
            .makeListEnvelope(now: { Date(timeIntervalSince1970: 10) })

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.command, ["run", "list"])
        XCTAssertEqual(envelope.mode, .inspection)
        XCTAssertEqual(envelope.result.entryCount, 3)
        XCTAssertEqual(envelope.result.scannedDirectoryCount, 3)
        XCTAssertEqual(Set(envelope.result.entries.map(\.kind)), Set(["run_directory", "plan_file", "report_file"]))
        XCTAssertTrue(envelope.actions.contains { $0.id == "inspect-selected" })

        let runEntry = try XCTUnwrap(envelope.result.entries.first { $0.kind == "run_directory" })
        XCTAssertEqual(runEntry.relativePath, "runs/render")
        XCTAssertEqual(runEntry.state, "running")
        XCTAssertEqual(runEntry.format, ImageGenerationRunPlan.kind)
        XCTAssertEqual(runEntry.eventCount, 2)
        XCTAssertEqual(runEntry.createdAt, Date(timeIntervalSince1970: 0))
        XCTAssertEqual(runEntry.updatedAt, Date(timeIntervalSince1970: 2))
        XCTAssertTrue(runEntry.actions.contains { $0.id == "inspect" })
        XCTAssertTrue(runEntry.actions.contains { $0.id == "open" })

        let planEntry = try XCTUnwrap(envelope.result.entries.first { $0.kind == "plan_file" })
        XCTAssertEqual(planEntry.command, ["image", "generate"])
        XCTAssertEqual(planEntry.format, ImageGenerationRunPlan.kind)
        XCTAssertEqual(planEntry.createdAt, Date(timeIntervalSince1970: 0))
        XCTAssertEqual(planEntry.updatedAt, Date(timeIntervalSince1970: 0))

        let reportEntry = try XCTUnwrap(envelope.result.entries.first { $0.kind == "report_file" })
        XCTAssertEqual(reportEntry.command, ["image", "generate"])
        XCTAssertEqual(reportEntry.state, "ok")
        XCTAssertEqual(reportEntry.createdAt, Date(timeIntervalSince1970: 0))
        XCTAssertEqual(reportEntry.updatedAt, Date(timeIntervalSince1970: 0))

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(RunListEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded, envelope)
    }

    func testRunListReportsSinglePlanFileRoot() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let planURL = temp.appendingPathComponent("render.plan.json")
        try writePlanHeader(to: planURL)

        let envelope = try RunList.parse(["--root", planURL.path, "--json"])
            .makeListEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.entryCount, 1)
        XCTAssertEqual(envelope.result.entries.first?.kind, "plan_file")
        XCTAssertEqual(envelope.result.entries.first?.relativePath, ".")
    }

    func testRunListReportsLegacyPluginRunTimestamps() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let runDir = temp.appendingPathComponent("legacy-run", isDirectory: true)
        try writeLegacyPluginRunDirectory(at: runDir)

        let envelope = try RunList.parse(["--root", temp.path, "--json"])
            .makeListEnvelope(now: { Date(timeIntervalSince1970: 1) })

        let entry = try XCTUnwrap(envelope.result.entries.first)
        XCTAssertEqual(entry.status, .warning)
        XCTAssertEqual(entry.state, "running")
        XCTAssertEqual(entry.createdAt, Self.isoDate("2026-06-27T14:34:25Z"))
        XCTAssertEqual(entry.updatedAt, Self.isoDate("2026-06-27T14:34:45Z"))
    }

    func testRunListBlocksMissingRoot() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let missing = temp.appendingPathComponent("missing")
        let envelope = try RunList.parse(["--root", missing.path, "--json"])
            .makeListEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "run_list_root_not_found" })
    }

    func testRunInspectReportsRunDirectory() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let planURL = temp.appendingPathComponent("plan.json")
        try writePlanHeader(to: planURL)
        let actionsURL = temp.appendingPathComponent("actions.json")
        let action = DeclarativeAction(
            id: "start-generation",
            label: "Start generation",
            kind: .command,
            command: DeclarativeCommand(
                argv: ["mere.run", "image", "run-plan", planURL.path],
                cwd: temp.path,
                commandPath: ["image", "run-plan"]
            )
        )
        try StructuredRunOutput.encode([action]).write(to: actionsURL, atomically: true, encoding: .utf8)

        let outputURL = temp.appendingPathComponent("render.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: outputURL)
        try Data("weights".utf8).write(to: temp.appendingPathComponent("adapter.safetensors"))
        try """
        step,loss
        1,1.25
        2,0.75
        """.write(to: temp.appendingPathComponent("render.loss.csv"), atomically: true, encoding: .utf8)
        let samples = temp.appendingPathComponent("samples", isDirectory: true)
        try FileManager.default.createDirectory(at: samples, withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4E, 0x47])
            .write(to: samples.appendingPathComponent("step-2-sample.png"))
        let eventsURL = temp.appendingPathComponent("render.events.jsonl")
        try writeEvents(
            [
                LoRATrainingRunEvent(sequence: 0, createdAt: Date(timeIntervalSince1970: 1), type: "run_planned"),
                LoRATrainingRunEvent(sequence: 1, createdAt: Date(timeIntervalSince1970: 2), type: "run_started"),
            ],
            to: eventsURL
        )
        let manifest = LoRATrainingRunManifest(
            createdAt: Date(timeIntervalSince1970: 0),
            format: ImageGenerationRunPlan.kind,
            model: "local-zimage",
            isEdit: false,
            dataRoot: nil,
            dataRootRelative: nil,
            dataFingerprint: nil,
            checkpointFiles: [
                "plan": "plan.json",
                "actions": "actions.json",
                "events": "render.events.jsonl",
                "output_image": "render.png",
                "lora_adapter": "adapter.safetensors",
                "loss_csv": "render.loss.csv",
            ],
            step: 0,
            totalSteps: 6,
            seed: 42,
            rngState: nil,
            datasetFingerprint: nil,
            configFingerprint: "abc",
            configSnapshot: ["kind": ImageGenerationRunPlan.kind]
        )
        let manifestData = try StructuredRunOutput.encoder().encode(manifest)
        try manifestData.write(to: temp.appendingPathComponent("run.json"))

        let command = try RunInspect.parse([temp.path, "--json"])
        let envelope = command.makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 10) })

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.mode, .inspection)
        XCTAssertEqual(envelope.command, ["run", "inspect"])
        XCTAssertEqual(envelope.result.kind, "run_directory")
        XCTAssertEqual(envelope.result.runDirectory?.status, "running")
        XCTAssertEqual(envelope.result.runDirectory?.manifest?.format, ImageGenerationRunPlan.kind)
        XCTAssertEqual(envelope.result.runDirectory?.manifest?.progress, 0)
        XCTAssertEqual(envelope.result.runDirectory?.events.count, 2)
        XCTAssertEqual(envelope.result.runDirectory?.events.types, ["run_planned", "run_started"])
        XCTAssertEqual(envelope.result.runDirectory?.actions.first?.id, "start-generation")
        XCTAssertTrue(envelope.result.runDirectory?.artifacts.contains { $0.name == "render.png" && $0.exists } == true)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.lossPointCount, 2)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.firstStep, 1)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.latestStep, 2)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.latestLoss, 0.75)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.sampleImageCount, 1)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.adapterCount, 1)
        XCTAssertTrue(envelope.actions.contains { $0.id == "open-run-directory" })
        XCTAssertTrue(envelope.actions.contains { $0.id == "preflight-plan" })

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(RunInspectionEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded, envelope)
    }

    func testRunInspectReportsStructuredReportFile() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let reportURL = temp.appendingPathComponent("report.json")
        let report = RunInspectionEnvelope(
            schemaVersion: 1,
            mereRunVersion: "test",
            command: ["run", "inspect"],
            mode: .inspection,
            status: .ok,
            createdAt: Date(timeIntervalSince1970: 0),
            cwd: temp.path,
            summary: "fixture report",
            request: RunInspectionRequest(path: temp.path),
            result: RunInspectionResult(kind: "fixture", path: temp.path, runDirectory: nil, report: nil, plan: nil),
            diagnostics: [
                PreflightDiagnostic(id: "note", severity: .note, title: "Note", message: "Fixture note."),
            ],
            actions: [
                DeclarativeAction(id: "open-run-directory", label: "Open", kind: .openDirectory, path: temp.path),
            ]
        )
        try StructuredRunOutput.encode(report).write(to: reportURL, atomically: true, encoding: .utf8)

        let envelope = try RunInspect.parse([reportURL.path, "--json"])
            .makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.kind, "report_file")
        XCTAssertEqual(envelope.result.report?.command, ["run", "inspect"])
        XCTAssertEqual(envelope.result.report?.mode, .inspection)
        XCTAssertEqual(envelope.result.report?.diagnosticCount, 1)
        XCTAssertEqual(envelope.result.report?.actionCount, 1)
    }

    func testRunInspectReportsPlanFile() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let planURL = temp.appendingPathComponent("plan.json")
        try writePlanHeader(to: planURL)

        let envelope = try RunInspect.parse([planURL.path, "--json"])
            .makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.kind, "plan_file")
        XCTAssertEqual(envelope.result.plan?.kind, ImageGenerationRunPlan.kind)
        XCTAssertEqual(envelope.result.plan?.command, ["image", "generate"])
    }

    func testRunInspectReportsLegacyPluginRunManifest() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeLegacyPluginRunDirectory(at: temp)

        let envelope = try RunInspect.parse([temp.path, "--json"])
            .makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .warning)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "legacy_run_manifest_detected" })
        XCTAssertFalse(envelope.diagnostics.contains { $0.id == "run_manifest_decode_failed" })
        XCTAssertEqual(envelope.result.runDirectory?.status, "running")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.contractVersion, "mere.run/plugin-run.v1")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.runID, "br2049-style")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.pluginName, "mere-runpod")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.provider, "runpod")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.datasetPairCount, 18)
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.recipeID, "klein-style")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.command.first, "mere.run")
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.createdAt, Self.isoDate("2026-06-27T14:34:25Z"))
        XCTAssertEqual(envelope.result.runDirectory?.legacyManifest?.updatedAt, Self.isoDate("2026-06-27T14:34:45Z"))
        XCTAssertTrue(envelope.result.runDirectory?.artifacts.contains {
            $0.kind == "early-samples" && $0.name == "step150-sample.png"
        } == true)
        XCTAssertEqual(envelope.result.runDirectory?.metrics.sampleImageCount, 1)
    }

    func testRunInspectDoesNotReportManifestMissingWhenManifestDecodeFails() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try "{}".write(to: temp.appendingPathComponent("run.json"), atomically: true, encoding: .utf8)

        let envelope = try RunInspect.parse([temp.path, "--json"])
            .makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "run_manifest_decode_failed" })
        XCTAssertFalse(envelope.diagnostics.contains { $0.id == "run_manifest_missing" })
    }

    func testRunInspectBlocksMissingPath() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let missing = temp.appendingPathComponent("missing")
        let envelope = try RunInspect.parse([missing.path, "--json"])
            .makeInspectionEnvelope(now: { Date(timeIntervalSince1970: 1) })

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "path_missing" })
    }

    private func makeTempDir() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("RunCommandTests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func writePlanHeader(to url: URL) throws {
        let json = """
        {
          "schema_version": 1,
          "kind": "\(ImageGenerationRunPlan.kind)",
          "command": ["image", "generate"],
          "created_at": "1970-01-01T00:00:00Z",
          "cwd": "\(url.deletingLastPathComponent().path)"
        }
        """
        try json.write(to: url, atomically: true, encoding: .utf8)
    }

    private static func isoDate(_ value: String) -> Date? {
        ISO8601DateFormatter().date(from: value)
    }

    private func writeMinimalRunDirectory(at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)

        let planURL = url.appendingPathComponent("plan.json")
        try writePlanHeader(to: planURL)
        let action = DeclarativeAction(
            id: "start-generation",
            label: "Start generation",
            kind: .command,
            command: DeclarativeCommand(
                argv: ["mere.run", "image", "run-plan", planURL.path],
                cwd: url.path,
                commandPath: ["image", "run-plan"]
            )
        )
        try StructuredRunOutput.encode([action])
            .write(to: url.appendingPathComponent("actions.json"), atomically: true, encoding: .utf8)

        try Data([0x89, 0x50, 0x4E, 0x47])
            .write(to: url.appendingPathComponent("render.png"))
        try Data("weights".utf8)
            .write(to: url.appendingPathComponent("adapter.safetensors"))
        try """
        step,loss
        1,1.25
        2,0.75
        """.write(to: url.appendingPathComponent("render.loss.csv"), atomically: true, encoding: .utf8)
        let samples = url.appendingPathComponent("samples", isDirectory: true)
        try FileManager.default.createDirectory(at: samples, withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4E, 0x47])
            .write(to: samples.appendingPathComponent("step-2-sample.png"))
        try writeEvents(
            [
                LoRATrainingRunEvent(sequence: 0, createdAt: Date(timeIntervalSince1970: 1), type: "run_planned"),
                LoRATrainingRunEvent(sequence: 1, createdAt: Date(timeIntervalSince1970: 2), type: "run_started"),
            ],
            to: url.appendingPathComponent("render.events.jsonl")
        )
        let manifest = LoRATrainingRunManifest(
            createdAt: Date(timeIntervalSince1970: 0),
            format: ImageGenerationRunPlan.kind,
            model: "local-zimage",
            isEdit: false,
            dataRoot: nil,
            dataRootRelative: nil,
            dataFingerprint: nil,
            checkpointFiles: [
                "plan": "plan.json",
                "actions": "actions.json",
                "events": "render.events.jsonl",
                "output_image": "render.png",
                "lora_adapter": "adapter.safetensors",
                "loss_csv": "render.loss.csv",
            ],
            step: 0,
            totalSteps: 6,
            seed: 42,
            rngState: nil,
            datasetFingerprint: nil,
            configFingerprint: "abc",
            configSnapshot: ["kind": ImageGenerationRunPlan.kind]
        )
        let manifestData = try StructuredRunOutput.encoder().encode(manifest)
        try manifestData.write(to: url.appendingPathComponent("run.json"))
    }

    private func writeLegacyPluginRunDirectory(at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let samples = url.appendingPathComponent("early-samples", isDirectory: true)
        try FileManager.default.createDirectory(at: samples, withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4E, 0x47])
            .write(to: samples.appendingPathComponent("step150-sample.png"))
        let json = """
        {
          "artifacts": {
            "localDirectory": "\(url.appendingPathComponent("artifacts").path)",
            "remoteDirectory": "/workspace/run/artifacts"
          },
          "command": [
            "mere.run",
            "image",
            "train-lora"
          ],
          "contractVersion": "mere.run/plugin-run.v1",
          "createdAt": "2026-06-27T14:34:25+00:00",
          "dataset": {
            "pairCount": 18,
            "path": "/datasets/br2049"
          },
          "plugin": {
            "name": "mere-runpod"
          },
          "recipe": {
            "applyModel": "image-klein-9b",
            "id": "klein-style",
            "trainModel": "image-klein-base-9b"
          },
          "remote": {
            "gpu": "NVIDIA H100 80GB HBM3",
            "provider": "runpod"
          },
          "runId": "br2049-style",
          "status": "running",
          "updatedAt": "2026-06-27T14:34:45+00:00"
        }
        """
        try json.write(to: url.appendingPathComponent("run.json"), atomically: true, encoding: .utf8)
    }

    private func writeEvents(_ events: [LoRATrainingRunEvent], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let text = try events.map { event in
            String(data: try encoder.encode(event), encoding: .utf8) ?? ""
        }.joined(separator: "\n") + "\n"
        try text.write(to: url, atomically: true, encoding: .utf8)
    }
}
