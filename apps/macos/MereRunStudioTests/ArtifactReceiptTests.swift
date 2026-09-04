@testable import MereRunApp
import Foundation
import MereRunContract
import XCTest

/// The receipt path: the flags the app adds, the parser that reads the CLI's result line, the
/// precedence `ArtifactResolver` applies, and the roles that reach `Job.artifacts`.
@MainActor
final class ArtifactReceiptTests: XCTestCase {
    // MARK: Parsing

    func testValidReceiptDecodesPrimaryAndSidecarRoles() throws {
        let stdout = """
        Loading model…
        /out/render.png
        {"event":"result","exit":0,"outputs":[\
        {"kind":"image","path":"/out/render.png"},\
        {"kind":"json","path":"/out/render.prompt.json","role":"structured-prompt"}]}
        """

        let receipt = try XCTUnwrap(StudioRunReceipt.parse(stdout: stdout))
        XCTAssertEqual(receipt.event, "result")
        XCTAssertEqual(receipt.exit, 0)
        XCTAssertEqual(receipt.outputs.map(\.path), ["/out/render.png", "/out/render.prompt.json"])
        XCTAssertEqual(receipt.outputs.map(\.kind), ["image", "json"])
        XCTAssertEqual(receipt.outputs.map(\.role), [nil, "structured-prompt"])
    }

    func testMalformedAndAbsentReceiptsAreIgnored() {
        XCTAssertNil(StudioRunReceipt.parse(stdout: ""))
        XCTAssertNil(StudioRunReceipt.parse(stdout: "/out/render.png\nDone.\n"))
        // Truncated JSON, a wrong event, and a shape that is not a receipt at all.
        XCTAssertNil(StudioRunReceipt.parse(stdout: #"{"event":"result","exit":0,"outputs":[{"kind""#))
        XCTAssertNil(StudioRunReceipt.parse(stdout: #"{"event":"progress","stage":"denoising","step":1,"total_steps":4}"#))
        XCTAssertNil(StudioRunReceipt.parse(stdout: #"{"status":"result","output":"/out/render.mp4"}"#))
    }

    func testReceiptIsFoundAmongOtherJSONLinesAndAfterTrailingOutput() throws {
        let stdout = """
        {"event":"progress","stage":"denoising","step":3,"total_steps":4}
        {"status":"result","output":"/out/other.mp4"}
        {"event":"result","exit":0,"outputs":[{"kind":"audio","path":"/out/song.wav"}]}
        Wrote /out/song.wav
        {"note":"trailing"}
        """

        let receipt = try XCTUnwrap(StudioRunReceipt.parse(stdout: stdout))
        XCTAssertEqual(receipt.outputs.map(\.path), ["/out/song.wav"])
    }

    func testTheLastReceiptWinsWhenARunPrintsMoreThanOne() throws {
        let stdout = """
        {"event":"result","exit":0,"outputs":[{"kind":"image","path":"/out/first.png"}]}
        {"event":"result","exit":0,"outputs":[{"kind":"image","path":"/out/second.png"}]}
        """
        XCTAssertEqual(try XCTUnwrap(StudioRunReceipt.parse(stdout: stdout)).outputs.first?.path, "/out/second.png")
    }

    func testStrippingReceiptLinesLeavesTheRunsOwnOutput() {
        let stdout = """
        the quick brown fox
        {"event":"result","exit":0,"outputs":[{"kind":"text","path":"/out/transcript.txt"}]}
        """
        XCTAssertEqual(StudioRunReceipt.strippingReceiptLines(from: stdout), "the quick brown fox")
        XCTAssertEqual(StudioRunReceipt.strippingReceiptLines(from: "no receipt here"), "no receipt here")
    }

    // MARK: Precedence

    func testReceiptOutranksTheDeclaredOutputPathAndStdoutHeuristics() throws {
        let probe = StubFileProbe()
        probe.existingPaths = ["/out/expected.png", "/out/printed.png"]
        let resolver = ArtifactResolver(fileSystem: probe)
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a cat"
        draft.outputPath = "/out/expected.png"
        let stdout = """
        /out/printed.png
        {"event":"result","exit":0,"outputs":[\
        {"kind":"image","path":"/out/receipt.png"},\
        {"kind":"json","path":"/out/receipt.prompt.json","role":"structured-prompt"}]}
        """

        let resolution = resolver.resolve(
            template: template,
            draft: draft,
            expected: ArtifactResolver.expectedOutput(template: template, draft: draft),
            stdout: stdout
        )

        XCTAssertEqual(resolution.source, .receipt)
        XCTAssertEqual(resolution.primary?.path, "/out/receipt.png")
        XCTAssertEqual(resolution.sidecars.map(\.url.path), ["/out/receipt.prompt.json"])
        XCTAssertEqual(resolution.sidecars.map(\.sidecarRole), ["structured-prompt"])
        XCTAssertEqual(resolution.sidecars.first?.roleLabel, "Structured prompt")
    }

    func testTheDeclaredOutputPathOutranksStdoutHeuristics() throws {
        let probe = StubFileProbe()
        probe.existingPaths = ["/out/expected.png", "/out/printed.png"]
        let resolver = ArtifactResolver(fileSystem: probe)
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a cat"
        draft.outputPath = "/out/expected.png"

        let resolution = resolver.resolve(
            template: template,
            draft: draft,
            expected: ArtifactResolver.expectedOutput(template: template, draft: draft),
            stdout: "/out/printed.png\n"
        )

        XCTAssertEqual(resolution.source, .declaredOutput)
        XCTAssertEqual(resolution.primary?.path, "/out/expected.png")
    }

    func testProbingIsTheFallbackWhenNoReceiptAndNoDeclaredOutputLanded() throws {
        let probe = StubFileProbe()
        probe.existingPaths = ["/out/printed.png"]
        let resolver = ArtifactResolver(fileSystem: probe)
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a cat"
        draft.outputPath = "/out/never-written.png"

        let resolution = resolver.resolve(
            template: template,
            draft: draft,
            expected: ArtifactResolver.expectedOutput(template: template, draft: draft),
            stdout: "/out/printed.png\n"
        )

        XCTAssertEqual(resolution.source, .probe)
        XCTAssertEqual(resolution.primary?.path, "/out/printed.png")
    }

    func testAFailedRunWithNothingOnDiskResolvesToNoArtifacts() throws {
        let resolver = ArtifactResolver(fileSystem: StubFileProbe())
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.outputPath = "/out/never-written.png"

        let resolution = resolver.resolve(
            template: template,
            draft: draft,
            expected: ArtifactResolver.expectedOutput(template: template, draft: draft),
            stdout: "error: model not installed\n"
        )
        XCTAssertEqual(resolution, .empty)
    }

    func testLivePrimaryDetectionPrefersTheReceiptOverTheExpectedPath() {
        let probe = StubFileProbe()
        probe.existingPaths = ["/out/expected.png"]
        let resolver = ArtifactResolver(fileSystem: probe)
        let expected = URL(fileURLWithPath: "/out/expected.png")

        XCTAssertEqual(resolver.primaryOutput(expected: expected, stdout: "")?.path, "/out/expected.png")
        XCTAssertEqual(
            resolver.primaryOutput(
                expected: expected,
                stdout: #"{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/out/receipt.png"}]}"#
            )?.path,
            "/out/receipt.png"
        )
    }

    func testProbeDiscoveredSidecarsStillCarryTheirRole() throws {
        let temp = try makeTemporaryDirectory()
        let overlay = temp.appendingPathComponent("frame.png", isDirectory: false)
        let detections = temp.appendingPathComponent("frame.json", isDirectory: false)
        try Data("png".utf8).write(to: overlay)
        try Data("{}".utf8).write(to: detections)

        let probe = StubFileProbe()
        probe.existingPaths = [overlay.path]
        let resolver = ArtifactResolver(fileSystem: probe)
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionGround))
        var draft = template.defaultDraft()
        draft.inputPath = "/in/frame.png"
        draft.prompt = "a cat"
        draft.outputPath = overlay.path
        draft.visionJSONOutputPath = detections.path

        let resolution = resolver.resolve(
            template: template,
            draft: draft,
            expected: ArtifactResolver.expectedOutput(template: template, draft: draft),
            stdout: ""
        )

        XCTAssertEqual(resolution.source, .declaredOutput)
        XCTAssertEqual(resolution.primary?.path, overlay.path)
        XCTAssertEqual(resolution.sidecars.map(\.sidecarRole), ["detections"])
    }

    // MARK: Flags

    func testFlagsAreAddedOnlyForCapabilitiesThatDeclareThem() throws {
        for template in CommandCatalog.templates {
            var draft = template.defaultDraft()
            draft.preflight = false
            let arguments = template.arguments(from: draft)
            let flags = StudioMachineOutputFlags.arguments(
                template: template,
                draft: draft,
                appendingTo: arguments
            )
            let capabilityID = template.id.capabilityID
            XCTAssertEqual(
                flags.contains(StudioMachineOutputFlags.receipt),
                capabilityID.map(MereRunCapabilityCatalog.receiptCapabilityIDs.contains) ?? false,
                "\(template.id) receipt flag"
            )
            XCTAssertEqual(
                flags.contains(StudioMachineOutputFlags.progressJSON),
                capabilityID.map(MereRunCapabilityCatalog.progressJSONCapabilityIDs.contains) ?? false,
                "\(template.id) progress flag"
            )
            // Whatever the app adds, the contract must already declare it for that capability.
            if let capability = template.id.capability {
                let declared = Set(capability.options.map(\.flag))
                for flag in flags {
                    XCTAssertTrue(declared.contains(flag), "\(template.id) emits undeclared \(flag)")
                }
            }
        }
    }

    func testNoMachineOutputFlagIsEverAddedToAPreflightRun() throws {
        for template in CommandCatalog.templates where template.id.emitsRunReceipt {
            var draft = template.defaultDraft()
            draft.preflight = true
            let arguments = template.arguments(from: draft)
            XCTAssertEqual(
                StudioMachineOutputFlags.arguments(template: template, draft: draft, appendingTo: arguments),
                [],
                "\(template.id) must not combine --receipt with --preflight"
            )
            // Also guarded when a surface spells preflight with a different draft field.
            var other = template.defaultDraft()
            other.preflight = false
            XCTAssertEqual(
                StudioMachineOutputFlags.arguments(
                    template: template,
                    draft: other,
                    appendingTo: template.arguments(from: other) + ["--preflight"]
                ),
                []
            )
        }
    }

    func testFlagsAreNotDuplicatedWhenTheDraftAlreadyPassesThem() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a cat"
        draft.outputPath = "/out/render.png"
        draft.progressJSON = true
        let arguments = template.arguments(from: draft)

        XCTAssertTrue(arguments.contains(StudioMachineOutputFlags.progressJSON))
        XCTAssertEqual(
            StudioMachineOutputFlags.arguments(template: template, draft: draft, appendingTo: arguments),
            [StudioMachineOutputFlags.receipt]
        )
    }

    func testSubmittedRunCarriesTheFlagsWhileThePreviewStaysTheTypedCommand() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a cat"
        draft.outputPath = try makeTemporaryDirectory().appendingPathComponent("render.png").path

        XCTAssertTrue(
            controller.run(
                studio: StudioRunRequest(
                    mode: .createImage,
                    templateID: .imageGenerate,
                    template: template,
                    draft: draft
                )
            )
        )

        let launched = runner.starts[0].configuration.arguments
        XCTAssertEqual(Array(launched.suffix(2)), ["--receipt", "--progress-json"])
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        XCTAssertFalse(preview.contains("--receipt"))
        XCTAssertFalse(preview.contains("--progress-json"))
        XCTAssertFalse(controller.commandArguments(template: template, draft: draft).contains("--receipt"))
    }

    // MARK: Job lifecycle

    func testReceiptSettlesArtifactsWithRolesAndSkipsTheOutputWatch() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner, fileSystem: StubFileProbe())
        let temp = try makeTemporaryDirectory()
        let render = temp.appendingPathComponent("render.png", isDirectory: false)
        let structuredPrompt = temp.appendingPathComponent("render.prompt.json", isDirectory: false)
        let id = store.submit(try makeImageRequest(output: render))
        let job = try XCTUnwrap(store.job(id))

        XCTAssertTrue(job.expectsRunReceipt)
        XCTAssertNil(job.outputWatchTask, "a receipt run needs no filesystem poll")

        runner.starts[0].stdout("\(render.path)\n")
        runner.starts[0].stdout(
            #"{"event":"result","exit":0,"outputs":[{"kind":"image","path":"\#(render.path)"},"#
                + #"{"kind":"json","path":"\#(structuredPrompt.path)","role":"structured-prompt"}]}"# + "\n"
        )
        await settle()

        XCTAssertEqual(job.primaryArtifactURL?.path, render.path)
        XCTAssertFalse(
            job.log.lines.contains { $0.text.contains("\"event\":\"result\"") },
            "the receipt is transport, not console output"
        )

        runner.starts[0].termination(0)
        await settle()

        XCTAssertEqual(job.artifacts.map(\.url.path), [render.path, structuredPrompt.path])
        XCTAssertEqual(job.artifacts.map(\.role), [.primary, .sidecar])
        XCTAssertEqual(job.artifacts.map(\.sidecarRole), [nil, "structured-prompt"])
        let result = try XCTUnwrap(job.result)
        XCTAssertEqual(result.outputURL?.path, render.path)
        XCTAssertEqual(result.artifactRoles, [structuredPrompt.path: "structured-prompt"])
        XCTAssertEqual(result.outputText, render.path, "the receipt never reaches the library row")
    }

    func testCommandsWithoutAReceiptKeepTheOutputWatch() async throws {
        let runner = RecordingProcessRunner()
        let store = JobStore(processRunner: runner, fileSystem: StubFileProbe())
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionCaption))
        var draft = template.defaultDraft()
        draft.inputPath = "/in/frame.png"
        let id = store.submit(makeRequest(template: template, draft: draft))
        let job = try XCTUnwrap(store.job(id))

        XCTAssertFalse(job.expectsRunReceipt)
        XCTAssertNotNil(job.outputWatchTask)

        runner.starts[0].termination(0)
        await settle()
    }

    func testLibraryRowKeepsSidecarRolesAndStaysDecodableWithoutThem() throws {
        let roles = ["/out/render.prompt.json": "structured-prompt"]
        var item = makeLibraryItem(
            mode: .createImage,
            outputURL: URL(fileURLWithPath: "/out/render.png"),
            artifactURLs: [URL(fileURLWithPath: "/out/render.prompt.json")],
            roles: roles
        )
        XCTAssertEqual(item.artifactRoleLabel(for: URL(fileURLWithPath: "/out/render.prompt.json")), "Structured prompt")
        XCTAssertNil(item.artifactRoleLabel(for: URL(fileURLWithPath: "/out/render.png")))

        let encoded = try JSONEncoder().encode(item)
        let decoded = try JSONDecoder().decode(StudioLibraryItem.self, from: encoded)
        XCTAssertEqual(decoded.artifactRoles, roles)

        // A row written before receipts has no `artifactRoles` key at all.
        item.artifactRoles = nil
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(item)) as? [String: Any]
        )
        object.removeValue(forKey: "artifactRoles")
        let legacy = try JSONDecoder().decode(
            StudioLibraryItem.self,
            from: JSONSerialization.data(withJSONObject: object)
        )
        XCTAssertNil(legacy.artifactRoles)
    }

    /// The Analyze result column reads the sidecar the receipt named rather than the first
    /// artifact whose extension happens to match.
    func testAnalyzeReadsTheRoleNamedResultDocumentBeforeGuessingByExtension() {
        let overlay = URL(fileURLWithPath: "/out/frame.png")
        let timings = URL(fileURLWithPath: "/out/frame.timings.json")
        let detections = URL(fileURLWithPath: "/out/frame.detections.json")
        var item = makeLibraryItem(
            mode: .findObjects,
            outputURL: overlay,
            artifactURLs: [timings, detections],
            roles: [timings.path: "timings", detections.path: "detections"]
        )
        XCTAssertEqual(StudioAnalyzeDocumentSource.url(for: item), detections)

        // A row from before receipts still resolves by extension order.
        item.artifactRoles = nil
        XCTAssertEqual(StudioAnalyzeDocumentSource.url(for: item), timings)
    }

    func testEveryReceiptRoleTheCLIEmitsHasALabel() {
        for role in StudioArtifactRole.known {
            XCTAssertNotNil(StudioArtifactRole.label(for: role), role)
        }
        // A role from a newer CLI is humanized rather than dropped.
        XCTAssertEqual(StudioArtifactRole.label(for: "depth-map"), "Depth map")
        XCTAssertNil(StudioArtifactRole.label(for: nil))
    }

    /// The commands that write a file whether or not they were asked to, and that the app still
    /// offers no destination for, because the CLI picks the location itself. Each entry says why
    /// the app cannot name it.
    static let selfLocatingOutputTemplateIDs: [CommandTemplateID: String] = [
        .adapterPull: "Installs the adapter into the managed store; there is no destination option.",
        .imageRunPlan: "Materializes into the managed run store, named by `--materialize`.",
        .visionPose: "Writes `<image>_pose.json` beside the image unless `--json-output` moves it.",
    ]

    /// `CommandTemplate.outputKind` is the destination the app offers for `draft.outputPath`;
    /// the contract says whether the CLI has anywhere to put it. A command writes a file either
    /// because that is all it does (`kind` is `file` or `directory`) or because the caller can
    /// name a destination (`flag`) — a `text` or `service` command prints to stdout until it is
    /// asked to write.
    ///
    /// So the app may only offer a destination the contract declares, and every command that
    /// writes one unasked must either take that destination or be recorded above with its
    /// reason. Drift means either the contract or `CommandCatalog` is wrong.
    func testTheAppOffersExactlyTheDestinationsTheContractDeclares() throws {
        var offered: Set<CommandTemplateID> = []
        var unnamed: Set<CommandTemplateID> = []
        for template in CommandCatalog.templates {
            guard let declared = template.declaredOutputKind else { continue }
            if template.outputKind != .none, !template.producesOutputFile {
                offered.insert(template.id)
            }
            let writesUnasked = declared == .file || declared == .directory
            if writesUnasked, template.outputKind == .none {
                unnamed.insert(template.id)
            }
        }
        XCTAssertEqual(offered, [], "The app asks for an output path the CLI does not parse.")
        XCTAssertEqual(unnamed, Set(Self.selfLocatingOutputTemplateIDs.keys))

        // An exemption is only for a command that writes without being told where.
        for (id, reason) in Self.selfLocatingOutputTemplateIDs {
            let template = try XCTUnwrap(CommandCatalog.template(id: id))
            XCTAssertTrue(template.producesOutputFile, "\(id): \(reason)")
            XCTAssertEqual(template.outputKind, CommandOutputKind.none, "\(id): \(reason)")
            XCTAssertNotEqual(template.declaredOutputKind, .text, "\(id): \(reason)")
        }
    }

    /// A command that only writes when asked has nothing to wait for until it is asked: the
    /// resolver adopts the requested destination, and stays empty-handed without one.
    func testTheResolverOnlyExpectsAFileTheRunWasAskedToWrite() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .speechTranscribe))
        XCTAssertTrue(template.producesOutputFile)

        var draft = template.defaultDraft()
        draft.outputPath = ""
        XCTAssertNil(ArtifactResolver.expectedOutput(template: template, draft: draft))

        draft.outputPath = "/out/interview.txt"
        XCTAssertEqual(
            ArtifactResolver.expectedOutput(template: template, draft: draft),
            URL(fileURLWithPath: "/out/interview.txt")
        )

        // `model benchmark chat` parses no destination at all, so it offers no path to expect.
        let benchmark = try XCTUnwrap(CommandCatalog.template(id: .modelBenchmarkChat))
        XCTAssertFalse(benchmark.producesOutputFile)
        XCTAssertEqual(benchmark.outputKind, CommandOutputKind.none)
        XCTAssertNil(
            StudioOutputLocation.templateOutputPath(
                templateID: benchmark.id,
                title: benchmark.title,
                outputKind: benchmark.outputKind
            )
        )
    }

    // MARK: Helpers

    private func makeImageRequest(output: URL) throws -> JobRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a cat"
        draft.outputPath = output.path
        return makeRequest(template: template, draft: draft)
    }

    private func makeRequest(template: CommandTemplate, draft: CommandDraft) -> JobRequest {
        let arguments = template.arguments(from: draft)
        let launched = arguments + StudioMachineOutputFlags.arguments(
            template: template,
            draft: draft,
            appendingTo: arguments
        )
        return JobRequest(
            lane: .inference,
            template: template,
            draft: draft,
            requestID: UUID(),
            conversationID: nil,
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: launched,
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: false
            ),
            displayCommand: (["mere.run"] + arguments).shellQuoted()
        )
    }

    private func makeLibraryItem(
        mode: StudioMode,
        outputURL: URL,
        artifactURLs: [URL],
        roles: [String: String]
    ) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(),
            mode: mode,
            prompt: "a cat",
            inputURL: nil,
            outputURL: outputURL,
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run \(mode.rawValue)",
            outputText: nil,
            customTitle: nil,
            artifactURLs: artifactURLs,
            artifactRoles: roles
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ArtifactReceiptTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func settle() async {
        for _ in 0..<6 { await Task.yield() }
    }
}

/// The `--progress-json` convention, which every lane that streams progress now follows.
final class ProgressConventionTests: XCTestCase {
    func testZeroBasedStepsAndTheTerminalEventSpanTheWholeBar() throws {
        let first = try XCTUnwrap(StudioProgressParser.parse(event(stage: "denoising", step: 0, total: 4)))
        XCTAssertEqual(first.label, "Denoising")
        XCTAssertEqual(try XCTUnwrap(first.fractionCompleted), 0.25, accuracy: 0.001)
        XCTAssertEqual(first.detail, "Step 1 of 4")

        let terminal = try XCTUnwrap(StudioProgressParser.parse(event(stage: "denoising", step: 4, total: 4)))
        XCTAssertEqual(try XCTUnwrap(terminal.fractionCompleted), 1.0, accuracy: 0.001)
        XCTAssertEqual(terminal.detail, "Step 4 of 4")
    }

    func testIndeterminateStagesReportNoFractionSoSurfacesShowASpinner() throws {
        // `speech synthesize` streams tokens with no known length.
        let streaming = try XCTUnwrap(StudioProgressParser.parse(event(stage: "generating", step: 128, total: 0)))
        XCTAssertEqual(streaming.label, "Generating")
        XCTAssertNil(streaming.fractionCompleted)
        XCTAssertEqual(streaming.detail, "Step 128")

        let opening = try XCTUnwrap(StudioProgressParser.parse(event(stage: "loadingModel", step: 0, total: 0)))
        XCTAssertEqual(opening.label, "Loading model")
        XCTAssertNil(opening.fractionCompleted)
        XCTAssertNil(opening.detail)
    }

    func testEveryLanesStagesResolveToALabelledUpdate() throws {
        let stages: [(stage: String, label: String)] = [
            // image and video generation (`GenerationStage`)
            ("loadingModel", "Loading model"),
            ("loadingEncoder", "Loading encoder"),
            ("encodingText", "Encoding prompt"),
            ("encodingReferenceImages", "Encoding references"),
            ("loadingTransformer", "Loading transformer"),
            ("loadingLoRA", "Loading adapter"),
            ("loadingVAE", "Loading VAE"),
            ("denoising", "Denoising"),
            ("decoding", "Decoding"),
            ("saving", "Saving output"),
            // speech synthesis (`TTSStage`)
            ("preprocessingReference", "Preparing reference"),
            ("encodingReference", "Encoding reference"),
            ("buildingPrompt", "Building prompt"),
            ("tokenizing", "Tokenizing"),
            ("generating", "Generating"),
            // music generation milestones
            ("semantic", "Semantic frames"),
            // video sliding windows
            ("window", "Window"),
        ]
        for (stage, label) in stages {
            let progress = try XCTUnwrap(
                StudioProgressParser.parse(event(stage: stage, step: 1, total: 8)),
                stage
            )
            XCTAssertEqual(progress.label, label, stage)
            XCTAssertEqual(try XCTUnwrap(progress.fractionCompleted), 0.25, accuracy: 0.001, stage)
        }
    }

    func testAStageThisBuildDoesNotKnowIsStillRendered() throws {
        let progress = try XCTUnwrap(StudioProgressParser.parse(event(stage: "superResolving", step: 1, total: 2)))
        XCTAssertEqual(progress.label, "Super resolving")
        XCTAssertEqual(try XCTUnwrap(progress.fractionCompleted), 1.0, accuracy: 0.001)
    }

    func testNonProgressJSONLinesAreNotMistakenForProgress() {
        XCTAssertNil(StudioProgressParser.parse(#"{"event":"result","exit":0,"outputs":[]}"#))
        XCTAssertNil(StudioProgressParser.parse(#"{"status":"result","output":"/out/x.mp4"}"#))
        XCTAssertNil(StudioProgressParser.parse(#"{"event":"progress","stage":"","step":1,"total_steps":4}"#))
    }

    private func event(stage: String, step: Int, total: Int) -> String {
        #"{"event":"progress","stage":"\#(stage)","step":\#(step),"total_steps":\#(total)}"#
    }
}
