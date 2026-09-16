import ArgumentParser
import Foundation
import MereRunContract
import MereRunCore
import XCTest
@testable import MereRunCLI

final class VisionDepthCommandTests: XCTestCase {
    func testCatalogCheckpointChoicesMatchTheRuntime() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "vision.depth"))
        let checkpoint = try XCTUnwrap(capability.options.first { $0.flag == "--checkpoint" })
        XCTAssertEqual(checkpoint.choices, MarigoldV2DepthCheckpoint.allCases.map(\.rawValue))
        XCTAssertEqual(checkpoint.choices, MarigoldV2GenerationSettings.checkpointNames)
        XCTAssertTrue(capability.options.contains { $0.flag == "--receipt" })
    }

    func testParsesProductionOptions() throws {
        let command = try VisionDepth.parse([
            "/tmp/frame.png",
            "--output", "/tmp/depth",
            "--model", "/tmp/marigold",
            "--max-edge", "2048",
            "--checkpoint", "log-layered",
            "--json",
            "--receipt",
        ])
        XCTAssertEqual(command.input, "/tmp/frame.png")
        XCTAssertEqual(command.output, "/tmp/depth")
        XCTAssertEqual(command.model, "/tmp/marigold")
        XCTAssertEqual(command.maxEdge, 2_048)
        XCTAssertEqual(command.checkpoint, "log-layered")
        XCTAssertTrue(command.json)
        XCTAssertTrue(command.receipt)
        XCTAssertFalse(command.native)

        let request = try command.makeGenerationRequest()
        XCTAssertEqual(request.imageURL.path, "/tmp/frame.png")
        XCTAssertEqual(request.outputDirectory.path, "/tmp/depth")
        XCTAssertEqual(request.model, "/tmp/marigold")
        XCTAssertEqual(request.settings.configuration.checkpoint, .logLayered)
        XCTAssertEqual(request.settings.configuration.maximumEdge, 2_048)
    }

    func testCommandDoesNotReconstructOperationDefaults() throws {
        let request = try VisionDepth.parse(["/tmp/frame.png"]).makeGenerationRequest()
        XCTAssertEqual(request, MarigoldV2GenerationRequest(
            imageURL: URL(fileURLWithPath: "/tmp/frame.png"),
            outputDirectory: URL(fileURLWithPath: "/tmp/frame-depth", isDirectory: true),
            settings: try MarigoldV2GenerationSettings()
        ))
        XCTAssertNil(request.model)
        XCTAssertEqual(request.settings.configuration.checkpoint, MarigoldV2GenerationSettings.defaultCheckpoint)
        XCTAssertEqual(request.settings.configuration.maximumEdge, MarigoldV2GenerationSettings.defaultMaximumEdge)

        let native = try VisionDepth.parse(["/tmp/frame.png", "--native"]).makeGenerationRequest()
        XCTAssertNil(native.settings.configuration.maximumEdge)
    }

    func testDefaultsToThePaperCheckpoint() throws {
        XCTAssertEqual(try MarigoldV2GenerationSettings.checkpoint(named: nil), .logStage2)
        XCTAssertEqual(try MarigoldV2GenerationSettings.checkpoint(named: ""), .logStage2)
    }

    func testCheckpointSelectionIsCaseInsensitiveAndRejectsUnknownNames() throws {
        XCTAssertEqual(try MarigoldV2GenerationSettings.checkpoint(named: "Disparity-Base"), .disparityBase)
        XCTAssertThrowsError(try MarigoldV2GenerationSettings.checkpoint(named: "log-stage3")) { error in
            XCTAssertEqual(error as? MarigoldV2GenerationError, .unknownCheckpoint("log-stage3"))
        }
        let command = try VisionDepth.parse(["/tmp/frame.png", "--checkpoint", "log-stage3"])
        XCTAssertThrowsError(try command.makeGenerationRequest()) { error in
            XCTAssertEqual(error as? MarigoldV2GenerationError, .unknownCheckpoint("log-stage3"))
        }
    }

    func testDefaultOutputDirectoryIsDerivedFromTheInput() {
        let input = URL(fileURLWithPath: "/tmp/shot.001.png")
        XCTAssertEqual(
            VisionDepth.resolveOutputURL(nil, inputURL: input).path,
            "/tmp/shot.001-depth"
        )
        XCTAssertEqual(
            VisionDepth.resolveOutputURL("/tmp/elsewhere", inputURL: input).path,
            "/tmp/elsewhere"
        )
    }

    func testPlanReportsTheAlignedInferenceSize() throws {
        let plan = try Self.plan(width: 1_920, height: 1_080, maximumEdge: 1_024)
        let payload = VisionDepth.makePlan(plan)
        XCTAssertEqual(payload.status, "planned")
        XCTAssertEqual(payload.imageWidth, 1_920)
        XCTAssertEqual(payload.inferenceWidth, 1_024)
        XCTAssertEqual(payload.inferenceHeight, 576)
        XCTAssertEqual(payload.checkpoint, "log-stage2")
        XCTAssertEqual(payload.parameterization, "log")
        XCTAssertFalse(payload.seeThrough)
        XCTAssertEqual(payload.semantics, "affine-relative")
        XCTAssertEqual(payload.model, "vision-depth-marigold-v2")
        XCTAssertFalse(payload.managedModelInstalled)
    }

    func testPlanKeepsTheNativeResolutionWhenRequested() throws {
        let payload = VisionDepth.makePlan(try Self.plan(width: 1_920, height: 1_080, maximumEdge: nil))
        XCTAssertEqual(payload.inferenceWidth, 1_920)
        XCTAssertEqual(payload.inferenceHeight, 1_088)
    }

    func testPlanDoesNotAdvertiseCameraOrPointCloudArtifacts() throws {
        let payload = VisionDepth.makePlan(try Self.plan(width: 512, height: 512, maximumEdge: 1_024))
        // Marigold recovers depth up to an unknown scale and shift and never
        // estimates intrinsics, so projecting its output would imply a camera.
        XCTAssertFalse(payload.outputKinds.contains("camera-json"))
        XCTAssertFalse(payload.outputKinds.contains("point-cloud-ply"))
        XCTAssertTrue(payload.outputKinds.contains("depth-exr"))
    }

    func testNativeAndMaxEdgeCannotBeCombined() async throws {
        var command = try VisionDepth.parse(["/tmp/frame.png", "--native", "--max-edge", "512"])
        do {
            try await command.run()
            XCTFail("Expected --native and --max-edge to be rejected together")
        } catch let error as ValidationError {
            XCTAssertEqual(error.message, "--native and --max-edge cannot be combined")
        }
    }

    func testMaxEdgeBelowThePatchGridIsRejected() async throws {
        var command = try VisionDepth.parse(["/tmp/frame.png", "--max-edge", "8"])
        do {
            try await command.run()
            XCTFail("Expected a max edge below the patch grid to be rejected")
        } catch let error as ValidationError {
            XCTAssertEqual(error.message, "--max-edge must be at least 16")
        }
    }

    func testMissingInputIsAValidationError() async throws {
        var command = try VisionDepth.parse(["/definitely/missing/frame.png", "--dry-run"])
        do {
            try await command.run()
            XCTFail("Expected the missing input to be rejected")
        } catch let error as ValidationError {
            XCTAssertEqual(error.message, "Input image not found: /definitely/missing/frame.png")
        }
    }

    func testReceiptCannotBeCombinedWithDryRun() async throws {
        var command = try VisionDepth.parse(["/tmp/frame.png", "--dry-run", "--receipt"])
        do {
            try await command.run()
            XCTFail("Expected --receipt and --dry-run to be rejected together")
        } catch let error as ValidationError {
            XCTAssertEqual(error.message, RunReceipt.dryRunConflictMessage)
        }
    }

    func testReceiptListsTheDepthPreviewAndManifest() throws {
        let outputs = RunReceipt.depthOutputs(
            depth: URL(fileURLWithPath: "/out/shot-depth.exr"),
            preview: URL(fileURLWithPath: "/out/shot-depth.png"),
            manifest: URL(fileURLWithPath: "/out/shot-depth.json")
        )
        XCTAssertEqual(outputs.map(\.path), ["/out/shot-depth.exr", "/out/shot-depth.png", "/out/shot-depth.json"])
        XCTAssertEqual(outputs.map(\.kind), [.image, .image, .json])
        XCTAssertEqual(outputs.map(\.role), [nil, "preview", "manifest"])
        XCTAssertEqual(
            try RunReceipt(outputs: outputs).line(),
            #"{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/out/shot-depth.exr"},"#
                + #"{"kind":"image","path":"/out/shot-depth.png","role":"preview"},"#
                + #"{"kind":"json","path":"/out/shot-depth.json","role":"manifest"}]}"#
        )
    }

    private static func plan(width: Int, height: Int, maximumEdge: Int?) throws -> MarigoldV2GenerationPlan {
        let input = URL(fileURLWithPath: "/tmp/shot.png")
        let request = MarigoldV2GenerationRequest(
            imageURL: input,
            outputDirectory: VisionDepth.resolveOutputURL(nil, inputURL: input),
            settings: try MarigoldV2GenerationSettings(maximumEdge: maximumEdge, nativeResolution: maximumEdge == nil)
        )
        return try MarigoldV2GenerationPlan(
            request: request, imageWidth: width, imageHeight: height, managedModelInstalled: false
        )
    }
}
