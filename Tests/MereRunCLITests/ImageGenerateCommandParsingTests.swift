import XCTest
@testable import MereRunCLI
@testable import MereRunCore
import MediaIO

final class ImageGenerateCommandParsingTests: XCTestCase {
    func testDefaultManagedImageModelIsNano() {
        XCTAssertEqual(ImageGenerate.defaultManagedModelID, .zetaNano)
    }

    func testParsesHiDreamReferenceOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "place this subject in a studio",
            "--model", "image-hidream-o1-dev",
            "--ref-image", "/tmp/ref1.png",
            "--ref-image", "/tmp/ref2.png",
            "--keep-original-aspect",
            "--width", "1024",
            "--height", "768",
        ])

        XCTAssertEqual(cmd.prompt, "place this subject in a studio")
        XCTAssertEqual(cmd.model, "image-hidream-o1-dev")
        XCTAssertEqual(cmd.referenceImages, ["/tmp/ref1.png", "/tmp/ref2.png"])
        XCTAssertTrue(cmd.keepOriginalAspect)
        XCTAssertEqual(cmd.width, 1024)
        XCTAssertEqual(cmd.height, 768)
        XCTAssertNil(cmd.steps)
        XCTAssertNil(cmd.cfgScale)
    }

    func testParsesExplicitHiDreamStepAndCFGOverrides() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a brass camera",
            "--model", "image-hidream-o1",
            "--steps", "4",
            "--cfg", "1.0",
        ])

        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.cfgScale, 1.0)
    }

    func testParsesOrderedLoRAStackWithIndependentScales() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a studio portrait",
            "--lora", "flux2-dev-turbo-8step=1.0",
            "--lora", "/tmp/style.safetensors=0.65",
            "--lora-scale", "0.8",
        ])

        XCTAssertEqual(
            cmd.loraArguments,
            ["flux2-dev-turbo-8step=1.0", "/tmp/style.safetensors=0.65"]
        )
        XCTAssertEqual(cmd.lora, "flux2-dev-turbo-8step=1.0")
        XCTAssertEqual(
            try ImageGenerate.parseLoRAArguments(
                cmd.loraArguments,
                defaultScale: cmd.loraScale
            ),
            [
                .init(raw: "flux2-dev-turbo-8step=1.0", reference: "flux2-dev-turbo-8step", scale: 1),
                .init(raw: "/tmp/style.safetensors=0.65", reference: "/tmp/style.safetensors", scale: 0.65),
            ]
        )
    }

    func testParsesTurboSigmaScheduleWithOptionalTerminalZero() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a studio portrait",
            "--sigmas", "1, 0.6509, 0.4374, 0",
        ])

        XCTAssertEqual(
            try ImageGenerate.parseSigmaList(cmd.sigmaList),
            [1, 0.6509, 0.4374]
        )
    }

    func testParsesOptionalImageStrength() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a brass camera",
            "--strength", "0.35",
        ])

        XCTAssertEqual(cmd.strength, 0.35)
    }

    func testParsesMaskOutpaintAndFeatherOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "extend the garden",
            "--input", "/tmp/source.png",
            "--mask", "/tmp/mask.png",
            "--outpaint", "64,128,64,0",
            "--mask-feather", "16",
        ])

        XCTAssertEqual(cmd.mask, "/tmp/mask.png")
        XCTAssertEqual(cmd.outpaint, "64,128,64,0")
        XCTAssertEqual(cmd.maskFeather, 16)
        XCTAssertEqual(
            try ImageOutpaintInsets.parse(cmd.outpaint ?? ""),
            ImageOutpaintInsets(top: 64, right: 128, bottom: 64, left: 0)
        )
    }

    func testOutpaintRejectsInvalidInsetsAndMaskFeatherSmoothsBoundary() throws {
        XCTAssertThrowsError(try ImageOutpaintInsets.parse("64,-1,0,0"))
        XCTAssertThrowsError(try ImageOutpaintInsets.parse("0,0,0,0"))
        XCTAssertThrowsError(try ImageOutpaintInsets.parse("64,32"))

        let feathered = ImageEditPreparation.feather(
            [0, 0, 255, 255, 255],
            width: 5,
            height: 1,
            radius: 1
        )
        XCTAssertEqual(feathered.count, 5)
        XCTAssertGreaterThan(feathered[1], 0)
        XCTAssertLessThan(feathered[2], 255)
    }

    func testImageEditRestoresProtectedPixelsAndUsesGeneratedMaskedPixels() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-mask-test-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let inputURL = root.appendingPathComponent("input.png")
        let maskURL = root.appendingPathComponent("mask.png")
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 2,
                height: 1,
                rgba8: [255, 0, 0, 255, 0, 255, 0, 255]
            ),
            to: inputURL
        )
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 2,
                height: 1,
                rgba8: [0, 0, 0, 255, 255, 255, 255, 255]
            ),
            to: maskURL
        )
        let preparation = try ImageEditPreparation.make(
            inputURL: inputURL,
            maskURL: maskURL,
            outpaint: nil,
            width: 2,
            height: 1,
            featherPixels: 0
        )
        defer { preparation.cleanup() }
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 2,
                height: 1,
                rgba8: [0, 0, 255, 255, 0, 0, 255, 255]
            ),
            to: preparation.generationInputURL
        )

        try preparation.finish(generatedURL: preparation.generationInputURL)
        let result = try MediaImageIO.decode(preparation.generationInputURL)
        XCTAssertEqual(Array(result.rgba8[0..<4]), [255, 0, 0, 255])
        XCTAssertEqual(Array(result.rgba8[4..<8]), [0, 0, 255, 255])
    }

    func testOutpaintInsetsCreateExactEditableEdges() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-outpaint-test-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let inputURL = root.appendingPathComponent("input.png")
        try MediaImageIO.writePNG(
            try MediaImage(width: 1, height: 1, rgba8: [255, 0, 0, 255]),
            to: inputURL
        )
        let preparation = try ImageEditPreparation.make(
            inputURL: inputURL,
            maskURL: nil,
            outpaint: ImageOutpaintInsets(top: 0, right: 1, bottom: 0, left: 1),
            width: 3,
            height: 1,
            featherPixels: 0
        )
        defer { preparation.cleanup() }

        XCTAssertEqual(preparation.editMask, [255, 0, 255])
        XCTAssertEqual(Array(preparation.baseImage.rgba8[4..<8]), [255, 0, 0, 255])
    }

    func testKleinTreatsInputAsReferenceImage() {
        let inputURL = URL(fileURLWithPath: "/tmp/source.png")
        let resolved = ImageGenerate.resolveConditioningInputs(
            family: .klein,
            inputImage: inputURL,
            referenceImages: [],
            strength: nil
        )

        XCTAssertNil(resolved.inputImage)
        XCTAssertEqual(resolved.referenceImages, [inputURL])
        XCTAssertEqual(resolved.strength, 0.75)
        XCTAssertEqual(resolved.referenceStrength, 0.75)
    }

    func testKleinReferenceImagesDefaultToCleanReferenceStrength() {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let resolved = ImageGenerate.resolveConditioningInputs(
            family: .klein,
            inputImage: nil,
            referenceImages: [refURL],
            strength: nil
        )

        XCTAssertNil(resolved.inputImage)
        XCTAssertEqual(resolved.referenceImages, [refURL])
        XCTAssertEqual(resolved.strength, 0.75)
        XCTAssertEqual(resolved.referenceStrength, 0.0)
    }

    func testKleinReferenceImagesHonorExplicitStrength() {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let resolved = ImageGenerate.resolveConditioningInputs(
            family: .klein,
            inputImage: nil,
            referenceImages: [refURL],
            strength: 0.35
        )

        XCTAssertNil(resolved.inputImage)
        XCTAssertEqual(resolved.referenceImages, [refURL])
        XCTAssertEqual(resolved.strength, 0.35)
        XCTAssertEqual(resolved.referenceStrength, 0.35)
    }

    func testNonKleinPreservesInputImageSemantics() {
        let inputURL = URL(fileURLWithPath: "/tmp/source.png")
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let resolved = ImageGenerate.resolveConditioningInputs(
            family: .hidream,
            inputImage: inputURL,
            referenceImages: [refURL],
            strength: nil
        )

        XCTAssertEqual(resolved.inputImage, inputURL)
        XCTAssertEqual(resolved.referenceImages, [refURL])
        XCTAssertEqual(resolved.strength, 0.75)
        XCTAssertEqual(resolved.referenceStrength, 0.0)
    }

    func testParsesStructuredPromptOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a knight and a white horse",
            "--model", "image-ideogram4-sdnq-uint4",
            "--structured-prompt",
            "--structured-prompt-model", "text-chat-q36-nano",
            "--structured-prompt-model-root", "/tmp/q36",
            "--structured-prompt-max-tokens", "3072",
            "--structured-prompt-output", "/tmp/prompt.json",
        ])

        XCTAssertTrue(cmd.structuredPrompt)
        XCTAssertEqual(cmd.structuredPromptModel, "text-chat-q36-nano")
        XCTAssertEqual(cmd.structuredPromptModelRoot, "/tmp/q36")
        XCTAssertEqual(cmd.structuredPromptMaxTokens, 3_072)
        XCTAssertEqual(cmd.structuredPromptOutput, "/tmp/prompt.json")
    }

    func testParsesJSONPromptAlias() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "editorial product photo",
            "--json-prompt",
        ])

        XCTAssertTrue(cmd.structuredPrompt)
        XCTAssertEqual(cmd.structuredPromptModel, "text-chat-gemma4-12b-4bit")
    }

    func testParsesKreaConditioningRebalanceOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a printed diner ticket",
            "--model", "image-krea2-turbo",
            "--krea-conditioning-multiplier", "1.0",
            "--krea-conditioning-layer-weights", "1,1,1,1,1,1,1,2.5,5,1.1,4,1",
        ])

        XCTAssertEqual(cmd.kreaConditioningMultiplier, 1.0)
        XCTAssertEqual(cmd.kreaConditioningLayerWeights, "1,1,1,1,1,1,1,2.5,5,1.1,4,1")

        let rebalance = try ImageGenerate.resolveKreaConditioningRebalance(
            multiplier: cmd.kreaConditioningMultiplier,
            layerWeights: cmd.kreaConditioningLayerWeights
        )
        XCTAssertEqual(rebalance?.multiplier, 1.0)
        XCTAssertEqual(rebalance?.layerWeights, [1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1, 4, 1])
    }

    func testParsesKreaBaseQuantizationBits() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a printed diner ticket",
            "--model", "image-krea2-turbo",
            "--krea-base-quantization-bits", "4",
        ])

        XCTAssertEqual(cmd.kreaBaseQuantizationBits, 4)
        let envelope = cmd.makePreflightEnvelope(outputURL: URL(fileURLWithPath: "/tmp/krea.png"))
        XCTAssertEqual(envelope.request.kreaBaseQuantizationBits, 4)
        XCTAssertEqual(envelope.result.runPlan.arguments.kreaBaseQuantizationBits, 4)
        XCTAssertTrue(envelope.result.runPlan.arguments.generateArguments().contains("--krea-base-quantization-bits"))
    }

    func testKreaConditioningLayerWeightsAcceptSemicolonsAndWhitespace() throws {
        let weights = try ImageGenerate.parseKreaConditioningLayerWeights(" 1 ; 2.5, 3 ")
        XCTAssertEqual(weights, [1, 2.5, 3])
    }

    func testKreaConditioningLayerWeightsRejectInvalidValues() {
        XCTAssertThrowsError(try ImageGenerate.parseKreaConditioningLayerWeights("1,nope,3"))
    }

    func testParsesImageGenerationPreflightJSONOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a clean product render",
            "--preflight",
            "--json",
        ])

        XCTAssertTrue(cmd.preflight)
        XCTAssertTrue(cmd.json)
    }

    func testParsesProgressJSONFlag() throws {
        let defaulted = try ImageGenerate.parse([
            "--prompt", "a clean product render",
        ])
        XCTAssertFalse(defaulted.progressJson)

        let cmd = try ImageGenerate.parse([
            "--prompt", "a clean product render",
            "--progress-json",
            "--quiet",
        ])
        XCTAssertTrue(cmd.progressJson)
        XCTAssertTrue(cmd.quiet)
    }

    func testImageGenerationPreflightReportsReadyLocalRequest() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-zimage", family: .zimage, to: model)

        let input = temp.appendingPathComponent("input.png")
        let reference = temp.appendingPathComponent("reference.png")
        let lora = temp.appendingPathComponent("style.safetensors")
        try Data("input".utf8).write(to: input)
        try Data("reference".utf8).write(to: reference)
        try Data("lora".utf8).write(to: lora)

        let output = temp.appendingPathComponent("render.png")
        let cmd = try ImageGenerate.parse([
            "--prompt", "turn this into a matte catalog image",
            "--model", model.path,
            "--output", output.path,
            "--input", input.path,
            "--ref-image", reference.path,
            "--lora", lora.path,
            "--lora-scale", "0.8",
            "--steps", "6",
            "--seed", "42",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.command, ["image", "generate"])
        XCTAssertEqual(envelope.result.model.kind, "local_path")
        XCTAssertEqual(envelope.result.model.family, "zimage")
        XCTAssertEqual(envelope.result.output.path, output.path)
        XCTAssertEqual(envelope.result.inputs.inputImage?.path, input.path)
        XCTAssertEqual(envelope.result.inputs.referenceImages.first?.path, reference.path)
        XCTAssertEqual(envelope.result.lora?.path, lora.path)
        XCTAssertEqual(envelope.result.lora?.scale, 0.8)
        XCTAssertEqual(envelope.result.plan.effectiveSteps, 6)
        XCTAssertEqual(envelope.result.plan.inputMode, "image_to_image_with_references")
        XCTAssertEqual(envelope.result.runPlan.kind, ImageGenerationRunPlan.kind)
        XCTAssertEqual(envelope.result.runPlan.command, ["image", "generate"])
        XCTAssertEqual(envelope.result.runPlan.arguments.prompt, "turn this into a matte catalog image")
        XCTAssertEqual(envelope.result.runPlan.arguments.output, output.path)
        XCTAssertEqual(envelope.result.runPlan.arguments.model, model.path)
        XCTAssertEqual(envelope.result.runPlan.arguments.steps, 6)
        XCTAssertEqual(envelope.result.runPlan.arguments.seed, 42)
        XCTAssertEqual(envelope.result.runPlan.arguments.executableArgv().prefix(3), ["mere.run", "image", "generate"])

        let startGeneration = try XCTUnwrap(envelope.actions.first { $0.id == "start-generation" })
        XCTAssertTrue(startGeneration.enabled)
        XCTAssertEqual(startGeneration.command?.argv.prefix(5), [
            "mere.run",
            "image",
            "generate",
            "--prompt",
            "turn this into a matte catalog image",
        ])
        XCTAssertTrue(startGeneration.command?.argv.contains(output.path) == true)

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(ImageGenerationPreflightEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded.status, .ok)
        XCTAssertEqual(decoded.result.model.family, "zimage")
        XCTAssertFalse(decoded.result.structuredPrompt.backend.contains("default device"))

        let encodedPlan = try StructuredRunOutput.encode(envelope.result.runPlan)
        let planURL = temp.appendingPathComponent("generate.plan.json")
        try Data(encodedPlan.utf8).write(to: planURL)

        let runPlanCommand = try ImageRunPlan.parse([
            planURL.path,
            "--preflight",
            "--json",
        ])
        let loadedWorkflowPlan = try runPlanCommand.loadWorkflowPlan()
        XCTAssertEqual(loadedWorkflowPlan, .generate(envelope.result.runPlan))

        let commandFromPlan = try runPlanCommand.makeGenerateCommand(from: envelope.result.runPlan)
        XCTAssertEqual(commandFromPlan.prompt, "turn this into a matte catalog image")
        XCTAssertEqual(commandFromPlan.output, output.path)
        XCTAssertEqual(commandFromPlan.model, model.path)
        XCTAssertEqual(commandFromPlan.input, input.path)
        XCTAssertEqual(commandFromPlan.referenceImages, [reference.path])
        XCTAssertEqual(commandFromPlan.lora, lora.path)
        XCTAssertEqual(commandFromPlan.loraScale, 0.8)
        XCTAssertEqual(commandFromPlan.steps, 6)
        XCTAssertEqual(commandFromPlan.seed, 42)
        XCTAssertFalse(commandFromPlan.preflight)
        XCTAssertFalse(commandFromPlan.json)
    }

    func testImageGenerationPreflightRoundTripsStackedLocalLoRAs() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "image-flux2-dev", family: .klein, to: model)
        let turbo = temp.appendingPathComponent("turbo.safetensors")
        let style = temp.appendingPathComponent("style.safetensors")
        try Data("turbo".utf8).write(to: turbo)
        try Data("style".utf8).write(to: style)

        let command = try ImageGenerate.parse([
            "--prompt", "TRIGGER_TOKEN a studio portrait",
            "--model", model.path,
            "--lora", "\(turbo.path)=1.0",
            "--lora", "\(style.path)=0.7",
            "--preflight",
            "--json",
        ])
        let envelope = command.makePreflightEnvelope(
            outputURL: temp.appendingPathComponent("stacked.png")
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.loras.map(\.path), [turbo.path, style.path])
        XCTAssertEqual(envelope.result.loras.map(\.scale), [1, 0.7])
        XCTAssertEqual(
            envelope.result.runPlan.arguments.loras,
            ["\(turbo.path)=1.0", "\(style.path)=0.7"]
        )

        let runPlan = try ImageRunPlan.parse(["/tmp/unused-plan.json"])
        let replay = try runPlan.makeGenerateCommand(from: envelope.result.runPlan)
        XCTAssertEqual(
            replay.loraArguments,
            ["\(turbo.path)=1.0", "\(style.path)=0.7"]
        )
    }

    func testFlux1PreflightAcceptsStackedLoRAsAndUsesManifestDefaults() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let manifest = MereRunModelManifest(
            id: Flux1Resources.modelID,
            engine: .flux1,
            family: .flux1,
            variant: .standard,
            precision: .bf16,
            defaults: .init(steps: 28, cfg: 3.5),
            supports: [.txt2img, .loraInference]
        )
        try manifest.write(to: model)
        let fast = temp.appendingPathComponent("fast.safetensors")
        let style = temp.appendingPathComponent("style.safetensors")
        try Data("fast".utf8).write(to: fast)
        try Data("style".utf8).write(to: style)

        let command = try ImageGenerate.parse([
            "--prompt", "a studio portrait",
            "--model", model.path,
            "--lora", "\(fast.path)=0.9",
            "--lora", "\(style.path)=0.65",
            "--preflight",
            "--json",
        ])
        let envelope = command.makePreflightEnvelope(
            outputURL: temp.appendingPathComponent("flux1.png")
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.model.family, MereRunModelManifest.Family.flux1.rawValue)
        XCTAssertEqual(envelope.result.plan.effectiveSteps, 28)
        XCTAssertEqual(envelope.result.plan.effectiveCFGScale, 3.5)
        XCTAssertEqual(envelope.result.loras.map(\.scale), [0.9, 0.65])
        XCTAssertFalse(envelope.diagnostics.contains { $0.id == "lora_stack_model_unsupported" })
    }

    func testFlux2DevTurboPreflightAppliesPublishedRecipe() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "image-flux2-dev", family: .klein, to: model)
        let command = try ImageGenerate.parse([
            "--prompt", "a studio portrait",
            "--model", model.path,
            "--lora", ManagedAdapterCatalog.flux2DevTurboEightStepID,
            "--preflight",
            "--json",
        ])
        let envelope = command.makePreflightEnvelope(
            outputURL: temp.appendingPathComponent("turbo.png")
        )

        XCTAssertEqual(envelope.result.plan.effectiveSteps, Flux2DevTurboRecipe.steps)
        XCTAssertEqual(envelope.result.plan.effectiveCFGScale, Flux2DevTurboRecipe.guidanceScale)
        XCTAssertEqual(envelope.result.plan.effectiveSigmas, Flux2DevTurboRecipe.sigmas)
        let turbo = try XCTUnwrap(envelope.result.loras.first)
        XCTAssertEqual(turbo.requested, ManagedAdapterCatalog.flux2DevTurboEightStepID)
        if turbo.exists {
            XCTAssertEqual(envelope.status, .ok)
            XCTAssertFalse(envelope.diagnostics.contains { $0.id == "managed_lora_missing" })
        } else {
            XCTAssertEqual(envelope.status, .blocked)
            XCTAssertTrue(envelope.diagnostics.contains { $0.id == "managed_lora_missing" })
            XCTAssertTrue(envelope.actions.contains { $0.id == "pull-adapter-1" })
        }
    }

    func testImageRunPlanMaterializesGenerationRunDirectory() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-zimage", family: .zimage, to: model)

        let output = temp.appendingPathComponent("outside", isDirectory: true)
            .appendingPathComponent("render.png")
        let structuredPromptOutput = temp.appendingPathComponent("outside", isDirectory: true)
            .appendingPathComponent("prompt.json")
        let generate = try ImageGenerate.parse([
            "--prompt", "a quiet product photo",
            "--model", model.path,
            "--output", output.path,
            "--steps", "6",
            "--seed", "42",
            "--structured-prompt",
            "--structured-prompt-output", structuredPromptOutput.path,
        ])
        let preflight = generate.makePreflightEnvelope(
            outputURL: output,
            now: { Date(timeIntervalSince1970: 0) }
        )
        let sourcePlanURL = temp.appendingPathComponent("source.plan.json")
        try StructuredRunOutput.encode(preflight.result.runPlan)
            .write(to: sourcePlanURL, atomically: true, encoding: .utf8)

        let runDirectory = temp.appendingPathComponent("runs", isDirectory: true)
            .appendingPathComponent("render", isDirectory: true)
        let command = try ImageRunPlan.parse([
            sourcePlanURL.path,
            "--materialize",
            runDirectory.path,
            "--json",
        ])
        let envelope = try command.materializeEnvelope(
            plan: preflight.result.runPlan,
            runDirectory: runDirectory.path,
            now: { Date(timeIntervalSince1970: 10) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.mode, .materialize)
        XCTAssertEqual(URL(fileURLWithPath: envelope.request.planFile).lastPathComponent, "source.plan.json")
        XCTAssertEqual(URL(fileURLWithPath: envelope.result.runDirectory).lastPathComponent, "render")
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-generation" })
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "output_relocated" })
        XCTAssertEqual(URL(fileURLWithPath: try XCTUnwrap(envelope.result.structuredPromptOutputPath)).lastPathComponent, "prompt.json")

        let materializedPlanURL = URL(fileURLWithPath: envelope.result.planPath)
        let materializedPlan = try ImageGenerationRunPlan.decode(from: materializedPlanURL)
        XCTAssertEqual(materializedPlan.arguments.output, envelope.result.outputPath)
        XCTAssertEqual(materializedPlan.arguments.structuredPromptOutput, envelope.result.structuredPromptOutputPath)

        let commandFromPlan = try command.makeGenerateCommand(from: materializedPlan)
        XCTAssertEqual(commandFromPlan.output, envelope.result.outputPath)
        XCTAssertEqual(commandFromPlan.structuredPromptOutput, envelope.result.structuredPromptOutputPath)

        let decoder = JSONDecoder()
        let actionsURL = URL(fileURLWithPath: envelope.result.actionsPath)
        let actions = try decoder.decode([DeclarativeAction].self, from: Data(contentsOf: actionsURL))
        let startGeneration = try XCTUnwrap(actions.first { $0.id == "start-generation" })
        XCTAssertEqual(startGeneration.command?.argv, ["mere.run", "image", "run-plan", materializedPlanURL.path])
        XCTAssertFalse(actions.contains { $0.id == "visualize-run" })

        let manifestURL = URL(fileURLWithPath: envelope.result.runManifestPath)
        let manifest = try LoRATrainingRunManifest.decode(from: Data(contentsOf: manifestURL))
        XCTAssertEqual(manifest.format, ImageGenerationRunPlan.kind)
        XCTAssertEqual(manifest.step, 0)
        XCTAssertEqual(manifest.totalSteps, 6)
        XCTAssertEqual(manifest.seed, 42)
        XCTAssertEqual(manifest.checkpointFiles["plan"], "plan.json")
        XCTAssertEqual(manifest.checkpointFiles["actions"], "actions.json")
        XCTAssertEqual(manifest.checkpointFiles["output_image"], "render.png")
        XCTAssertEqual(manifest.checkpointFiles["structured_prompt"], "prompt.json")
        XCTAssertEqual(manifest.configSnapshot?["input_mode"], "text_to_image")

        let eventsURL = URL(fileURLWithPath: envelope.result.eventsPath)
        let events = try LoRATrainingRunEvent.load(from: eventsURL)
        XCTAssertEqual(events.first?.type, "run_planned")
        XCTAssertEqual(events.first?.path, envelope.result.outputPath)

        let viewer = LoRATrainingRunViewer(runDirectoryURL: URL(fileURLWithPath: envelope.result.runDirectory))
        let snapshot = try viewer.snapshot()
        XCTAssertEqual(snapshot.status, "planned")
        XCTAssertEqual(snapshot.events.map(\.type), ["run_planned"])
        XCTAssertTrue(snapshot.artifacts.contains { $0.kind == "root" && $0.name == "plan.json" })
        XCTAssertTrue(snapshot.artifacts.contains { $0.kind == "root" && $0.name == "actions.json" })

        let eventLogger = try XCTUnwrap(
            commandFromPlan.makeRunEventLoggerIfNeeded(
                outputURL: URL(fileURLWithPath: envelope.result.outputPath),
                modelRoot: model,
                modelManifest: MereRunModelManifest(
                    id: "local-zimage",
                    family: .zimage,
                    variant: .distilled,
                    precision: .bf16,
                    supports: []
                ),
                effectiveSteps: 6,
                effectiveCFGScale: 1.0,
                effectiveSigmaShift: nil,
                inputMode: "text_to_image"
            )
        )
        XCTAssertEqual(try LoRATrainingRunEvent.load(from: eventsURL).map(\.type), ["run_planned", "run_started"])
        XCTAssertEqual(try viewer.snapshot().status, "running")

        try eventLogger.record(
            type: "run_finished",
            stage: "finished",
            step: 6,
            totalSteps: 6,
            fraction: 1,
            path: envelope.result.outputPath
        )
        XCTAssertEqual(try viewer.snapshot().status, "finished")
    }

    func testImageGenerationPreflightBlocksMissingInputImage() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-zimage", family: .zimage, to: model)

        let missingInput = temp.appendingPathComponent("missing.png")
        let output = temp.appendingPathComponent("render.png")
        let cmd = try ImageGenerate.parse([
            "--prompt", "clean up this image",
            "--model", model.path,
            "--output", output.path,
            "--input", missingInput.path,
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "input_image_missing" })

        let startGeneration = try XCTUnwrap(envelope.actions.first { $0.id == "start-generation" })
        XCTAssertFalse(startGeneration.enabled)
        XCTAssertEqual(startGeneration.disabledReason, "Resolve hard blockers first.")
    }

    func testImageGenerationResourceBlockerDisablesAnOtherwiseReadyRequest() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-zimage", family: .zimage, to: model)
        let command = try ImageGenerate.parse(["--prompt", "test", "--model", model.path])
        let resources = MachineInferencePreflight.diagnostics(
            arguments: ["mere.run", "image", "generate", "--model", model.path],
            host: MachineInferenceHostSnapshot(
                physicalMemoryBytes: 8_589_934_592, availableMemoryBytes: 4_294_967_296,
                memoryPressure: .nominal, availableDiskBytes: 100_000_000_000
            )
        )
        let envelope = command.makePreflightEnvelope(
            outputURL: temp.appendingPathComponent("render.png"), resourceDiagnostics: resources
        )
        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.result.model.installed)
        XCTAssertEqual(envelope.diagnostics.map(\.id), ["machine_insufficient_memory"])
        XCTAssertFalse(try XCTUnwrap(envelope.actions.first { $0.id == "start-generation" }).enabled)
    }

    func testImageGenerationPreflightBlocksUnsupportedModelFamily() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-gemma", family: .gemma, to: model)
        let output = temp.appendingPathComponent("render.png")

        let cmd = try ImageGenerate.parse([
            "--prompt", "a clean product render",
            "--model", model.path,
            "--output", output.path,
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "model_family_unsupported" })
    }

    func testImageGenerationPreflightAcceptsQwenImageEditAndUsesManifestDefaults() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(
            id: "local-qwen-edit",
            family: .qwen,
            engine: .qwenImageEdit,
            defaults: .init(steps: 40, cfg: 4),
            to: model
        )
        let input = temp.appendingPathComponent("input.png")
        try Data("input".utf8).write(to: input)
        let output = temp.appendingPathComponent("render.png")

        let cmd = try ImageGenerate.parse([
            "--prompt", "remove the truck",
            "--model", model.path,
            "--input", input.path,
            "--output", output.path,
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.model.family, "qwen")
        XCTAssertEqual(envelope.result.plan.effectiveSteps, 40)
        XCTAssertEqual(envelope.result.plan.effectiveCFGScale, 4)
        XCTAssertEqual(envelope.result.plan.inputMode, "image_to_image")
        XCTAssertTrue(try XCTUnwrap(envelope.actions.first { $0.id == "start-generation" }).enabled)
    }

    func testImageGenerationPreflightRejectsNonImageQwenEngine() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(
            id: "local-qwen-text",
            family: .qwen,
            engine: .qwen35HybridMoE,
            to: model
        )
        let output = temp.appendingPathComponent("render.png")
        let cmd = try ImageGenerate.parse([
            "--prompt", "a clean product render",
            "--model", model.path,
            "--output", output.path,
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            outputURL: output,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "model_family_unsupported" })
    }

    func testStructuredPromptAdapterValidatesPaperSchema() throws {
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(Self.validStructuredCaptionJSON)

        XCTAssertEqual(caption.shortDescription, "A knight riding a white horse in a sunlit meadow.")
        XCTAssertEqual(caption.objects.first?.shapeAndColor, "armored human figure in silver and blue")
        XCTAssertEqual(caption.textRender, [])
    }

    func testStructuredPromptAdapterCleansJSONCodeFence() throws {
        let wrapped = """
        <think>draft</think>
        ```json
        \(Self.validStructuredCaptionJSON)
        ```
        """

        let cleaned = StructuredImagePromptAdapter.cleanedJSONCandidate(from: wrapped)
        XCTAssertNoThrow(try StructuredImagePromptAdapter.validateCaptionJSON(cleaned))
    }

    func testStructuredPromptAdapterNormalizesQ36NearSchema() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.q36NearStructuredCaptionJSON,
            fallbackPrompt: "a small matte red cube on a white table"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.shortDescription, "A clean product photograph featuring a small matte red cube.")
        XCTAssertEqual(caption.objects.first?.description, "matte red cube, small, matte finish, red color")
        XCTAssertEqual(caption.objects.first?.texture, "matte surface")
        XCTAssertEqual(caption.textRender, [])
    }

    func testStructuredPromptAdapterNormalizesCompactGemmaSchema() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.gemmaCompactStructuredCaptionJSON,
            fallbackPrompt: "a small matte red cube on a white table"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.objects.first?.description, "small matte red cube")
        XCTAssertEqual(caption.lighting.conditions, "Soft window light, natural diffusion")
        XCTAssertEqual(caption.photographicCharacteristics.depthOfField, "Product photography, sharp focus")
        XCTAssertEqual(caption.textRender, [])
    }

    func testStructuredPromptAdapterNormalizesGemmaSnakeCaseSchema() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.gemmaSnakeCaseStructuredCaptionJSON,
            fallbackPrompt: "a cafe sign reads 'TOKENS: FREE TODAY'"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.backgroundSetting, "quiet sidewalk cafe")
        XCTAssertEqual(caption.aesthetics.colorScheme, "warm earth tones")
        XCTAssertEqual(caption.photographicCharacteristics.cameraAngle, "eye-level")
        XCTAssertEqual(caption.styleMedium, "Photography")
        XCTAssertEqual(caption.artisticStyle, "Candid street photography")
        XCTAssertEqual(caption.textRender.map(\.text), ["TOKENS: FREE TODAY"])
    }

    func testStructuredPromptAdapterRepairsGemmaJSONPrefix() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.gemmaTruncatedStructuredCaptionJSON,
            fallbackPrompt: "a cafe sign reads 'TOKENS: FREE TODAY' and 'ask for Dewane'"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.shortDescription, "A hand-painted cafe sign.")
        XCTAssertEqual(caption.backgroundSetting, "quiet sidewalk cafe")
        XCTAssertEqual(caption.textRender.map(\.text), ["TOKENS: FREE TODAY", "ask for Dewane"])
        XCTAssertEqual(caption.textRender.last?.font, "unspecified")
    }

    func testStructuredPromptAdapterRejectsGemmaSpecialTokenSpill() {
        XCTAssertThrowsError(try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.gemmaNonsenseTailStructuredCaptionJSON,
            fallbackPrompt: "a cafe sign reads 'TOKENS: FREE TODAY' and 'ask for Dewane'"
        )) { error in
            guard case StructuredImagePromptAdapterError.invalidCaptionJSON(let detail) = error else {
                return XCTFail("Expected invalidCaptionJSON, got \(error)")
            }
            XCTAssertTrue(detail.contains("multimodal special tokens"))
        }
    }

    func testStructuredPromptAdapterBuildsDeterministicFallbackJSON() throws {
        let json = try StructuredImagePromptAdapter.deterministicCaptionJSON(
            for: "a cafe sign reads 'TOKENS: FREE TODAY'"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(json)
        XCTAssertEqual(caption.shortDescription, "a cafe sign reads 'TOKENS: FREE TODAY'")
        XCTAssertEqual(caption.objects.first?.description, "a cafe sign reads 'TOKENS: FREE TODAY'")
        XCTAssertEqual(caption.textRender.map(\.text), ["TOKENS: FREE TODAY"])
        XCTAssertFalse(StructuredImagePromptAdapter.containsGeneratedSpecialTokenSpill(json))
    }

    func testQuotedStringsExtraction() {
        let prompt = """
        a trailhead sign reads 'THE LOCAL WILD' with a smaller line below reading \
        'do not feed the models', the sign's letters carved into wood, titled "Field Notes"
        """
        let quoted = StructuredImagePromptAdapter.quotedStrings(in: prompt)
        XCTAssertEqual(quoted, ["Field Notes", "THE LOCAL WILD", "do not feed the models"])
    }

    func testQuotedStringsIgnoresApostrophes() {
        let quoted = StructuredImagePromptAdapter.quotedStrings(
            in: "the sign's weathered face catches the morning's first light"
        )
        XCTAssertEqual(quoted, [])
    }

    func testEnsuringTextRenderInjectsQuotedPromptText() throws {
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(Self.validStructuredCaptionJSON)
        XCTAssertEqual(caption.textRender, [])

        let ensured = StructuredImagePromptAdapter.ensuringTextRender(
            caption,
            prompt: "a banner above the knight reads 'ONWARD'"
        )
        XCTAssertEqual(ensured.textRender.count, 1)
        XCTAssertEqual(ensured.textRender.first?.text, "ONWARD")
    }

    func testEnsuringTextRenderKeepsModelProvidedEntries() throws {
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(Self.validStructuredCaptionJSON)
        let withEntry = StructuredImagePromptAdapter.ensuringTextRender(
            caption,
            prompt: "a banner reads 'ONWARD'"
        )
        // A second pass must not duplicate or overwrite the existing entry.
        let unchanged = StructuredImagePromptAdapter.ensuringTextRender(
            withEntry,
            prompt: "a banner reads 'SOMETHING ELSE'"
        )
        XCTAssertEqual(unchanged.textRender, withEntry.textRender)
    }

    func testNormalizedCaptionJSONInjectsTextRenderForQuotedPrompt() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.validStructuredCaptionJSON,
            fallbackPrompt: "a knight under a banner that reads 'ONWARD', sunny meadow"
        )
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.textRender.map(\.text), ["ONWARD"])
    }

    func testIsParseableJSONDistinguishesNearMissFromTokenSalad() {
        XCTAssertTrue(StructuredImagePromptAdapter.isParseableJSON(#"{"objects": "wrong shape"}"#))
        XCTAssertFalse(StructuredImagePromptAdapter.isParseableJSON(
            "{ed feetization mas Dod asked Lolamente lesser0 or<audio|>"
        ))
        XCTAssertFalse(StructuredImagePromptAdapter.isParseableJSON(""))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-image-preflight-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        return temp
    }

    private func writeManifest(
        id: String,
        family: MereRunModelManifest.Family,
        engine: MereRunModelManifest.Engine? = nil,
        defaults: MereRunModelManifest.Defaults? = nil,
        to directory: URL
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifest = MereRunModelManifest(
            id: id,
            engine: engine,
            family: family,
            variant: .distilled,
            precision: .bf16,
            defaults: defaults,
            supports: []
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest)
            .write(to: directory.appendingPathComponent(MereRunModelManifest.filename))
    }

    private static let validStructuredCaptionJSON = """
    {
      "short description": "A knight riding a white horse in a sunlit meadow.",
      "objects": [
        {
          "description": "A calm medieval knight seated on a horse, wearing polished armor and a blue cloak.",
          "location": "center foreground",
          "relationship": "The knight is riding the horse and looking toward the horizon.",
          "relative size": "large within frame",
          "shape and color": "armored human figure in silver and blue",
          "texture": "metallic armor and woven fabric",
          "appearance details": "helmet visor raised, cloak moving lightly",
          "number of objects": null,
          "pose": "upright seated riding pose",
          "expression": "calm and focused",
          "clothing": "plate armor and blue cloak",
          "action": "riding a horse",
          "gender": "unspecified",
          "skin tone and texture": null,
          "orientation": "facing right"
        }
      ],
      "background setting": "Open meadow with distant trees and low hills under a clear sky.",
      "lighting": {
        "conditions": "bright daylight",
        "direction": "side-lit from left",
        "shadows": "soft shadows falling to the right"
      },
      "aesthetics": {
        "composition": "centered heroic composition",
        "color scheme": "natural greens with silver and blue accents",
        "mood atmosphere": "noble and serene"
      },
      "photographic characteristics": {
        "depth of field": "moderate",
        "focus": "sharp focus on knight and horse",
        "camera angle": "eye-level",
        "lens focal length": "normal lens"
      },
      "style medium": "digital illustration",
      "text render": [],
      "context": "Fantasy character illustration suitable for concept art.",
      "artistic style": "cinematic realism"
    }
    """

    private static let q36NearStructuredCaptionJSON = """
    {
      "short description": "A clean product photograph featuring a small matte red cube.",
      "objects": [
        {
          "name": "matte red cube",
          "attributes": [
            "small",
            "matte finish",
            "red color"
          ]
        }
      ],
      "background setting": "Minimalist white tabletop and clean neutral background.",
      "lighting": {
        "conditions": "Soft, diffused natural light",
        "direction": "From a side window",
        "shadows": "Soft, subtle shadows"
      },
      "aesthetics": {
        "composition": "Centered product composition",
        "color scheme": "Red and white with neutral tones",
        "mood atmosphere": "Clean and professional"
      },
      "photographic characteristics": {
        "depth of field": "Shallow to medium",
        "focus": "Sharp focus on the cube",
        "camera angle": "Slightly elevated",
        "lens focal length": "50mm"
      },
      "style medium": "photograph",
      "artistic style": "minimal product",
      "context": "Product showcase",
      "text render": null
    }
    """

    private static let gemmaCompactStructuredCaptionJSON = """
    ```json
    {
      "short description": "A small matte red cube on a white table.",
      "objects": [
        "small matte red cube",
        "white table"
      ],
      "background setting": "Minimalist interior, clean studio environment",
      "lighting": "Soft window light, natural diffusion",
      "aesthetics": "Minimalist, clean, modern",
      "photographic characteristics": "Product photography, sharp focus",
      "style medium": "Photography",
      "artistic style": "Commercial product photography",
      "context": "Product showcase",
      "text render": "none"
    }
    ```
    """

    private static let gemmaSnakeCaseStructuredCaptionJSON = """
    {
      "short description": "A hand-painted cafe sign.",
      "objects": [
        {
          "description": "wooden sandwich board",
          "location": "sidewalk",
          "relative size": "medium",
          "shape and color": "brown A-frame board",
          "texture": "painted wood",
          "appearance details": "chalk lettering",
          "relationship": "foreground",
          "number of objects": 1
        }
      ],
      "background_setting": "quiet sidewalk cafe",
      "lighting": "soft morning light",
      "aesthetics": {
        "composition": "off-center street photo",
        "color_scheme": "warm earth tones",
        "mood_atmosphere": "inviting"
      },
      "photographic_characteristics": {
        "depth_of_field": "shallow",
        "focus": "sharp sign text",
        "camera_angle": "eye-level",
        "lens_focal_length": "35mm"
      },
      "style_medium": "Photography",
      "artistic_style": "Candid street photography",
      "context": "morning cafe promotion",
      "text_render": [
        {
          "text": "TOKENS: FREE TODAY",
          "location": "main board",
          "size": "large",
          "color": "white chalk",
          "font": "handwritten",
          "appearance details": "chalk texture"
        }
      ]
    }
    """

    private static let gemmaTruncatedStructuredCaptionJSON = """
    {
      "short_description": "A hand-painted cafe sign.",
      "objects": ["wooden sandwich board"],
      "background_setting": "quiet sidewalk cafe",
      "lighting": "soft morning light",
      "aesthetics": "warm street photo",
      "photographic_characteristics": "35mm lens, eye-level",
      "style_medium": "Photography",
      "artistic_style": "Candid street photography",
      "context": "morning cafe promotion",
      "text_render": [
        {
          "text": "TOKENS: FREE TODAY",
          "location": "main board",
          "size": "large",
          "color": "white chalk",
          "font": "handwritten"
        },
        {
          "text": "ask for Dewane",
          "location": "bottom of board",
          "size": "small",
          "color": "White chalk",
    """

    private static let gemmaNonsenseTailStructuredCaptionJSON = """
    {
      "short_description": "A hand-painted cafe sign.",
      "objects": ["wooden sandwich board"],
      "background_setting": "quiet sidewalk cafe",
      "lighting": {
        "conditions": "soft morning light",
        "direction": "front-left"
      },
      "aesthetics": {
        "composition": "off-center street photo",
        "color_scheme": "warm earth tones",
        "mood_atmosphere": "inviting"
      },
      "photographic_characteristics": {
        "depth_of_field": "shallow",
        "focus": "sharp sign text",
        "camera_angle": "eye-level",
        "lens_focal_length": "35mm"
      },
      "style_medium": "Photography",
      "artistic_style": "Candid street photography",
      "context": "morning cafe promotion",
      "text_render": [
        {
          "text": "TOKENS: FREE TODAY",
          "location": "main board",
          "size": "large",
          "color": "white chalk",
          "font": "handwritten",
          "appearance details": "chalk texture"
        },
        {
          "text": "ask for Dewane",
          "location": "bottom of board",
          "size": "small",
          "color": "White chalk",
          "font": "Handwritten, rustic $" loose token salad <audio|> still spilling,
    """
}
