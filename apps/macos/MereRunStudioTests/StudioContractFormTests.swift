@testable import MereRunApp
import MereRunContract
import XCTest

/// The inspector and the Command view are rendered from `MereRunCapabilityCatalog`: one control
/// per option `kind`, sections in the contract's `group` order, rows gated by `depends_on`, values
/// clamped to `range`, and a control at its `default_value` emitting nothing. The argv must not
/// move: a draft round-tripped through every binding produces exactly the command the app already
/// produced.
final class StudioContractFormTests: XCTestCase {
    // MARK: Control per kind

    func testEachOptionKindPicksItsControl() throws {
        let image = try XCTUnwrap(StudioContractSchema.capability(for: .createImage))
        func control(_ flag: String, in capability: MereRunCommandCapability) throws -> StudioContractControl {
            let option = try XCTUnwrap(capability.options.first { $0.flag == flag })
            return StudioContractField(option: option, bindings: [.text("x", \.model)]).control
        }

        XCTAssertEqual(try control("--negative-prompt", in: image), .field, "string")
        XCTAssertEqual(try control("--structured-prompt", in: image), .toggle, "boolean")
        XCTAssertEqual(try control("--krea-base-quantization-bits", in: image), .segmented, "choice of two")
        XCTAssertEqual(try control("--input", in: image), .path, "file")
        XCTAssertEqual(try control("--structured-prompt-model-root", in: image), .path, "directory")
        XCTAssertEqual(try control("--strength", in: image), .slider, "number with both ends")
        XCTAssertEqual(try control("--krea-conditioning-multiplier", in: image), .stepper, "number with no range")

        let video = try XCTUnwrap(StudioContractSchema.capability(for: .video))
        XCTAssertEqual(try control("--h3-acceleration", in: video), .picker, "choice of seven")
        let music = try XCTUnwrap(StudioContractSchema.capability(for: .music))
        XCTAssertEqual(try control("--retake-seed", in: music), .stepper, "an integer with only a lower bound")
    }

    func testAnOverriddenFlagRendersItsOwnEditor() {
        let fields = StudioContractSchema.fields(for: .createImage)
        let width = fields.first { $0.flag == "--width" }
        XCTAssertEqual(width?.control, .override)
        XCTAssertEqual(width?.overrideID, .dimensions)
        XCTAssertEqual(width?.draftFieldIDs, ["width", "height"], "one row writes both")
        XCTAssertNil(fields.first { $0.flag == "--height" }, "the pair renders once, at the first flag")
    }

    // MARK: The override registry

    func testTheOverrideRegistryOwnsTheCompositeEditorsAndNothingElse() {
        let image = StudioContractSchema.fields(for: .createImage)
        XCTAssertEqual(image.first { $0.flag == "--mask" }?.overrideID, .imageCanvas)
        XCTAssertEqual(
            image.first { $0.flag == "--mask" }?.draftFieldIDs,
            ["imageMaskPath", "imageMaskFeather", "imageOutpaintTop", "imageOutpaintRight",
             "imageOutpaintBottom", "imageOutpaintLeft"],
            "the canvas owns the mask, the feather, and all four outpaint edges"
        )
        XCTAssertEqual(image.first { $0.flag == "--lora" }?.overrideID, .lora)
        XCTAssertNil(image.first { $0.flag == "--sigma-shift" }?.overrideID, "plain options render from the contract")

        XCTAssertEqual(StudioContractSchema.fields(for: .video).first { $0.flag == "--reference" }?.overrideID,
                       .orderedReferences)
        XCTAssertEqual(StudioContractSchema.fields(for: .music).first { $0.flag == "--adapter" }?.overrideID,
                       .musicAdapters)

        // An attachment the composer's well owns is still a field — the Command view lists it —
        // but the inspector never repeats it.
        let draft = StudioDraft()
        XCTAssertNotNil(StudioContractSchema.fields(for: .createImage).first { $0.flag == "--input" })
        XCTAssertNil(
            StudioContractSchema.inspectorFields(for: .createImage, draft: draft).first { $0.flag == "--input" },
            "the attachment well owns --input"
        )
    }

    // MARK: Sections

    func testSectionsFollowTheContractsGroupOrder() {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        XCTAssertEqual(
            StudioContractSchema.sections(for: .createImage, draft: draft).map(\.title),
            ["Prompt", "Inputs", "Output", "Model & adapters", "Sampling"]
        )
        XCTAssertEqual(
            StudioContractGroup.allCases.map(\.title),
            ["Prompt", "Inputs", "Output", "Model & adapters", "Sampling", "Run", "Options"]
        )

        // Every group's fields keep the contract's own declaration order.
        let sampling = StudioContractSchema.sections(for: .createImage, draft: draft)
            .first { $0.group == .sampling }
        XCTAssertEqual(sampling?.fields.map(\.flag), ["--cfg", "--steps", "--seed"])
    }

    func testTiersSplitTheInspectorBetweenItsSectionsAndAdvanced() {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        let sections = StudioContractSchema.sections(for: .createImage, draft: draft)
        XCTAssertTrue(sections.flatMap(\.fields).allSatisfy { $0.tier != .expert })
        let advanced = StudioContractSchema.expertFields(for: .createImage, draft: draft)
        XCTAssertTrue(advanced.allSatisfy { $0.tier == .expert })
        XCTAssertTrue(advanced.contains { $0.flag == "--sigma-shift" })
        XCTAssertFalse(advanced.isEmpty)
    }

    // MARK: depends_on

    func testDependentRowsStayHiddenUntilTheirOptionCarriesAValue() {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        let fields = StudioContractSchema.boundFields(for: .createImage, draft: draft)
        let promptModel = fields.first { $0.flag == "--structured-prompt-model" }!

        XCTAssertFalse(
            StudioContractSchema.isVisible(promptModel, for: .createImage, in: draft),
            "the prompt model depends on --structured-prompt"
        )
        draft.structuredPrompt = true
        XCTAssertTrue(StudioContractSchema.isVisible(promptModel, for: .createImage, in: draft))

        // A chained dependency: --json needs --preflight, which nothing else gates.
        let json = fields.first { $0.flag == "--json" }!
        XCTAssertFalse(StudioContractSchema.isVisible(json, for: .createImage, in: draft))
        draft.preflight = true
        XCTAssertTrue(StudioContractSchema.isVisible(json, for: .createImage, in: draft))

        // A dependency the composer's well owns is read from the draft all the same.
        let feather = fields.first { $0.flag == "--mask" }!
        XCTAssertFalse(StudioContractSchema.isVisible(feather, for: .createImage, in: draft))
        draft.inputPath = "/tmp/mug.png"
        XCTAssertTrue(StudioContractSchema.isVisible(feather, for: .createImage, in: draft))
    }

    // MARK: default_value

    func testAControlAtItsDefaultEmitsNothing() throws {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        // Per flag, before the composite editors fold their flags into one row.
        let fields = StudioContractSchema.boundFields(for: .createImage, draft: draft)
        let feather = try XCTUnwrap(fields.first { $0.flag == "--mask-feather" })
        XCTAssertEqual(feather.option.defaultValue, "8")
        XCTAssertTrue(feather.isAtDefault(in: draft))
        XCTAssertFalse(feather.emits(in: draft))
        draft.imageMaskFeather = 16
        XCTAssertTrue(feather.emits(in: draft))

        // "1.0" and 1 are the same default.
        let scale = try XCTUnwrap(fields.first { $0.flag == "--lora-scale" })
        XCTAssertEqual(scale.option.defaultValue, "1.0")
        XCTAssertTrue(scale.isAtDefault(in: draft))

        // Blank text and an unset optional are both "no flag".
        let negative = try XCTUnwrap(fields.first { $0.flag == "--negative-prompt" })
        XCTAssertFalse(negative.emits(in: draft))
        draft.secondaryText = "blurry"
        XCTAssertTrue(negative.emits(in: draft))

        // A number the contract declares no default for: zero, and anything below the declared
        // minimum, is the app saying it has no opinion, and the argv leaves the flag out.
        var chat = StudioDraft()
        chat.reset(for: .chat)
        let chatFields = StudioContractSchema.boundFields(for: .chat, draft: chat)
        let kvBits = try XCTUnwrap(chatFields.first { $0.flag == "--kv-bits" })
        XCTAssertEqual(chat.kvBits, 0)
        XCTAssertFalse(kvBits.emits(in: chat))
        let scheme = try XCTUnwrap(chatFields.first { $0.flag == "--kv-quant-scheme" })
        XCTAssertFalse(
            StudioContractSchema.isVisible(scheme, for: .chat, in: chat),
            "the KV scheme depends on --kv-bits"
        )
        chat.kvBits = 4
        XCTAssertTrue(kvBits.emits(in: chat))
        XCTAssertTrue(StudioContractSchema.isVisible(scheme, for: .chat, in: chat))

        var video = StudioDraft()
        video.reset(for: .video)
        let weights = try XCTUnwrap(
            StudioContractSchema.boundFields(for: .video, draft: video).first { $0.flag == "--h3-weight-mode" }
        )
        XCTAssertFalse(weights.emits(in: video), "an optional the draft leaves nil")
        video.h3WeightMode = "quantized"
        XCTAssertTrue(weights.emits(in: video))
    }

    // MARK: range

    func testValuesClampAndSnapToTheContractsRange() throws {
        let fields = StudioContractSchema.boundFields(for: .createImage)
        let steps = try XCTUnwrap(fields.first { $0.flag == "--steps" })
        XCTAssertEqual(steps.clamped(.integer(500)), .integer(100), "the contract caps steps at 100")
        XCTAssertEqual(steps.clamped(.integer(0)), .integer(1))

        let strength = try XCTUnwrap(fields.first { $0.flag == "--strength" })
        XCTAssertEqual(strength.clamped(.number(2.0)), .number(1.0))
        if case .number(let snapped) = strength.clamped(.number(0.42)) {
            XCTAssertEqual(snapped, 0.4, accuracy: 0.0001, "0.05 steps")
        } else {
            XCTFail("expected a number")
        }

        // An option with no range passes through untouched.
        let krea = try XCTUnwrap(fields.first { $0.flag == "--krea-conditioning-multiplier" })
        XCTAssertEqual(krea.clamped(.number(37.5)), .number(37.5))

        // Writing through the field clamps as well, so a control can never store an invalid value.
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        steps.write(.integer(9_000), to: &draft)
        XCTAssertEqual(draft.steps, 100)
    }

    // MARK: The argv golden test

    func testEveryBoundControlRoundTripsWithoutMovingTheArgv() throws {
        for mode in StudioMode.allCases {
            var draft = StudioDraft()
            draft.reset(for: mode)
            // Move every control the mode's form owns off its default, the way editing would.
            for field in StudioContractSchema.fields(for: mode, draft: draft) {
                for binding in field.bindings {
                    binding.write(&draft, Self.perturbed(binding.read(draft), field: field))
                }
            }

            let edited = draft
            let before = try StudioCommandAdapter.makeRequest(mode: mode, draft: edited, validating: false)
            StudioContractSchema.roundTrip(&draft, for: mode)
            let after = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft, validating: false)

            XCTAssertEqual(draft, edited, "\(mode): a control that reads and writes itself must not move the draft")
            XCTAssertEqual(
                after.template.arguments(from: after.draft),
                before.template.arguments(from: before.draft),
                "\(mode): the contract-driven form changed the argv"
            )
        }
    }

    func testEveryModeBindsOnlyFlagsItsCapabilityDeclares() {
        for mode in StudioMode.allCases {
            guard let capability = StudioContractSchema.capability(for: mode) else { continue }
            let declared = Set(capability.options.map(\.flag))
            for field in StudioContractSchema.fields(for: mode) where field.flag != "read-image-action" {
                XCTAssertTrue(declared.contains(field.flag), "\(mode) binds \(field.flag), which \(capability.id) does not declare")
            }
        }
    }

    func testEveryDraftFieldIsBoundByAtMostOneControlPerMode() {
        for mode in StudioMode.allCases {
            let ids = StudioContractSchema.fields(for: mode).flatMap(\.draftFieldIDs)
            XCTAssertEqual(ids.count, Set(ids).count, "\(mode) binds a draft field twice: \(ids)")
        }
    }

    /// A value one edit away from the one it is given, inside whatever the contract allows.
    private static func perturbed(_ value: StudioContractValue, field: StudioContractField) -> StudioContractValue {
        switch value {
        case .flag(let on):
            return .flag(!on)
        case .integer(let integer):
            return field.clamped(.integer(integer + Int(field.option.range?.step ?? 1)))
        case .number(let number):
            return field.clamped(.number(number + (field.option.range?.step ?? 0.5)))
        case .text(let text):
            return .text(replacement(for: field, avoiding: text))
        case .unset:
            switch field.kind {
            case .integer: return field.clamped(.integer(Int(field.option.range?.min ?? 0) + 1))
            case .number: return field.clamped(.number((field.option.range?.min ?? 0) + 0.5))
            case .boolean: return .flag(true)
            default: return .text(replacement(for: field, avoiding: ""))
            }
        }
    }

    private static func replacement(for field: StudioContractField, avoiding current: String) -> String {
        if !field.option.choices.isEmpty {
            return field.option.choices.first { $0 != current } ?? current
        }
        switch field.kind {
        case .file: return "/tmp/mere-run-contract-form.bin"
        case .directory: return "/tmp/mere-run-contract-form"
        default: return current + "-edited"
        }
    }
}
