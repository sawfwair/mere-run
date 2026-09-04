@testable import MereRunApp
import MereRunContract
import XCTest

/// The Command view's rows come from the template's own argv for the draft, grouped by flag,
/// with the contract's labels and boolean kinds; the preview wraps the display command.
final class StudioCommandRowsTests: XCTestCase {
    func testFlagsFileUnderTheContractsGroups() {
        let image = MereRunCapabilityCatalog.command(id: "image.generate")
        func group(_ flag: String) -> StudioCommandRowGroup {
            StudioCommandRowGroup.group(forFlag: flag, in: image)
        }
        XCTAssertEqual(group("--prompt"), .contract(.prompt))
        XCTAssertEqual(group("--negative-prompt"), .contract(.prompt))
        XCTAssertEqual(group("--input"), .contract(.inputs))
        XCTAssertEqual(group("--width"), .contract(.output))
        XCTAssertEqual(group("--output"), .contract(.output))
        XCTAssertEqual(group("--model"), .contract(.model))
        XCTAssertEqual(group("--lora-scale"), .contract(.model))
        XCTAssertEqual(group("--steps"), .contract(.sampling))
        XCTAssertEqual(group("--sigma-shift"), .contract(.sampling))
        XCTAssertEqual(group("--progress-json"), .contract(.run))
        XCTAssertEqual(group("--krea-conditioning-multiplier"), .contract(.sampling))
        XCTAssertEqual(group("--not-a-contract-flag"), .contract(.options), "a flag the contract has yet to describe")
    }

    func testParseSplitsPositionalsAndFlagValues() {
        let parsed = StudioCommandRows.parse(
            arguments: ["text", "embed", "one", "two", "--json", "--model", "m", "--quiet"],
            commandPathCount: 2
        )
        XCTAssertEqual(parsed.positional, ["one", "two"])
        XCTAssertEqual(parsed.flags.map(\.0), ["--json", "--model", "--quiet"])
        XCTAssertEqual(parsed.flags.map(\.1), [nil, "m", nil])
    }

    func testImageGenerateRowsCarryTheDraftValues() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "a ceramic coffee mug in soft morning light"
        draft.outputPath = "~/Pictures/mere.run/Image/mug.png"
        draft.width = 1024
        draft.height = 1024
        draft.steps = 4
        draft.cfgScale = 3.5
        draft.sigmaShift = 3.0
        draft.progressJSON = true

        let groups = StudioCommandRows.groups(template: template, draft: draft)
        XCTAssertEqual(
            groups.map(\.group.title),
            ["Prompt", "Inputs", "Output", "Model & adapters", "Sampling", "Run"]
        )

        func row(_ flag: String) -> StudioCommandRow? {
            groups.flatMap(\.rows).first { $0.flag == flag }
        }
        XCTAssertEqual(row("--prompt")?.value, .text("a ceramic coffee mug in soft morning light"))
        XCTAssertEqual(row("--prompt")?.label, "Prompt")
        XCTAssertEqual(row("--negative-prompt")?.value, .text(""), "declared but unset options still show")
        XCTAssertEqual(row("--width")?.value, .text("1024"))
        XCTAssertEqual(row("--cfg")?.value, .text("3.5"))
        XCTAssertEqual(row("--sigma-shift")?.value, .text("3"))
        XCTAssertEqual(row("--seed")?.value, .text(""))
        XCTAssertEqual(row("--progress-json")?.value, .toggle(true))
        XCTAssertEqual(row("--preflight")?.value, .toggle(false))
        XCTAssertEqual(row("--model")?.value, .text(template.defaultModel))

        let sampling = try XCTUnwrap(groups.first { $0.group == .contract(.sampling) })
        XCTAssertEqual(sampling.rows.prefix(3).map(\.flag), ["--cfg", "--sigma-shift", "--steps"], "set rows first, in declaration order")
        XCTAssertTrue(sampling.rows.last?.isSet == false)
    }

    func testRepeatedFlagsJoinAndPositionalsGetTheirOwnGroup() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .textEmbed))
        var draft = template.defaultDraft()
        draft.prompt = "query one\nquery two"
        let groups = StudioCommandRows.groups(template: template, draft: draft)
        let arguments = try XCTUnwrap(groups.first { $0.group == .arguments })
        XCTAssertEqual(arguments.rows.map(\.value), [.positional("query one"), .positional("query two")])
    }

    func testPreviewWrapsFlagPairsWithContinuations() {
        let command = "mere.run image generate --prompt 'a ceramic coffee mug in soft morning light' "
            + "--output '~/Pictures/mere.run/Image/mug.png' --width 1024 --height 1024 --steps 4 "
            + "--model image-zimage-nano --cfg 3.5 --sigma-shift 3.0 --progress-json"
        let wrapped = StudioCommandPreviewFormatter.wrapped(command, width: 60)
        let lines = wrapped.components(separatedBy: "\n")
        XCTAssertEqual(lines.first, "mere.run image generate \\")
        XCTAssertTrue(lines.dropFirst().allSatisfy { $0.hasPrefix("  ") })
        XCTAssertTrue(lines.dropLast().allSatisfy { $0.hasSuffix(" \\") })
        XCTAssertFalse(lines.last?.hasSuffix("\\") ?? true)
        XCTAssertTrue(lines.contains { $0.contains("--prompt 'a ceramic coffee mug in soft morning light'") })
        XCTAssertTrue(lines.allSatisfy { $0.count <= 60 + 4 || $0.contains("--prompt") })
        XCTAssertEqual(StudioCommandPreviewFormatter.wrapped("mere.run status"), "mere.run status")
    }
}
