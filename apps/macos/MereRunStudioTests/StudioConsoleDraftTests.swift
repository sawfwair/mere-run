@testable import MereRunApp
import MereRunContract
import XCTest

/// The Command Console renders and runs from `MereRunCapabilityCatalog` rather than from a
/// hand-written view per template. These tests hold the two properties that makes safe: opening
/// a template starts on exactly the command the app would otherwise have run, and every option
/// the contract declares has a row.
final class StudioConsoleDraftTests: XCTestCase {
    // MARK: Seeding

    /// The console reads a template's own argv into contract values, and builds the same argv
    /// back out of them. If this fails for a template, the console would silently launch a
    /// different command than the catalog does.
    func testSeedingEveryTemplateRebuildsItsOwnCommand() throws {
        for template in CommandCatalog.templates {
            guard let capability = template.id.capability else { continue }
            let expected = template.arguments(from: template.defaultDraft())
            let draft = StudioConsoleCommand.seed(template: template, draft: template.defaultDraft())
            let rebuilt = StudioConsoleCommand.arguments(for: capability, draft: draft)

            XCTAssertEqual(
                Array(rebuilt.prefix(capability.command.count)),
                capability.command,
                "\(template.id) command path"
            )
            XCTAssertEqual(
                Self.pairs(of: rebuilt, capability: capability),
                Self.pairs(of: expected, capability: capability),
                "\(template.id) rebuilt a different command"
            )
        }
    }

    /// A Library row records the argv its run launched; "Edit command" reopens on that, so an
    /// option the console edited but no `CommandDraft` field carries is not lost.
    func testALibraryRowsArgumentsReopenTheSameCommand() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "image.generate"))
        let argv = [
            "image", "generate", "--prompt", "a kite", "--output", "/tmp/kite.png",
            "--width", "768", "--height", "768", "--steps", "6", "--sigma-shift", "3.0",
        ]
        let draft = StudioConsoleCommand.seed(capability: capability, arguments: argv)
        XCTAssertEqual(draft.text("--prompt"), "a kite")
        XCTAssertEqual(draft.text("--sigma-shift"), "3.0")
        XCTAssertEqual(
            Self.pairs(of: StudioConsoleCommand.arguments(for: capability, draft: draft), capability: capability),
            Self.pairs(of: argv, capability: capability)
        )
    }

    /// Arguments the contract does not declare survive the round trip in Extra arguments rather
    /// than being dropped.
    func testUndeclaredPositionalsLandInExtraArguments() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "model.info"))
        let draft = StudioConsoleCommand.seed(
            capability: capability,
            arguments: ["model", "info", "image-zimage-nano", "extra-token"]
        )
        XCTAssertEqual(draft.arguments, ["image-zimage-nano"])
        XCTAssertEqual(draft.extraArguments, "extra-token")
        XCTAssertEqual(
            StudioConsoleCommand.arguments(for: capability, draft: draft),
            ["model", "info", "image-zimage-nano", "extra-token"]
        )
    }

    // MARK: Coverage

    /// Every option of every capability gets a row, in one group, exactly once. The console is
    /// the guaranteed home for a capability nobody has designed a surface for, so a flag with no
    /// row would be unreachable in the app.
    func testEveryCapabilityOptionHasExactlyOneRow() {
        for capability in MereRunCapabilityCatalog.document.commands {
            let rows = StudioConsoleCommand.groups(for: capability).flatMap(\.fields)
            let flags = rows.map(\.flag)
            XCTAssertEqual(
                Set(flags).count, flags.count,
                "\(capability.id) draws a flag twice: \(flags)"
            )
            for option in capability.options {
                XCTAssertTrue(flags.contains(option.flag), "\(capability.id) has no row for \(option.flag)")
            }
            for argument in capability.arguments {
                XCTAssertTrue(flags.contains(argument.name), "\(capability.id) has no row for \(argument.name)")
            }
        }
    }

    /// Positionals come first, then the contract's own groups in their declared order.
    func testGroupsFollowTheContractsOrder() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "image.generate"))
        let titles = StudioConsoleCommand.groups(for: capability).map(\.title)
        XCTAssertEqual(titles.first, "Prompt", "image generate takes no positional")
        XCTAssertEqual(
            titles,
            StudioContractGroup.allCases.map(\.title).filter { titles.contains($0) }
        )

        let info = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "model.info"))
        XCTAssertEqual(StudioConsoleCommand.groups(for: info).first?.title, "Arguments")
    }

    // MARK: Emission

    func testOnlyValuesTheDraftCarriesReachTheCommandLine() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "image.generate"))
        var draft = StudioConsoleDraft()
        draft["--prompt"] = .text("a kite")
        XCTAssertEqual(
            StudioConsoleCommand.arguments(for: capability, draft: draft),
            ["image", "generate", "--prompt", "a kite"]
        )

        // A blank value is "no flag", even where the contract declares a default: the console
        // shows exactly what it runs.
        draft["--negative-prompt"] = .text("")
        draft["--steps"] = .unset
        XCTAssertEqual(
            StudioConsoleCommand.arguments(for: capability, draft: draft),
            ["image", "generate", "--prompt", "a kite"]
        )

        // A boolean emits its flag and nothing else; off emits nothing.
        draft["--preflight"] = .flag(true)
        XCTAssertTrue(StudioConsoleCommand.arguments(for: capability, draft: draft).contains("--preflight"))
        draft["--preflight"] = .flag(false)
        XCTAssertFalse(StudioConsoleCommand.arguments(for: capability, draft: draft).contains("--preflight"))
    }

    func testARepeatableOptionEmitsOncePerLine() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "image.generate"))
        let reference = try XCTUnwrap(capability.options.first { $0.flag == "--ref-image" })
        XCTAssertTrue(reference.repeatable)

        var draft = StudioConsoleDraft()
        draft["--prompt"] = .text("a kite")
        draft["--ref-image"] = .text("/tmp/one.png\n/tmp/two.png\n")
        XCTAssertEqual(
            StudioConsoleCommand.arguments(for: capability, draft: draft),
            ["image", "generate", "--prompt", "a kite", "--ref-image", "/tmp/one.png", "--ref-image", "/tmp/two.png"]
        )
    }

    func testExtraArgumentsAreAppendedAsTyped() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "model.list"))
        var draft = StudioConsoleDraft()
        draft.extraArguments = "--verbose \"two words\""
        XCTAssertEqual(
            StudioConsoleCommand.arguments(for: capability, draft: draft),
            ["model", "list", "--verbose", "two words"]
        )
    }

    // MARK: Validation and output

    func testARequiredArgumentBlocksTheRun() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "model.info"))
        var draft = StudioConsoleDraft()
        XCTAssertNotNil(StudioConsoleCommand.validationMessage(for: capability, draft: draft))
        draft.arguments = ["image-zimage-nano"]
        XCTAssertNil(StudioConsoleCommand.validationMessage(for: capability, draft: draft))
    }

    /// The job lifecycle prepares the output folder and adopts the written file from
    /// `CommandDraft.outputPath`; the console fills it from the option the contract names.
    func testTheOutputPathComesFromTheContractsDeclaredFlag() throws {
        let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: "image.generate"))
        XCTAssertEqual(capability.output.flag, "--output")
        var draft = StudioConsoleDraft()
        draft["--output"] = .text("/tmp/kite.png")
        XCTAssertEqual(StudioConsoleCommand.outputPath(for: capability, draft: draft), "/tmp/kite.png")

        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let command = StudioConsoleCommand.commandDraft(
            seed: CommandDraft(),
            template: template,
            capability: capability,
            draft: draft
        )
        XCTAssertEqual(command.outputPath, "/tmp/kite.png")
    }

    /// A key typed into the console travels in the environment, as it does everywhere else in
    /// the app: the flag is left out of the command and the draft field the launcher reads is
    /// filled instead.
    func testASecretNeverReachesTheCommandLine() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        var draft = StudioConsoleCommand.seed(template: template, draft: template.defaultDraft())
        draft["--api-key"] = .text("sk-not-in-argv")

        let launch = try XCTUnwrap(StudioConsoleRun(template: template, draft: draft, seed: template.defaultDraft()))
        XCTAssertFalse(launch.arguments.contains("--api-key"))
        XCTAssertFalse(launch.arguments.contains("sk-not-in-argv"))
        XCTAssertEqual(launch.commandDraft.apiKey, "sk-not-in-argv")
        XCTAssertEqual(
            CommandLaunchEnvironment.overrides(templateID: template.id, draft: launch.commandDraft)[
                CommandLaunchEnvironment.apiKeyEnvironmentKey
            ],
            "sk-not-in-argv"
        )
    }

    /// The Custom row has no capability; it keeps the catalog's raw-argument path.
    func testTheCustomTemplateRunsWhatIsTyped() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var draft = StudioConsoleDraft()
        XCTAssertNotNil(
            StudioConsoleRun(template: template, draft: draft, seed: template.defaultDraft())?.validationMessage,
            "an empty command line cannot run"
        )
        draft.extraArguments = "model list --json"
        let launch = try XCTUnwrap(StudioConsoleRun(template: template, draft: draft, seed: template.defaultDraft()))
        XCTAssertNil(launch.validationMessage)
        XCTAssertEqual(launch.arguments, ["model", "list", "--json"])
    }

    /// A template that hands off to another product has no command to build.
    func testAnExternalTemplateHasNothingToLaunch() throws {
        let external = try XCTUnwrap(CommandCatalog.templates.first { $0.externalURL != nil })
        XCTAssertNil(StudioConsoleRun(template: external, draft: StudioConsoleDraft(), seed: CommandDraft()))
    }

    /// A run the console starts is a normal inference job: the same queue, progress, artifact
    /// resolution and Library row a run from a designed surface gets.
    @MainActor
    func testAConsoleRunLaunchesTheArgumentsItShows() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        let template = try XCTUnwrap(CommandCatalog.template(id: .modelList))
        controller.select(template)
        let capability = try XCTUnwrap(template.id.capability)
        var draft = StudioConsoleCommand.seed(template: template, draft: controller.draft)
        draft.extraArguments = "--json"

        let arguments = StudioConsoleCommand.arguments(for: capability, draft: draft)
        controller.runConsole(
            template: template,
            draft: controller.draft,
            arguments: arguments,
            requestID: UUID()
        )
        let launched = try XCTUnwrap(runner.starts.last).configuration.arguments
        XCTAssertEqual(Array(launched.suffix(3)), ["model", "list", "--json"])
    }

    // MARK: Helpers

    /// The command as a set of comparable units: the positional tail by index, and every
    /// flag/value pair, regardless of the order the two builders emit them in.
    ///
    /// A template's default argv carries placeholders for what the user has yet to fill in —
    /// `--data ""`, an empty positional — that the console leaves out rather than passing an
    /// empty string to the CLI. Those are dropped from both sides; a bare boolean flag, which
    /// parses with no value at all, is kept.
    private static func pairs(of arguments: [String], capability: MereRunCommandCapability) -> Set<String> {
        let parsed = StudioCommandRows.parse(arguments: arguments, commandPathCount: capability.command.count)
        var units = Set(
            parsed.positional.enumerated()
                .filter { !$0.element.isEmpty }
                .map { "\($0.offset)=\($0.element)" }
        )
        for (flag, value) in parsed.flags {
            guard let value else {
                units.insert(flag)
                continue
            }
            guard !value.isEmpty else { continue }
            units.insert("\(flag)=\(value)")
        }
        return units
    }
}
