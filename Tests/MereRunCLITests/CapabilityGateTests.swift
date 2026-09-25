import ArgumentParser
import Foundation
import MereRunContract
import MereRunCore
import Testing
import XCTest

@testable import MereRunCLI

// MARK: - Generated cases: every routed capability, every family, every option

/// One command line and what the gate must do with it.
private struct GateCase: CustomStringConvertible {
    enum Expectation: Equatable {
        /// Runs on `family` with no violation and no warning.
        case passes(family: String)
        /// Runs on `family` with exactly one warning, naming `flag`.
        case warns(family: String, flag: String)
        /// Refused with a message naming `flag` and the family's title.
        case rejects(flag: String, familyTitle: String)
    }

    let arguments: [String]
    let expectation: Expectation

    var description: String { "\(arguments.joined(separator: " ")) → \(expectation)" }
}

/// Cases derived from the contract, not written by hand: a minimal command line per family, and
/// that line plus each option in scope (passes), ignored (warns), or out of scope (rejects),
/// plus each family rule's allowed, disallowed, missing, and excess values.
private func gateCases(for capability: MereRunCommandCapability) -> [GateCase] {
    guard let routing = capability.routing else { return [] }
    let routingFlags = Set(
        routing.modelFlags + routing.families.compactMap(\.modelFlag)
            + routing.families.flatMap(\.selectors).map(\.flag)
            + routing.defaultModels.flatMap(\.whenAny).map(\.flag)
    )
    var cases: [GateCase] = []
    for family in routing.families {
        guard let base = minimalArguments(for: family, in: capability, routing: routing) else { continue }
        cases.append(GateCase(arguments: base, expectation: .passes(family: family.id)))
        for option in capability.options where !routingFlags.contains(option.flag) {
            let rule = option.familyRules.first { $0.family == family.id }
            let withOption = { (values: [String]) in base + values.flatMap { tokens(option, value: $0) } }
            let severe = { (severity: MereRunOptionViolation.Severity) -> GateCase.Expectation in
                severity == .error
                    ? .rejects(flag: option.flag, familyTitle: family.title)
                    : .warns(family: family.id, flag: option.flag)
            }
            guard option.families?.contains(family.id) ?? true else {
                let expectation: GateCase.Expectation = option.ignoredBy.contains(family.id)
                    ? .warns(family: family.id, flag: option.flag)
                    : .rejects(flag: option.flag, familyTitle: family.title)
                cases.append(GateCase(arguments: withOption([validValue(option, rule: rule)]), expectation: expectation))
                continue
            }
            if rule?.required == true {
                let without = base.enumerated().filter { index, token in
                    token != option.flag && !(index > 0 && base[index - 1] == option.flag)
                }.map(\.element)
                cases.append(GateCase(arguments: without, expectation: .rejects(flag: option.flag, familyTitle: family.title)))
                continue
            }
            cases.append(GateCase(arguments: withOption([validValue(option, rule: rule)]), expectation: .passes(family: family.id)))
            guard let rule else { continue }
            if let allowed = rule.values, let outside = valueOutside(allowed, option: option) {
                cases.append(GateCase(arguments: withOption([outside]), expectation: severe(rule.severity)))
            }
            if let range = rule.range {
                let outside = range.max.map { $0 + 1 } ?? range.min.map { $0 - 1 }
                if let outside {
                    cases.append(GateCase(arguments: withOption([render(outside, option)]), expectation: severe(rule.severity)))
                }
            }
            if let maximum = rule.maxCount {
                let values = Array(repeating: validValue(option, rule: rule), count: maximum + 1)
                cases.append(GateCase(arguments: withOption(values), expectation: severe(rule.severity)))
            }
        }
    }
    return cases
}

/// The selector flags, a representative model, and every option the family requires.
private func minimalArguments(
    for family: MereRunRuntimeFamily,
    in capability: MereRunCommandCapability,
    routing: MereRunCapabilityRouting
) -> [String]? {
    var arguments = family.selectors.flatMap { selectorTokens($0, in: capability) }
    if let model = family.models.first {
        guard let flag = family.modelFlag ?? routing.modelFlags.last else { return nil }
        arguments += [flag, model]
    } else if !routing.routesBySelectors {
        if let rule = routing.defaultModels.first(where: { $0.family == family.id }) {
            arguments += rule.whenAny.first.map { [$0.flag] + ($0.values.map { [$0[0]] } ?? []) } ?? []
        } else if !routing.identifiedModels.isEmpty, let flag = routing.modelFlags.last {
            arguments += [flag, identifiedPlaceholder + family.id]
        } else {
            return nil
        }
    }
    for option in capability.options where option.familyRules.contains(where: { $0.family == family.id && $0.required }) {
        arguments += tokens(option, value: validValue(option, rule: option.familyRules.first { $0.family == family.id }))
    }
    return arguments
}

/// Stands for a checkpoint of the named family in a capability whose family is known only once
/// the CLI's identifier inspects what is installed (`routing.identifiedModels`). The identifier's
/// own answers are tested against fixture folders by each domain; these cases test the scope.
private let identifiedPlaceholder = "identified-family:"

/// The gate's report for a generated command line: the CLI gate's, with the identifier answering
/// placeholders for the family they name.
private func gateReport(_ capability: MereRunCommandCapability, _ arguments: [String]) -> MereRunFamilyResolutionReport {
    guard arguments.contains(where: { $0.hasPrefix(identifiedPlaceholder) }) else {
        return CLICapabilityGate.evaluate(commandLine: capability.command + arguments)!.report
    }
    let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
    return capability.resolutionReport(invocation, platform: CLICapabilityGate.platform) { model in
        model.hasPrefix(identifiedPlaceholder)
            ? .family(String(model.dropFirst(identifiedPlaceholder.count)))
            : ModelFamilyIdentifier.identify(capabilityID: capability.id, model: model, invocation: invocation)
    }
}

/// The tokens that make `condition` hold: none for an absent flag, the flag for a Boolean, and
/// the flag with an allowed or sample value otherwise.
private func selectorTokens(_ condition: MereRunFlagCondition, in capability: MereRunCommandCapability) -> [String] {
    guard !condition.absent, let option = capability.options.first(where: { $0.flag == condition.flag }) else { return [] }
    return tokens(option, value: condition.values?.first ?? validValue(option, rule: nil))
}

private func tokens(_ option: MereRunCapabilityOption, value: String) -> [String] {
    option.kind == .boolean ? [option.flag] : [option.flag, value]
}

private func validValue(_ option: MereRunCapabilityOption, rule: MereRunOptionFamilyRule?) -> String {
    if let value = rule?.values?.first { return value }
    if option.kind == .choice, let choice = option.choices.first { return choice }
    if let minimum = (rule?.range ?? option.range)?.min { return render(minimum, option) }
    if let value = rule?.defaultValue ?? option.defaultValue { return value }
    switch option.kind {
    case .integer: return "1"
    case .number: return "1.5"
    default: return "value"
    }
}

private func valueOutside(_ allowed: [String], option: MereRunCapabilityOption) -> String? {
    switch option.kind {
    case .choice: return option.choices.first { !allowed.contains($0) }
    case .integer, .number:
        let largest = allowed.compactMap(Double.init).max() ?? 0
        return render(largest + 1, option)
    default: return "not-\(allowed[0])"
    }
}

private func render(_ value: Double, _ option: MereRunCapabilityOption) -> String {
    option.kind == .integer ? String(Int(value)) : String(value)
}

/// Runs `cases` through `evaluate` and checks each expectation.
private func expect(
    _ cases: [GateCase],
    evaluate: ([String]) -> MereRunFamilyResolutionReport,
    context: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    for gateCase in cases {
        let report = evaluate(gateCase.arguments)
        let label = "\(context): \(gateCase)"
        switch gateCase.expectation {
        case .passes(let family):
            #expect(report.family == family && report.violations.isEmpty && report.warnings.isEmpty,
                    "\(label) got \(report)", sourceLocation: sourceLocation)
        case let .warns(family, flag):
            #expect(report.family == family && report.violations.isEmpty, "\(label) got \(report)", sourceLocation: sourceLocation)
            #expect(report.warnings.count == 1 && report.warnings.allSatisfy { $0.contains(flag) },
                    "\(label) got \(report.warnings)", sourceLocation: sourceLocation)
        case let .rejects(flag, title):
            #expect(report.violations.contains { $0.contains(flag) && $0.contains(title) },
                    "\(label) got \(report)", sourceLocation: sourceLocation)
        }
    }
}

@Test func everyRoutedCapabilityPassesTheGateForEveryFamilyAndInScopeOption() throws {
    let routed = MereRunCapabilityCatalog.document.commands.filter { $0.routing != nil }
    #expect(!routed.isEmpty)
    for capability in routed {
        let cases = gateCases(for: capability)
        let families = Set(cases.compactMap { gateCase -> String? in
            if case .passes(let family) = gateCase.expectation { return family }
            return nil
        })
        #expect(families == Set(capability.routing?.families.map(\.id) ?? []), "\(capability.id): every family gets a case")
        expect(cases, evaluate: { arguments in gateReport(capability, arguments) }, context: capability.id)
        for gateCase in cases where !gateCase.arguments.contains(where: { $0.hasPrefix(identifiedPlaceholder) }) {
            let argv = ["mere.run"] + capability.command + gateCase.arguments
            if case .rejects = gateCase.expectation {
                #expect(throws: CLICapabilityGate.Rejection.self, "\(capability.id): \(gateCase)") {
                    try CLICapabilityGate.check(arguments: argv)
                }
            } else {
                #expect(throws: Never.self, "\(capability.id): \(gateCase)") { try CLICapabilityGate.check(arguments: argv) }
            }
        }
    }
}

/// The generator itself must produce rejecting, warning, and rule cases, and the gate's
/// decision must match each; this capability has every kind of scope.
@Test func theCaseGeneratorCoversEveryKindOfScope() {
    let capability = MereRunCommandCapability(
        id: "demo.render", command: ["demo", "render"], title: "Demo", summary: "A test capability.",
        options: [
            MereRunCapabilityOption(flag: "--model", aliases: ["-m"], label: "Model", kind: .string),
            MereRunCapabilityOption(flag: "--mode", label: "Mode", kind: .choice, choices: ["a", "b"], defaultValue: "a"),
            MereRunCapabilityOption(flag: "--steps", label: "Steps", kind: .integer,
                                    familyRules: [.init(family: "fast", values: ["4"])]),
            MereRunCapabilityOption(flag: "--cfg", label: "CFG", kind: .number, families: ["full"], ignoredBy: ["fast"]),
            MereRunCapabilityOption(flag: "--image", label: "Image", kind: .file, repeatable: true, families: ["full", "edit"],
                                    familyRules: [.init(family: "edit", required: true),
                                                  .init(family: "full", maxCount: 1, severity: .warning)]),
            MereRunCapabilityOption(flag: "--seed", label: "Seed", kind: .integer, families: ["full", "fast"],
                                    familyRules: [.init(family: "full", range: .init(min: 0, max: 99))]),
            MereRunCapabilityOption(flag: "--hq", label: "HQ", kind: .boolean, families: ["full"])
        ],
        output: .init(kind: .text),
        routing: MereRunCapabilityRouting(
            modelFlags: ["--model"],
            defaultModels: [.init(models: ["demo-fast"])],
            families: [
                .init(id: "fast", title: "Fast", models: ["demo-fast"]),
                .init(id: "full", title: "Full", models: ["demo-full"]),
                .init(id: "edit", title: "Edit", models: ["demo-edit"], selectors: [.init(flag: "--mode", values: ["b"])])
            ]
        )
    )
    let cases = gateCases(for: capability)
    let kinds = cases.map { gateCase -> String in
        switch gateCase.expectation {
        case .passes: "passes"
        case .warns: "warns"
        case .rejects: "rejects"
        }
    }
    #expect(kinds.filter { $0 == "warns" }.count == 2, "ignored --cfg on Fast and excess --image on Full")
    #expect(kinds.filter { $0 == "rejects" }.count == 8, "\(cases)")
    #expect(cases.contains { $0.arguments == ["--model", "demo-fast", "--steps", "5"] })
    #expect(cases.contains { $0.arguments == ["--mode", "b", "--model", "demo-edit"] })
    #expect(cases.contains { $0.arguments == ["--model", "demo-full", "--seed", "100"] })
    expect(cases, evaluate: { arguments in
        capability.resolutionReport(MereRunCommandInvocation(capability: capability, arguments: arguments))
    }, context: capability.id)
}

// MARK: - Excluded models, admission, and the invocation context

@Test func excludedModelsFailAtTheGateWithTheirReason() throws {
    for capability in MereRunCapabilityCatalog.document.commands {
        guard let routing = capability.routing, let flag = routing.modelFlags.last else { continue }
        for excluded in routing.excludedModels {
            let argv = ["mere.run"] + capability.command + [flag, excluded.id]
            let expected = "\(excluded.id) can't run \(capability.command.joined(separator: " ")): \(excluded.reason)"
            #expect(throws: CLICapabilityGate.Rejection(messages: [expected])) {
                try CLICapabilityGate.check(arguments: argv)
            }
        }
    }
}

@Test func theGateSkipsHelpUnroutedCommandsAndReadsPastRootOptions() throws {
    let excluded = ["music", "analyze", "song.wav", "--model", "music-acestep-lm-4b"]
    try CLICapabilityGate.check(arguments: ["mere.run"] + excluded + ["--help"])
    try CLICapabilityGate.check(arguments: ["mere.run", "text", "chat", "hi", "--model", "music-acestep-lm-4b"])
    #expect(throws: CLICapabilityGate.Rejection.self) {
        try CLICapabilityGate.check(arguments: ["mere.run", "--models-root", "/tmp/models"] + excluded)
    }
    #expect(throws: CLICapabilityGate.Rejection.self) {
        try CLICapabilityGate.check(arguments: ["mere.run", "--models-root=/tmp/models"] + excluded)
    }
    let alias = try #require(CLICapabilityGate.evaluate(commandLine: ["speech", "listen", "-m", "Speech-ASR-Qwen3"]))
    #expect(alias.report.family == "qwen3-asr" && alias.report.model == "speech-asr-qwen3")
    let local = try #require(CLICapabilityGate.evaluate(commandLine: ["vision", "segment", "a.png", "--model", "/tmp/sam"]))
    #expect(local.report.source == .unidentified && local.report.violations.isEmpty)
}

@Test func catalogResolveNeverTakesAnInferencePermit() {
    let argv = ["mere.run", "catalog", "resolve", "--json", "--", "video", "generate", "x", "--model", "video-ltx25-full-bf16"]
    #expect(CLIInferenceAdmissionClassifier.request(arguments: argv) == nil)
    #expect(CLIInferenceAdmissionClassifier.request(arguments: ["mere.run", "video", "generate", "x"]) != nil)
}

/// Root validation: the gate runs for leaf commands, and before admission. The models-root
/// override it applies is process state, so these restore it like `CLIModelStoreBootstrapTests`.
final class CapabilityGateRootValidationTests: XCTestCase {
    private var originalModelsDirEnvironmentValue: String?

    override func setUp() {
        super.setUp()
        originalModelsDirEnvironmentValue = ProcessInfo.processInfo.environment[MereRunModelPaths.modelsDirEnvironmentKey]
    }

    override func tearDown() {
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        if let originalModelsDirEnvironmentValue {
            setenv(MereRunModelPaths.modelsDirEnvironmentKey, originalModelsDirEnvironmentValue, 1)
        } else {
            unsetenv(MereRunModelPaths.modelsDirEnvironmentKey)
        }
        super.tearDown()
    }

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    }

    /// ArgumentParser validates the root before decoding the leaf, so the root's
    /// `validate(arguments:)`, and with it the gate, runs for every leaf command.
    func testRootValidationRunsForLeafCommandsThroughParseAsRoot() throws {
        let root = temporaryRoot()
        let command = try MereRunCLI.parseAsRoot(["--models-root", root.path, "music", "analyze", "song.wav"])
        XCTAssertTrue(command is MusicAnalyze)
        XCTAssertEqual(MereRunModelPaths.modelsDir.standardizedFileURL.path, root.standardizedFileURL.path)
    }

    func testRootValidationRejectsBeforeAdmission() throws {
        var command = MereRunCLI()
        command.modelsRoot = temporaryRoot().path
        var admitted: [[String]] = []
        let rejected = ["mere.run", "music", "analyze", "song.wav", "--model", "music-acestep-lm-4b"]
        XCTAssertThrowsError(try command.validate(arguments: rejected) { admitted.append($0) }) { error in
            XCTAssertEqual(
                (error as? CLICapabilityGate.Rejection)?.errorDescription,
                "music-acestep-lm-4b can't run music analyze: It is an ACE-Step language model; pass it as `--lm-model`."
            )
        }
        XCTAssertEqual(admitted, [])

        let accepted = ["mere.run", "music", "analyze", "song.wav", "--model", "music-acestep"]
        try command.validate(arguments: accepted) { admitted.append($0) }
        XCTAssertEqual(admitted, [accepted])
        XCTAssertEqual(CLIInvocationContext.report?.capability, "music.analyze")
        XCTAssertEqual(CLIInvocationContext.family, "ace-step")
    }
}
