@testable import MereRunApp
import MereRunContract
import XCTest

/// Guards the app's argv builder against the shared CLI contract.
///
/// `StudioTypesTests.testEveryLocalAdvancedTemplateIsBackedByTheSharedCLIContract` only
/// builds argv from each template's default draft, so a flag that is emitted only when a
/// field is set is never checked. These tests build probe drafts that reach every guarded
/// branch of `CommandTemplate.arguments(from:)` and assert that each emitted `--flag` is
/// declared for the template's capability in `MereRunCapabilityCatalog`.
final class CommandContractGuardTests: XCTestCase {
    func testEveryFlagEmittedFromProbeDraftsIsDeclaredByTheContract() throws {
        var driftedCommandPaths: [String] = []
        var undeclared: [String: String] = [:]

        for emission in try CommandDraftProbes.emissions.get() {
            let template = emission.template
            let capabilityID = try XCTUnwrap(template.id.capabilityID)
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: capabilityID))
            let declared = Set(capability.options.map(\.flag))

            if Array(emission.arguments.prefix(capability.command.count)) != capability.command {
                driftedCommandPaths.append("\(template.id) (probe: \(emission.probe))")
            }
            for flag in emission.arguments where flag.hasPrefix("--") && !declared.contains(flag) {
                let key = "\(template.id) emits \(flag), which \(capabilityID) does not declare"
                if undeclared[key] == nil { undeclared[key] = emission.probe }
            }
        }

        XCTAssertEqual(driftedCommandPaths, [], "Command paths drifted from the contract")
        XCTAssertEqual(
            undeclared.map { "\($0.key) (first probe: \($0.value))" }.sorted(),
            [],
            """
            The app emits flags the shared contract does not declare. Add each to \
            CommandCapabilityContract.swift if the CLI accepts it, or stop emitting it if the CLI \
            does not.
            """
        )
    }

    /// The maximal draft must move every field off its default. If this fails, `CommandDraft`
    /// gained a field whose type `CommandDraftProbes` cannot synthesize a value for.
    func testMaximalDraftSetsEveryField() throws {
        let defaults = CommandDraft()
        let maximal = try CommandDraftProbes.maximalDraft(from: defaults, booleans: true)
        let defaultChildren = Array(Mirror(reflecting: defaults).children)
        let maximalChildren = Array(Mirror(reflecting: maximal).children)
        XCTAssertEqual(defaultChildren.count, maximalChildren.count)

        for (before, after) in zip(defaultChildren, maximalChildren) {
            let label = try XCTUnwrap(before.label)
            XCTAssertEqual(label, after.label)
            if after.value is Bool {
                XCTAssertEqual(after.value as? Bool, true, "Maximal draft left \(label) false")
            } else {
                XCTAssertNotEqual(
                    String(describing: before.value),
                    String(describing: after.value),
                    "Maximal draft left \(label) at its default"
                )
            }
            let mirror = Mirror(reflecting: after.value)
            XCTAssertFalse(
                mirror.displayStyle == .optional && mirror.children.isEmpty,
                "Maximal draft left \(label) nil"
            )
            if let string = after.value as? String {
                XCTAssertFalse(string.isBlank, "Maximal draft left \(label) blank")
            }
        }
    }

    /// Every hand-named variant must target a real `CommandDraft` field so that renaming a
    /// field cannot silently turn a variant into a no-op.
    func testVariantOverridesNameRealDraftFields() {
        let fields = Set(Mirror(reflecting: CommandDraft()).children.compactMap(\.label))
        for field in CommandDraftProbes.variantValuesByField.keys {
            XCTAssertTrue(fields.contains(field), "Variant names unknown CommandDraft field \(field)")
        }
    }

    /// Reads the flag literals in `CommandTemplate.arguments(from:)` and asserts that the
    /// probe drafts emit each of them for at least one template. Without this, a guard the
    /// generic per-type rule cannot satisfy (for example `field == "specific-choice"`) would
    /// leave its flag unverified while every other test stayed green. A new literal that
    /// fails here needs a named entry in `CommandDraftProbes.variantValuesByField`.
    func testProbeDraftsReachEveryFlagLiteralInTheArgumentBuilder() throws {
        let literals = try CommandCatalogSource.argumentBuilderFlagLiterals()
        XCTAssertGreaterThan(literals.count, 400, "Flag literal scan found too few flags to be trusted")

        var emitted = Set<String>()
        for emission in try CommandDraftProbes.emissions.get() {
            for flag in emission.arguments where flag.hasPrefix("--") {
                emitted.insert(flag)
            }
        }

        let unreached = literals.subtracting(emitted).sorted()
        XCTAssertEqual(
            unreached,
            [],
            """
            arguments(from:) contains flag literals no probe draft reaches. Add a named variant \
            to CommandDraftProbes.variantValuesByField that satisfies the guard in front of each.
            """
        )
    }
}

private extension CommandTemplate {
    /// Templates that hand off to another product or run the user's raw command line do not
    /// build argv against a capability, matching `testEveryLocalAdvancedTemplateIsBackedByTheSharedCLIContract`.
    var buildsLocalArguments: Bool {
        externalURL == nil && id != .custom
    }
}

/// Builds `CommandDraft` values that exercise the argv builder without a per-field table.
///
/// Fields are discovered with `Mirror` and set by type: every `Bool` gets the requested
/// polarity, every `Int` and `Double` moves to a non-default positive value, every `String`
/// becomes non-blank, every raw-value enum takes a non-default case, every optional is set,
/// and every array is non-empty. The draft is materialised through its own `Codable`
/// conformance, so the synthesized coding keys are the only field names the builder relies on.
enum CommandDraftProbes {
    struct Probe {
        let name: String
        let draft: CommandDraft
    }

    struct Emission {
        let template: CommandTemplate
        let probe: String
        let arguments: [String]
    }

    /// Argv for every probe of every local template, built once and shared by the tests.
    static let emissions: Result<[Emission], Error> = Result {
        try CommandCatalog.templates.filter(\.buildsLocalArguments).flatMap { template in
            try probes(for: template).map { probe in
                Emission(template: template, probe: probe.name, arguments: template.arguments(from: probe.draft))
            }
        }
    }

    /// The one string value that every field can carry. It parses as an integer and a
    /// floating-point number, is a valid relative path, and is never mistaken for a flag.
    static let genericString = "1"

    /// Field values the generic per-type rule cannot produce because the argv builder compares
    /// the field against a specific literal. Each entry names the guard it exists for. The
    /// literal reach test fails when a guard whose flag appears nowhere else is missing here;
    /// the per-template contract check is what the remaining entries feed.
    static let variantValuesByField: [String: [String]] = [
        // `musicLMMode == "use"` emits --use-lm; `== "disable"` emits --no-lm.
        "musicLMMode": ["use", "disable"],
        // `["repaint", "lego"].contains(musicTask)` gates the repaint window flags.
        "musicTask": ["repaint", "lego"],
        // `voiceMode == "clone"` emits `--mode clone` for speech synthesis.
        "voiceMode": ["clone"],
        // `renderProfile == "quality"` emits the SCAIL sampler overrides.
        "renderProfile": ["quality"],
        // `StudioVideoModelFamily(model:)` reads `modelRoot` when it is non-blank and selects
        // the LTX, Wan, or MiniMax-H3 branch of `video generate`.
        "modelRoot": ["video-minimax-h3-ref2va", "video-wan22-ti2v-5b-mlx"]
    ]

    static func probes(for template: CommandTemplate) throws -> [Probe] {
        let defaults = template.defaultDraft()
        var probes = [
            Probe(name: "default", draft: defaults),
            Probe(name: "maximal", draft: try maximalDraft(from: defaults, booleans: true)),
            Probe(name: "maximal-booleans-off", draft: try maximalDraft(from: defaults, booleans: false))
        ]

        var variants = variantValuesByField.map { ($0.key, $0.value.map { $0 as Any }) }
        variants += enumCaseVariants(in: defaults)
        for (field, values) in variants.sorted(by: { $0.0 < $1.0 }) {
            for value in values {
                for booleans in [true, false] {
                    probes.append(Probe(
                        name: "\(field)=\(value) booleans=\(booleans)",
                        draft: try maximalDraft(from: defaults, booleans: booleans, overriding: [field: value])
                    ))
                }
            }
        }
        return probes
    }

    static func maximalDraft(
        from defaults: CommandDraft,
        booleans: Bool,
        overriding overrides: [String: Any] = [:]
    ) throws -> CommandDraft {
        var object: [String: Any] = [:]
        for child in Mirror(reflecting: defaults).children {
            let label = try XCTUnwrap(child.label)
            object[label] = try XCTUnwrap(
                maximalJSONValue(for: child.value, booleans: booleans),
                "CommandDraft.\(label) has type \(type(of: child.value)), which CommandDraftProbes cannot synthesize"
            )
        }
        for (field, value) in overrides {
            object[field] = value
        }
        let data = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(CommandDraft.self, from: data)
    }

    /// Every raw-value enum field contributes all of its cases, so `switch` statements over
    /// enum fields are fully exercised.
    private static func enumCaseVariants(in draft: CommandDraft) -> [(String, [Any])] {
        Mirror(reflecting: draft).children.compactMap { child in
            guard let label = child.label,
                  let iterable = type(of: child.value) as? any CaseIterable.Type else { return nil }
            return (label, rawValues(of: iterable))
        }
    }

    private static func maximalJSONValue(for value: Any, booleans: Bool) -> Any? {
        switch value {
        case let bool as Bool:
            return booleans
        case let int as Int:
            return max(int, 0) + 1
        case let double as Double:
            return max(double, 0) + 0.5
        case is String:
            return genericString
        case let strings as [String]:
            return strings.isEmpty ? [genericString] : strings
        default:
            break
        }

        if let iterable = type(of: value) as? any CaseIterable.Type,
           let current = (value as? any RawRepresentable).map({ "\($0.rawValue)" }) {
            let cases = rawValues(of: iterable)
            return cases.first { "\($0)" != current } ?? cases.first
        }

        let mirror = Mirror(reflecting: value)
        if mirror.displayStyle == .optional, mirror.children.isEmpty,
           let optional = type(of: value) as? OptionalWrapping.Type {
            return maximalJSONValue(forWrapped: optional.wrappedType, booleans: booleans)
        }
        return nil
    }

    private static func maximalJSONValue(forWrapped type: Any.Type, booleans: Bool) -> Any? {
        switch type {
        case is Bool.Type: return booleans
        case is Int.Type: return 1
        case is Double.Type: return 0.5
        case is String.Type: return genericString
        case is [String].Type: return [genericString]
        default: return nil
        }
    }

    private static func rawValues(of type: any CaseIterable.Type) -> [Any] {
        func open<T: CaseIterable>(_ type: T.Type) -> [Any] {
            type.allCases.compactMap { ($0 as? any RawRepresentable)?.rawValue }
        }
        return open(type)
    }
}

private protocol OptionalWrapping {
    static var wrappedType: Any.Type { get }
}

extension Optional: OptionalWrapping {
    static var wrappedType: Any.Type { Wrapped.self }
}

/// Reads flag literals out of `CommandCatalog.swift` so the reach test can prove the probes
/// exercise every branch of the argv builder.
enum CommandCatalogSource {
    static func argumentBuilderFlagLiterals() throws -> Set<String> {
        let testsDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let source = testsDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("MereRunStudio/CommandCatalog.swift")
        let lines = try String(contentsOf: source, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)

        let start = try XCTUnwrap(
            lines.firstIndex { $0.contains("func arguments(from draft: CommandDraft) -> [String]") },
            "arguments(from:) moved; update CommandCatalogSource"
        )
        // The builder and its private helpers end where the enclosing type closes.
        let end = lines[start...].firstIndex { $0 == "}" } ?? lines.endIndex

        let pattern = try NSRegularExpression(pattern: #""(--[a-z0-9][a-z0-9-]*)""#)
        var flags = Set<String>()
        for line in lines[start..<end] {
            let text = String(line)
            let range = NSRange(text.startIndex..., in: text)
            for match in pattern.matches(in: text, range: range) {
                if let flagRange = Range(match.range(at: 1), in: text) {
                    flags.insert(String(text[flagRange]))
                }
            }
        }
        return flags
    }
}
