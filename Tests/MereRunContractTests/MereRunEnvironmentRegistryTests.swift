import Foundation
import MereRunContract
import XCTest

final class MereRunEnvironmentRegistryTests: XCTestCase {
    func testRegistryVariablesAreUniqueAndWellFormed() {
        let variables = MereRunEnvironmentRegistry.variables
        XCTAssertGreaterThanOrEqual(variables.count, 280)

        var seen: Set<String> = []
        for variable in variables {
            XCTAssertTrue(
                variable.name.hasPrefix(MereRunEnvironmentRegistry.prefix),
                "Expected \(variable.name) to start with \(MereRunEnvironmentRegistry.prefix)"
            )
            XCTAssertFalse(
                variable.name.hasSuffix("_"),
                "Exact variable \(variable.name) should not end with an underscore; register prefixes separately."
            )
            XCTAssertFalse(variable.owner.isEmpty, "Missing owner for \(variable.name)")
            XCTAssertFalse(variable.summary.isEmpty, "Missing summary for \(variable.name)")
            XCTAssertTrue(seen.insert(variable.name).inserted, "Duplicate variable \(variable.name)")
            if variable.category == .security {
                XCTAssertTrue(variable.isSecret, "Security variable \(variable.name) must redact its value")
            }
        }
    }

    func testValueValidationAcrossKinds() {
        XCTAssertNil(MereRunEnvironmentValueKind.bool.validate("1"))
        XCTAssertNil(MereRunEnvironmentValueKind.bool.validate("TRUE"))
        XCTAssertNil(MereRunEnvironmentValueKind.bool.validate("off"))
        XCTAssertNil(MereRunEnvironmentValueKind.bool.validate("   "))
        XCTAssertNotNil(MereRunEnvironmentValueKind.bool.validate("maybe"))

        XCTAssertNil(MereRunEnvironmentValueKind.integer.validate("256"))
        XCTAssertNil(MereRunEnvironmentValueKind.integer.validate("-1"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.integer.validate("12.5"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.integer.validate("abc"))

        XCTAssertNil(MereRunEnvironmentValueKind.number.validate("0.4"))
        XCTAssertNil(MereRunEnvironmentValueKind.number.validate("42"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.number.validate("nan"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.number.validate("inf"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.number.validate("nope"))

        XCTAssertNil(MereRunEnvironmentValueKind.flag.validate("0"))
        XCTAssertNil(MereRunEnvironmentValueKind.flag.validate("1"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.flag.validate("true"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.flag.validate(" 1"))
        XCTAssertNotNil(MereRunEnvironmentValueKind.integer.validate(" 256"))

        let choice = MereRunEnvironmentValueKind.choice(["0", "1", "none"])
        XCTAssertNil(choice.validate("NONE"))
        XCTAssertNil(choice.validate("1"))
        XCTAssertNotNil(choice.validate("all"))
    }

    /// A value that validates has to behave as written. Runtime switches share one parser, so
    /// they accept the whole vocabulary; modes and strides keep their own values.
    func testValueKindsMatchHowReadersParseValues() {
        func expected(_ name: String, _ value: String) -> String? {
            MereRunEnvironmentRegistry.variable(named: name)?.valueKind.validate(value)
        }
        for name in ["MERERUN_Q38_TUNED_DEFAULTS", "MERERUN_Q35_PRISM_METAL", "MERERUN_FUSED_SDPA"] {
            for value in ["1", "true", "yes", "on", "0", "false", "no", "off"] {
                XCTAssertNil(expected(name, value), "\(name)=\(value)")
            }
            XCTAssertNotNil(expected(name, "sometimes"))
        }
        // Read outside the runtime parser by build scripts and test fixtures.
        XCTAssertNotNil(expected("MERERUN_SWIFT_DISABLE_INDEX_STORE", "true"))
        XCTAssertNil(expected("MERERUN_SWIFT_DISABLE_INDEX_STORE", "1"))
        // Modes and strides.
        XCTAssertNil(expected("MERERUN_Q35_PRISM_FUSION", "shared"))
        XCTAssertNil(expected("MERERUN_LAGUNA_PREFILL_ASYNC_LADDER", "16"))
        XCTAssertNotNil(expected("MERERUN_LAGUNA_PREFILL_ASYNC_LADDER", "41"))
        XCTAssertNotNil(expected("MERERUN_Q38_DECODE_ASYNC_BLOCKS", "2"))
        XCTAssertNil(expected("MERERUN_H3_CACHE_STRATEGY", "scheduled-tail"))
    }

    /// Runtime switches must be read through `MereRunEnvironmentSwitch`, so a registered `bool`
    /// variable may not be subscripted out of an environment dictionary and compared by hand.
    /// Writes (`env["X"] = "1"`) and presence checks (`env["X"] == nil`) stay allowed.
    func testRuntimeSwitchesAreReadThroughTheSharedParser() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MereRunContractTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // repo root
        let switches = MereRunEnvironmentRegistry.variables(ofKind: .bool).map(\.name)
        // A subscript follows an expression (`env[`, `environment?[`); array literals like
        // `keys: ["MERERUN_X"]` hand the name to the shared parser and are allowed.
        let pattern = #"(?<=[\w\)\]\?])\[\s*"(MERERUN_[A-Z0-9_]+)"\s*\](?!\s*(=[^=]|== nil))"#
        let regex = try NSRegularExpression(pattern: pattern)
        let switchSet = Set(switches)
        var handParsed: [String: Set<String>] = [:]

        let sources = repoRoot.appendingPathComponent("Sources")
        let enumerator = try XCTUnwrap(
            FileManager.default.enumerator(at: sources, includingPropertiesForKeys: [.isRegularFileKey])
        )
        for case let fileURL as URL in enumerator where fileURL.pathExtension == "swift" {
            let content = try String(contentsOf: fileURL, encoding: .utf8)
            guard content.contains("MERERUN_") else { continue }
            let nsContent = content as NSString
            for match in regex.matches(in: content, range: NSRange(location: 0, length: nsContent.length)) {
                let name = nsContent.substring(with: match.range(at: 1))
                guard switchSet.contains(name) else { continue }
                let relativePath = fileURL.path.replacingOccurrences(of: repoRoot.path + "/", with: "")
                handParsed[name, default: []].insert(relativePath)
            }
        }

        let details = handParsed.keys.sorted().map { name in
            "  - \(name) (\(handParsed[name, default: []].sorted().joined(separator: ", ")))"
        }
        XCTAssertTrue(
            details.isEmpty,
            "Read these switches with MereRunEnvironmentSwitch instead of comparing raw values:\n"
                + details.joined(separator: "\n")
        )
    }

    func testRemoteAuditRedactsPathsAndStringsButKeepsSwitches() {
        let audit = MereRunEnvironmentRegistry.redactingFreeformValues(
            MereRunEnvironmentRegistry.audit(environment: [
                "MERERUN_MODELS_DIR": "/Users/someone/models",
                "MERERUN_INKLING2_PREFILL_CHUNK": "512",
                "MERERUN_Q35_PRISM_METAL": "0",
                "MERERUN_API_KEY": "sk-secret",
            ])
        )
        let byName = Dictionary(uniqueKeysWithValues: audit.activeOverrides.map { ($0.name, $0.value) })
        XCTAssertEqual(byName["MERERUN_MODELS_DIR"], MereRunEnvironmentRegistry.redactedPlaceholder)
        XCTAssertEqual(byName["MERERUN_API_KEY"], MereRunEnvironmentRegistry.redactedPlaceholder)
        XCTAssertEqual(byName["MERERUN_INKLING2_PREFILL_CHUNK"], "512")
        XCTAssertEqual(byName["MERERUN_Q35_PRISM_METAL"], "0")
    }

    /// Build, release, and benchmark scripts export their own `MERERUN_*` settings into shells
    /// that also run `mere.run`. They stay unregistered and must not produce warnings.
    func testNamesThatResembleNoRegisteredVariableAreNotReported() {
        let audit = MereRunEnvironmentRegistry.audit(environment: [
            "MERERUN_CUDA_ARCHITECTURES": "80;90",
            "MERERUN_NATIVE_BUILD_JOBS": "8",
            "MERERUN_H3_BENCH_FRAMES": "49",
        ])
        XCTAssertTrue(audit.isEmpty)
        XCTAssertNil(MereRunEnvironmentRegistry.variable(named: "MERERUN_CUDA_ARCHITECTURES"))
    }

    func testSecretRedactionForExactVariablesAndDynamicPrefix() {
        let audit = MereRunEnvironmentRegistry.audit(environment: [
            "MERERUN_API_KEY": "sk-live-super-secret",
            "MERERUN_RELAY_TOKEN": "relay-secret-token",
            "MERERUN_SECRET_HF_TOKEN": "hf_secret_123",
            "MERERUN_INKLING2_PREFILL_CHUNK": "512",
            "PATH": "/usr/bin:/bin",
        ])

        XCTAssertFalse(audit.hasIssues)
        XCTAssertTrue(audit.warnings.isEmpty)
        XCTAssertEqual(audit.activeOverrides.count, 4)

        let byName = Dictionary(uniqueKeysWithValues: audit.activeOverrides.map { ($0.name, $0) })
        XCTAssertEqual(byName["MERERUN_API_KEY"]?.value, MereRunEnvironmentRegistry.redactedPlaceholder)
        XCTAssertEqual(byName["MERERUN_API_KEY"]?.isSecret, true)
        XCTAssertEqual(byName["MERERUN_RELAY_TOKEN"]?.value, MereRunEnvironmentRegistry.redactedPlaceholder)
        XCTAssertEqual(byName["MERERUN_SECRET_HF_TOKEN"]?.value, MereRunEnvironmentRegistry.redactedPlaceholder)
        XCTAssertEqual(byName["MERERUN_SECRET_HF_TOKEN"]?.isSecret, true)
        XCTAssertEqual(byName["MERERUN_INKLING2_PREFILL_CHUNK"]?.value, "512")
        XCTAssertEqual(byName["MERERUN_INKLING2_PREFILL_CHUNK"]?.isSecret, false)
    }

    func testTypoSuggestionsAndInvalidValues() {
        let audit = MereRunEnvironmentRegistry.audit(environment: [
            "MERERUN_INKLING2_PREFILL_CHUNKS": "512",
            "MERERUN_Q38_TUNED_DEFAULT": "0",
            "MERERUN_MODLES_DIR": "/tmp/models",
            "MERERUN_COMPLETELY_UNRELATED_XYZ": "1",
            "MERERUN_INKLING2_PREFILL_CHUNK": "five-hundred",
            "MERERUN_GEMMA4_CONTINUOUS_BATCHING": "sometimes",
            "MERERUN_TEST_CUSTOM_HARNESS_FLAG": "1",
        ])

        XCTAssertTrue(audit.hasIssues)
        XCTAssertEqual(audit.activeOverrides.map(\.name), [
            "MERERUN_GEMMA4_CONTINUOUS_BATCHING",
            "MERERUN_INKLING2_PREFILL_CHUNK",
            "MERERUN_TEST_CUSTOM_HARNESS_FLAG",
        ])

        let unknownByName = Dictionary(uniqueKeysWithValues: audit.unknownVariables.map { ($0.name, $0.suggestion) })
        XCTAssertEqual(unknownByName["MERERUN_INKLING2_PREFILL_CHUNKS"], "MERERUN_INKLING2_PREFILL_CHUNK")
        XCTAssertEqual(unknownByName["MERERUN_Q38_TUNED_DEFAULT"], "MERERUN_Q38_TUNED_DEFAULTS")
        XCTAssertEqual(unknownByName["MERERUN_MODLES_DIR"], "MERERUN_MODELS_DIR")
        XCTAssertNil(unknownByName["MERERUN_COMPLETELY_UNRELATED_XYZ"])

        let invalidByName = Dictionary(uniqueKeysWithValues: audit.invalidValues.map { ($0.name, $0) })
        XCTAssertEqual(invalidByName["MERERUN_INKLING2_PREFILL_CHUNK"]?.expected, "expected an integer")
        XCTAssertEqual(
            invalidByName["MERERUN_GEMMA4_CONTINUOUS_BATCHING"]?.expected,
            "expected a boolean (0, 1, true, false, yes, no, on, off)"
        )
        XCTAssertEqual(audit.warnings.count, 5)
        XCTAssertTrue(audit.warnings.contains(where: { $0.contains("did you mean MERERUN_INKLING2_PREFILL_CHUNK?") }))
    }

    func testAuditCodableRoundTrip() throws {
        let audit = MereRunEnvironmentRegistry.audit(environment: [
            "MERERUN_API_KEY": "secret",
            "MERERUN_INKLING2_PREFILL_CHUNK": "bad",
            "MERERUN_MODLES_DIR": "/tmp/models",
        ])
        let data = try JSONEncoder().encode(audit)
        let decoded = try JSONDecoder().decode(MereRunEnvironmentAudit.self, from: data)
        XCTAssertEqual(decoded, audit)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("secret"))
    }

    /// Repository drift guard: every `MERERUN_*` variable literal referenced in production Swift
    /// code (`Sources/` and non-test `apps/`) must be registered in `MereRunEnvironmentRegistry`.
    func testEveryProductionEnvironmentVariableIsRegistered() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MereRunContractTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // repo root

        let scanRoots = [
            repoRoot.appendingPathComponent("Sources"),
            repoRoot.appendingPathComponent("apps"),
        ]
        let regex = try NSRegularExpression(pattern: #""(MERERUN_[A-Z0-9_]+)""#)
        var unregistered: [String: [String]] = [:]

        for root in scanRoots {
            guard FileManager.default.fileExists(atPath: root.path),
                  let enumerator = FileManager.default.enumerator(
                      at: root,
                      includingPropertiesForKeys: [.isRegularFileKey],
                      options: [.skipsHiddenFiles]
                  ) else {
                continue
            }
            for case let fileURL as URL in enumerator {
                guard fileURL.pathExtension == "swift" else { continue }
                let relativePath = fileURL.path.replacingOccurrences(of: repoRoot.path + "/", with: "")
                if relativePath.contains("Tests/") || relativePath.contains("Tests.swift") {
                    continue
                }
                let content = try String(contentsOf: fileURL, encoding: .utf8)
                guard content.contains("MERERUN_") else { continue }
                let nsContent = content as NSString
                let matches = regex.matches(in: content, range: NSRange(location: 0, length: nsContent.length))
                for match in matches {
                    let name = nsContent.substring(with: match.range(at: 1))
                    if name.hasSuffix("_") { continue }
                    if !MereRunEnvironmentRegistry.isKnown(name) {
                        unregistered[name, default: []].append(relativePath)
                    }
                }
            }
        }

        if !unregistered.isEmpty {
            let details = unregistered.keys.sorted().map { name in
                let files = Array(Set(unregistered[name] ?? [])).sorted().joined(separator: ", ")
                return "  - \(name) (\(files))"
            }.joined(separator: "\n")
            XCTFail(
                "Unregistered MERERUN_* environment variable(s) found in production Swift code.\n"
                    + "Add them to Sources/MereRunContract/MereRunEnvironmentCatalog.swift:\n\(details)"
            )
        }
    }
}
