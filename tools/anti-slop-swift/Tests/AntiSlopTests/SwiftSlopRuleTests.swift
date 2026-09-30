import XCTest

@testable import AntiSlopCore

final class SwiftSlopRuleTests: XCTestCase {
    // MARK: Helpers

    private func lint(_ source: String, _ ruleType: SlopRule.Type) -> [Violation] {
        AntiSlop.lint(source: source, fileName: "test.swift", rules: [ruleType])
    }

    private func assertViolation(
        _ source: String,
        _ ruleType: SlopRule.Type,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let violations = lint(source, ruleType)
        XCTAssertFalse(violations.isEmpty, "expected a violation", file: file, line: line)
        XCTAssertEqual(
            violations.first?.ruleID,
            ruleType.id,
            file: file,
            line: line
        )
    }

    private func assertClean(
        _ source: String,
        _ ruleType: SlopRule.Type,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let violations = lint(source, ruleType)
        XCTAssertTrue(
            violations.isEmpty,
            "expected no violations, got \(violations)",
            file: file,
            line: line
        )
    }

    // MARK: no-fatal-error

    func testFatalErrorWithoutSafetyCommentIsRejected() {
        assertViolation("fatalError(\"unreachable\")", NoFatalErrorRule.self)
    }

    func testPreconditionFailureIsRejected() {
        assertViolation("preconditionFailure(\"nope\")", NoFatalErrorRule.self)
    }

    func testFatalErrorWithSafetyCommentIsClean() {
        assertClean(
            """
            // SAFETY: the switch above covers every case of the closed enum.
            fatalError("unreachable")
            """,
            NoFatalErrorRule.self
        )
    }

    func testOrdinaryCallsAreClean() {
        assertClean("print(\"hi\")", NoFatalErrorRule.self)
        assertClean("precondition(count > 0)", NoFatalErrorRule.self)
    }

    // MARK: no-implicitly-unwrapped-optionals

    func testIUOVariableIsRejected() {
        assertViolation("var cache: [String: Data]!", NoImplicitlyUnwrappedOptionalsRule.self)
    }

    func testIUOReturnTypeIsRejected() {
        assertViolation("func lookup() -> String! { nil }", NoImplicitlyUnwrappedOptionalsRule.self)
    }

    func testIBOutletIsExempt() {
        assertClean("@IBOutlet var label: UILabel!", NoImplicitlyUnwrappedOptionalsRule.self)
    }

    func testPlainOptionalsAreClean() {
        assertClean("var cache: [String: Data]?", NoImplicitlyUnwrappedOptionalsRule.self)
        assertClean("func lookup() -> String? { nil }", NoImplicitlyUnwrappedOptionalsRule.self)
    }

    // MARK: no-as-any-cast

    func testAsAnyWithoutSafetyCommentIsRejected() {
        assertViolation("let boxed = value as Any", NoAsAnyCastRule.self)
    }

    func testAsAnyObjectWithoutSafetyCommentIsRejected() {
        assertViolation("let boxed = value as AnyObject", NoAsAnyCastRule.self)
    }

    func testAsAnyWithSafetyCommentIsClean() {
        assertClean(
            """
            // SAFETY: heterogeneous diagnostic payload; only logging reads it.
            let boxed = value as Any
            """,
            NoAsAnyCastRule.self
        )
    }

    func testConcreteCastIsClean() {
        assertClean("let user = value as User", NoAsAnyCastRule.self)
    }

    // MARK: no-swallowed-errors

    func testEmptyCatchIsRejected() {
        assertViolation(
            """
            do {
                try run()
            } catch {}
            """,
            NoSwallowedErrorsRule.self
        )
    }

    func testCommentOnlyCatchIsStillEmpty() {
        assertViolation(
            """
            do {
                try run()
            } catch {
                // nothing to see here
            }
            """,
            NoSwallowedErrorsRule.self
        )
    }

    func testHandlingCatchIsClean() {
        assertClean(
            """
            do {
                try run()
            } catch {
                logger.report(error)
            }
            """,
            NoSwallowedErrorsRule.self
        )
    }

    func testRethrowingCatchIsClean() {
        assertClean(
            """
            do {
                try run()
            } catch let error as DecodingError {
                throw ParseError.failed(error)
            } catch {
                throw ParseError.failed(error)
            }
            """,
            NoSwallowedErrorsRule.self
        )
    }

    // MARK: no-bool-literal-comparisons

    func testTernaryOfLiteralsIsRejected() {
        assertViolation("let enabled = flag ? true : false", NoBoolLiteralComparisonsRule.self)
    }

    func testBinaryBoolLiteralComparisonsAreAllowed() {
        // Optional Bools make `x == true` meaningful; a syntax-level linter
        // cannot see operand types, so binary comparisons are not flagged.
        assertClean("if flag == true {}", NoBoolLiteralComparisonsRule.self)
        assertClean("if flag != false {}", NoBoolLiteralComparisonsRule.self)
    }

    func testDirectConditionIsClean() {
        assertClean("if flag {}", NoBoolLiteralComparisonsRule.self)
        assertClean("if !flag {}", NoBoolLiteralComparisonsRule.self)
        assertClean("let enabled = flag", NoBoolLiteralComparisonsRule.self)
    }

    func testValueComparisonAgainstBoolVariableIsClean() {
        assertClean("if lhs == rhs {}", NoBoolLiteralComparisonsRule.self)
    }

    // MARK: no-hardcoded-secrets

    func testHardcodedAPIKeyIsRejected() {
        assertViolation(
            "let apiKey = \"sk-live-abc123\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testHardcodedPasswordIsRejected() {
        assertViolation(
            "static let adminPassword = \"hunter2\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testEnvironmentLookupIsClean() {
        assertClean(
            "let apiKey = ProcessInfo.processInfo.environment[\"API_KEY\"]",
            NoHardcodedSecretsRule.self
        )
    }

    func testInnocentTokenNamesAreClean() {
        assertClean("let tokenizer = Tokenizer()", NoHardcodedSecretsRule.self)
        assertClean("let tokenCount = 3", NoHardcodedSecretsRule.self)
    }

    func testEnvironmentVariableNameLiteralsAreClean() {
        // A SCREAMING_CASE literal is an env-var name, not a committed secret.
        assertClean(
            "let apiKeyName = \"MERERUN_API_KEY\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testIdentifierShapedLiteralsWithKeyLikeNamesAreClean() {
        assertClean(
            "let evalKeyName = \"mere-run-local-eval\"",
            NoHardcodedSecretsRule.self
        )
        assertClean(
            "let defaultsKey = \"mererun.app.runtimeAPIKey\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testIdentifierShapedLiteralsWithoutKeyLikeNamesStayFlagged() {
        // Shape alone is not enough; declaration intent is required.
        assertViolation(
            "let password = \"correct-horse-battery-staple\"",
            NoHardcodedSecretsRule.self
        )
        assertViolation(
            "let apiKey = \"foo.bar\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testMixedAlphanumericValuesStayFlagged() {
        // Letter-only segment exemption must not swallow key-like tokens.
        assertViolation(
            "let apiKey = \"sk-live-abc123\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testShortGenericPasswordsStayFlagged() {
        assertViolation(
            "let adminPassword = \"admin\"",
            NoHardcodedSecretsRule.self
        )
    }

    func testEmptyAndInterpolatedStringsAreClean() {
        assertClean("var apiKey = \"\"", NoHardcodedSecretsRule.self)
        assertClean(
            """
            let raw = "stored"
            let password = "len: \\(raw.count)"
            """,
            NoHardcodedSecretsRule.self
        )
    }
}
