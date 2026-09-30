import XCTest

@testable import AntiSlopCore

final class WideningRuleTests: XCTestCase {
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

    // MARK: no-known-value-widening

    func testAnyAnnotationOnLiteralIsRejected() {
        assertViolation("let payload: Any = [\"a\": 1]", NoKnownValueWideningRule.self)
    }

    func testAnyAnnotationOnArrayLiteralIsRejected() {
        assertViolation("let items: AnyObject = [1, 2, 3]", NoKnownValueWideningRule.self)
    }

    func testOpenDictionaryAnnotationWithKnownKeysIsAllowed() {
        // Idiomatic Swift for small typed constants; upstream's Record slice
        // does not port.
        assertClean(
            "struct Handler {}; let h = Handler(); let handlers: [String: Handler] = [\"start\": h]",
            NoKnownValueWideningRule.self
        )
    }

    func testNonLiteralKeysAreClean() {
        assertClean(
            "let handlers: [String: Int] = [key: 1]",
            NoKnownValueWideningRule.self
        )
    }

    func testEmptyDictionaryAccumulatorIsClean() {
        assertClean("var handlers: [String: Int] = [:]", NoKnownValueWideningRule.self)
    }

    func testInferredDictionaryIsClean() {
        assertClean("let handlers = [\"start\": 1]", NoKnownValueWideningRule.self)
    }

    func testCallExpressionInitializerIsClean() {
        assertClean("func load() -> Any { 1 }\nlet payload: Any = load()", NoKnownValueWideningRule.self)
    }

    func testIdentifierInitializerIsClean() {
        assertClean("let source = 1\nlet payload: Any = source", NoKnownValueWideningRule.self)
    }

    func testOptionalNilInitializerIsClean() {
        assertClean("var selection: Any? = nil", NoKnownValueWideningRule.self)
    }

    func testQualifiedFoundationAnyIsRejected() {
        assertViolation("let payload: Foundation.Any = [1, 2]", NoKnownValueWideningRule.self)
        assertViolation("func load() -> Foundation.Any { 1 }", NoAnyReturnsRule.self)
    }

    func testQualifiedOtherMemberTypeIsClean() {
        assertClean("struct Foo { struct Any {} }; let v: Foo.Any = Foo.Any()", NoKnownValueWideningRule.self)
    }

    // MARK: no-widen-then-assert

    func testWidenThenForceCastIsRejected() {
        assertViolation(
            """
            struct User {}
            func load() -> User { User() }
            let loaded = load()
            let stored: Any = loaded
            let user = stored as! User
            """,
            NoWidenThenAssertRule.self
        )
    }

    func testWidenThenConditionalCastIsRejected() {
        assertViolation(
            """
            struct User {}
            func load() -> User { User() }
            let stored: Any = load()
            if let user = stored as? User {}
            """,
            NoWidenThenAssertRule.self
        )
    }

    func testCastViaAsAnyInitializerIsRejected() {
        assertViolation(
            """
            struct User {}
            func load() -> User { User() }
            let stored = load() as Any
            let user = stored as! User
            """,
            NoWidenThenAssertRule.self
        )
    }

    func testWidenedBindingWithoutLaterCastIsClean() {
        assertClean(
            """
            func parse() -> String { "x" }
            let stored: Any = parse()
            print(stored)
            """,
            NoWidenThenAssertRule.self
        )
    }

    func testNarrowBindingWithCastIsClean() {
        assertClean(
            """
            func load() -> Any { 1 }
            let stored = load()
            let count = stored as! Int
            """,
            NoWidenThenAssertRule.self
        )
    }

    func testShadowedNameDoesNotFire() {
        assertClean(
            """
            struct User {}
            func load() -> Any { 1 }
            let stored: Any = load()
            func inner(stored: User) {
                let user = stored as! User
            }
            """,
            NoWidenThenAssertRule.self
        )
    }

    func testCastBeforeDeclarationDoesNotFire() {
        assertClean(
            """
            struct User {}
            let user = stored as! User
            let stored: Any = 1
            """,
            NoWidenThenAssertRule.self
        )
    }

    // MARK: no-runtime-type-sniffing (metatype comparison slice)

    func testDynamicTypeEqualityIsRejected() {
        assertViolation(
            "class Dog {}\nclass Animal {}\nif type(of: a) == Dog.self {}",
            NoRuntimeTypeSniffingRule.self
        )
    }

    func testDynamicTypeEqualityWithSafetyCommentIsClean() {
        assertClean(
            """
            class Dog {}
            // SAFETY: exact-class check guards a decode fast path.
            if type(of: a) == Dog.self {}
            """,
            NoRuntimeTypeSniffingRule.self
        )
    }

    func testDynamicTypeComparedToDynamicTypeIsRejected() {
        assertViolation(
            "if type(of: a) == type(of: b) {}",
            NoRuntimeTypeSniffingRule.self
        )
    }

    func testOrdinaryEqualityChecksAreClean() {
        assertClean("if a == b {}", NoRuntimeTypeSniffingRule.self)
        assertClean("let t = type(of: a)", NoRuntimeTypeSniffingRule.self)
    }
}
