import Foundation
import SwiftOperators
import SwiftParser
import SwiftSyntax

/// A single rejection produced by a rule.
public struct Violation: Equatable, Sendable {
    /// File the violation was found in.
    public var fileName: String
    /// Stable identifier of the rule that reported this, e.g. "no-force-try".
    public var ruleID: String
    /// Human-readable explanation of the rejected pattern.
    public var message: String
    /// 1-based line of the offending node.
    public var line: Int
    /// 1-based column of the offending node.
    public var column: Int

    public init(
        fileName: String,
        ruleID: String,
        message: String,
        line: Int,
        column: Int
    ) {
        self.fileName = fileName
        self.ruleID = ruleID
        self.message = message
        self.line = line
        self.column = column
    }

    public var description: String {
        "[\(ruleID)] \(message)"
    }
}

/// Base class for every anti-slop rule.
///
/// A rule is a `SyntaxVisitor` that walks one parsed file and collects
/// violations. Subclasses override the `visitPost` / `visitPre` hooks they
/// care about and call `report(_:message:)` on rejected nodes.
open class SlopRule: SyntaxVisitor {
    /// Stable identifier used in output and configuration, e.g. "no-force-try".
    open class var id: String { "" }

    /// One-line description shown by `anti-slop --list-rules`.
    open class var summary: String { "" }

    public let fileName: String

    private let sourceLines: [String]
    private let converter: SourceLocationConverter
    public private(set) var violations: [Violation] = []

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        self.fileName = fileName
        self.sourceLines = sourceText
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        self.converter = converter
        super.init(viewMode: .sourceAccurate)
    }

    /// Record a rejection against the start position of `node`.
    public func report(_ node: some SyntaxProtocol, message: String) {
        let location = converter.location(for: node.positionAfterSkippingLeadingTrivia)
        violations.append(
            Violation(
                fileName: fileName,
                ruleID: Self.id,
                message: message,
                line: location.line,
                column: location.column
            )
        )
    }

    /// True when a comment containing "SAFETY:" appears in the contiguous
    /// block of non-blank comment lines immediately above `node`.
    ///
    /// This is the documented escape hatch for genuinely necessary force
    /// casts, force tries, and force unwraps: the author must state the
    /// invariant they checked by hand.
    func hasPrecedingSafetyComment(for node: some SyntaxProtocol) -> Bool {
        let line = converter.location(for: node.positionAfterSkippingLeadingTrivia).line
        var index = line - 2 // zero-based index of the previous line
        while index >= 0 {
            let trimmed = sourceLines[index].trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { return false }
            if trimmed.contains("SAFETY:") { return true }
            index -= 1
        }
        return false
    }
}

extension SlopRule {
    /// Every rule shipped with anti-slop, in stable display order.
    public static var allRules: [SlopRule.Type] {
        [
            NoChainedTypeCastsRule.self,
            RequireSafetyCommentForForcedCastRule.self,
            NoForceTryRule.self,
            NoForceUnwrapRule.self,
            NoFatalErrorRule.self,
            NoImplicitlyUnwrappedOptionalsRule.self,
            NoAsAnyCastRule.self,
            NoAnyParametersRule.self,
            NoAnyReturnsRule.self,
            NoAnyTypeAliasesRule.self,
            NoAnyDictionaryValueRule.self,
            NoKnownValueWideningRule.self,
            NoWidenThenAssertRule.self,
            NoSwallowedErrorsRule.self,
            NoBoolLiteralComparisonsRule.self,
            NoShapeInSymbolNamesRule.self,
            NoKeyValueCodingRule.self,
            NoRuntimeTypeSniffingRule.self,
            NoHardcodedSecretsRule.self,
        ]
    }
}

public enum AntiSlop {
    /// Parse `source` and run each rule type over it.
    ///
    /// Each rule gets a fresh visitor instance; violations from all rules are
    /// returned in walk order per rule. Sort before displaying if you need a
    /// deterministic cross-rule order.
    public static func lint(
        source: String,
        fileName: String,
        rules: [SlopRule.Type]
    ) -> [Violation] {
        let tree = Parser.parse(source: source)
        // Fold binary-operator sequences (`as!`, `as?`, `??`, ...) into typed
        // nodes so rules can see them; keep going on unfoldable input.
        let folded = OperatorTable.standardOperators.foldAll(tree) { _ in }
        let converter = SourceLocationConverter(fileName: fileName, tree: folded)
        return rules.flatMap { ruleType -> [Violation] in
            let rule = ruleType.init(fileName: fileName, sourceText: source, converter: converter)
            rule.walk(folded)
            return rule.violations
        }
    }
}
