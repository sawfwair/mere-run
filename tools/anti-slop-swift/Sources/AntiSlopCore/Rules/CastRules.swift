import SwiftSyntax

/// Port of upstream `no-chained-type-assertions`.
///
/// Rejects nested type casts (`x as! Foo as! Bar`, `x as? Foo as? Bar`) that
/// fabricate evidence: the first cast invents a type, and the second invents
/// another on top of it.
public final class NoChainedTypeCastsRule: SlopRule {
    override public class var id: String { "no-chained-type-casts" }
    override public class var summary: String {
        "Rejects nested type casts that fabricate evidence."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: AsExprSyntax) {
        guard node.expression.is(AsExprSyntax.self) else { return }
        let mark = node.questionOrExclamationMark?.text ?? " "
        report(
            node.asKeyword,
            message:
                "This cast is chained onto another cast (… as\(mark) \(node.type.trimmedDescription)); each hop fabricates evidence. Validate the real shape once at the boundary and convert explicitly."
        )
    }
}

/// Port of upstream `require-safety-comment-for-type-assertion`.
///
/// Every forced cast (`as!`) must document its checked invariant in a
/// preceding `// SAFETY:` comment.
public final class RequireSafetyCommentForForcedCastRule: SlopRule {
    override public class var id: String { "require-safety-comment-for-forced-cast" }
    override public class var summary: String {
        "Requires every `as!` cast to document its checked invariant in a preceding // SAFETY: comment."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: AsExprSyntax) {
        guard node.questionOrExclamationMark?.tokenKind == .exclamationMark else { return }
        guard !hasPrecedingSafetyComment(for: node) else { return }
        report(
            node.asKeyword,
            message:
                "Forced cast to \(node.type.trimmedDescription) has no documented invariant. Add a // SAFETY: comment stating what you checked, or handle the failure case."
        )
    }
}

/// Swift-specific. `try!` discards error evidence entirely; it must state its
/// invariant in a preceding `// SAFETY:` comment.
public final class NoForceTryRule: SlopRule {
    override public class var id: String { "no-force-try" }
    override public class var summary: String {
        "Requires every `try!` to document its checked invariant in a preceding // SAFETY: comment."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: TryExprSyntax) {
        guard node.questionOrExclamationMark?.tokenKind == .exclamationMark else { return }
        guard !hasPrecedingSafetyComment(for: node) else { return }
        report(
            node.tryKeyword,
            message:
                "`try!` discards the error channel without stating why it cannot fire. Add a // SAFETY: comment documenting the invariant, or propagate/handle the error."
        )
    }
}

/// Swift-specific. Force unwrapping optionals crashes on the first unexpected
/// value; require the author to document why that cannot happen.
public final class NoForceUnwrapRule: SlopRule {
    override public class var id: String { "no-force-unwrap" }
    override public class var summary: String {
        "Requires every force unwrap (`x!`) to document its checked invariant in a preceding // SAFETY: comment."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: ForceUnwrapExprSyntax) {
        let operatorToken = node.exclamationMark
        guard operatorToken.text == "!" else { return }
        guard !hasPrecedingSafetyComment(for: operatorToken) else { return }
        report(
            operatorToken,
            message:
                "Force unwrap crashes on the first unexpected nil. Add a // SAFETY: comment stating why the optional is known non-nil, or unwrap with `guard let`."
        )
    }
}
