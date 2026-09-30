import SwiftSyntax

/// Swift-specific. `fatalError` and `preconditionFailure` end the process on
/// the first unexpected state; require the author to document why that state
/// cannot occur.
public final class NoFatalErrorRule: SlopRule {
    override public class var id: String { "no-fatal-error" }
    override public class var summary: String {
        "Requires fatalError/preconditionFailure to document their invariant in a preceding // SAFETY: comment."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: FunctionCallExprSyntax) {
        guard let called = node.calledExpression.as(DeclReferenceExprSyntax.self),
            called.baseName.text == "fatalError" || called.baseName.text == "preconditionFailure"
        else { return }
        guard !hasPrecedingSafetyComment(for: node) else { return }
        report(
            called.baseName,
            message:
                "\(called.baseName.text) terminates the process without stating why this path is unreachable. Add a // SAFETY: comment documenting the invariant, or model the failure in the type system."
        )
    }
}

/// Swift-specific. An implicitly unwrapped optional is a force unwrap whose
/// justification lives somewhere else in time. Outlets are exempt.
public final class NoImplicitlyUnwrappedOptionalsRule: SlopRule {
    override public class var id: String { "no-implicitly-unwrapped-optionals" }
    override public class var summary: String {
        "Rejects implicitly unwrapped optionals (`Int!`), which are deferred force unwraps; @IBOutlet declarations are exempt."
    }

    /// Attributes whose members are populated by external machinery before
    /// use, where IUO is the accepted spelling.
    private static let exemptAttributes: Set<String> = ["IBOutlet", "IBInspectable"]

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: VariableDeclSyntax) {
        if isExempt(node.attributes) { return }
        for binding in node.bindings {
            guard let annotation = binding.typeAnnotation,
                implicitUnwrapped(in: TypeSyntax(annotation.type)) != nil
            else { continue }
            report(
                annotation.type,
                message:
                    "Implicitly unwrapped optional is a force unwrap with no stated invariant. Use a plain Optional and unwrap explicitly, or document the population guarantee with // SAFETY:."
            )
        }
    }

    public override func visitPost(_ node: FunctionDeclSyntax) {
        guard let returnClause = node.signature.returnClause,
            let iuo = implicitUnwrapped(in: returnClause.type)
        else { return }
        report(
            iuo.exclamationMark,
            message:
                "\(node.name.text)() returns an implicitly unwrapped optional; callers hold a crash waiting to happen. Return `Optional` and make callers unwrap."
        )
    }

    private func isExempt(_ attributes: AttributeListSyntax) -> Bool {
        attributes.contains { element in
            guard case .attribute(let attribute) = element,
                let name = attribute.attributeName.as(IdentifierTypeSyntax.self)
            else { return false }
            return Self.exemptAttributes.contains(name.name.text)
        }
    }

    private func implicitUnwrapped(in type: TypeSyntax) -> ImplicitlyUnwrappedOptionalTypeSyntax? {
        switch Syntax(type).as(SyntaxEnum.self) {
        case .implicitlyUnwrappedOptionalType(let iuo):
            return iuo
        default:
            return nil
        }
    }
}

/// Port of upstream `no-known-value-widening`'s spirit. Casting to `Any` or
/// `AnyObject` throws away everything the compiler knows about a value; it
/// must be justified like any other evidence-destroying operation.
public final class NoAsAnyCastRule: SlopRule {
    override public class var id: String { "no-as-any-cast" }
    override public class var summary: String {
        "Requires casts to Any/AnyObject to justify discarding type information in a preceding // SAFETY: comment."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: AsExprSyntax) {
        guard SimplifiedType.of(node.type).isAnyLike else { return }
        guard !hasPrecedingSafetyComment(for: node) else { return }
        report(
            node.asKeyword,
            message:
                "Cast to \(node.type.trimmedDescription) erases everything known about the value. Keep the concrete type, introduce one that models it, or add a // SAFETY: comment stating why the widening is necessary."
        )
    }
}

/// Rejects catch clauses that discard the error channel without binding or
/// handling anything.
public final class NoSwallowedErrorsRule: SlopRule {
    override public class var id: String { "no-swallowed-errors" }
    override public class var summary: String {
        "Rejects empty `catch` blocks that discard the error without logging, recovering, or rethrowing."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: CatchClauseSyntax) {
        guard node.body.statements.isEmpty else { return }
        report(
            node.catchKeyword,
            message:
                "Empty catch block swallows the error entirely; failures become invisible. Bind the error, log it, recover deliberately, or rethrow."
        )
    }
}

/// Rejects conditions built entirely from boolean literals — they carry no
/// signal.
///
/// Divergence note: binary comparisons like `flag == true` are NOT flagged.
/// When the operand is an optional Bool, `optBool == true` is meaningful Swift
/// ("non-nil and true"), and a syntax-level linter cannot see operand types —
/// production use showed hundreds of false positives. The ternary form is
/// always redundant regardless of types and remains rejected.
public final class NoBoolLiteralComparisonsRule: SlopRule {
    override public class var id: String { "no-bool-literal-comparisons" }
    override public class var summary: String {
        "Rejects `cond ? true : false`; a ternary between boolean literals is the condition spelled twice."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: TernaryExprSyntax) {
        guard node.thenExpression.is(BooleanLiteralExprSyntax.self),
            node.elseExpression.is(BooleanLiteralExprSyntax.self)
        else { return }
        report(
            node.questionMark,
            message:
                "Ternary between two boolean literals is the condition spelled twice. Use the condition directly, negated with `!` if needed."
        )
    }
}

/// Rejects secret-looking constants assigned string literals.
public final class NoHardcodedSecretsRule: SlopRule {
    override public class var id: String { "no-hardcoded-secrets" }
    override public class var summary: String {
        "Rejects credentials assigned string literals; read them from the environment or a secrets store."
    }

    /// Matched against the lowercased declaration name. Deliberately excludes
    /// the bare word "token", which appears in too many innocent names.
    private static let secretTerms: Set<String> = [
        "apikey", "api_key",
        "secret",
        "password", "passwd", "passphrase",
        "accesstoken", "access_token",
        "refreshtoken", "refresh_token",
        "privatekey", "private_key",
        "clientsecret",
        "credential", "credentials",
    ]

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: VariableDeclSyntax) {
        for binding in node.bindings {
            guard let pattern = binding.pattern.as(IdentifierPatternSyntax.self),
                containsSecretTerm(pattern.identifier.text),
                let initializer = binding.initializer?.value,
                let literal = plaintextStringLiteral(initializer),
                !looksLikeEnvironmentVariableName(literal),
                !(looksLikeIdentifier(literal) && hasKeyLikeName(pattern.identifier.text))
            else { continue }
            report(
                initializer,
                message:
                    "\"\(literal)\" looks like a committed credential. Read it from the environment, a secrets store, or local configuration instead."
            )
        }
    }

    private func containsSecretTerm(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return Self.secretTerms.contains { lowered.contains($0) }
    }

    /// `"MERERUN_API_KEY"`-style literals are environment variable names, not
    /// committed secrets. SCREAMING_CASE values are exempt.
    private func looksLikeEnvironmentVariableName(_ value: String) -> Bool {
        !value.isEmpty
            && value == value.uppercased()
            && value.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    /// Kebab-case slugs (`"mere-run-local-eval"`) and dotted key paths
    /// (`"mererun.app.runtimeAPIKey"`) are exempt only when the declaration
    /// name itself signals a key or label (`evalKeyName`, `defaultsKey`).
    /// Shape alone would hide real secrets like "correct-horse-battery-staple"
    /// assigned to `password`. Letter-only segments keep mixed-alphanumeric
    /// API-key fragments flagged.
    private func looksLikeIdentifier(_ value: String) -> Bool {
        if value.range(of: "^[a-z]+(-[a-z]+)+$", options: .regularExpression) != nil {
            return true
        }
        return value.range(of: "^[A-Za-z][A-Za-z0-9]*(\\.[A-Za-z][A-Za-z0-9]*)+$", options: .regularExpression) != nil
    }

    /// Names that declare "this holds the *name/address* of something" —
    /// per upstream review: KeyName, DefaultsKey, Identifier. Bare `apiKey`
    /// deliberately does not qualify: it names the secret's storage slot, and
    /// its value may be the credential itself.
    private func hasKeyLikeName(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return ["keyname", "defaultskey", "identifier"]
            .contains { lowered.hasSuffix($0) }
    }

    /// Returns the literal's text when `expr` is a plain, non-empty,
    /// non-interpolated string literal.
    private func plaintextStringLiteral(_ expr: ExprSyntax) -> String? {
        guard let literal = expr.as(StringLiteralExprSyntax.self),
            literal.openingPounds == nil && literal.closingPounds == nil
        else { return nil }

        var contents = ""
        for segment in literal.segments {
            guard case .stringSegment(let stringSegment) = segment else {
                return nil // interpolation or raw segment: not a static secret
            }
            contents += stringSegment.content.text
        }
        return contents.isEmpty ? nil : contents
    }
}
