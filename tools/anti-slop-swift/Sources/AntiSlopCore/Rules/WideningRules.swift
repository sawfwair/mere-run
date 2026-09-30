import SwiftSyntax

/// Port of upstream `no-known-value-widening`'s evidence slice: rejects
/// syntactically established values flowing into explicitly `Any`-shaped
/// targets that discard what the compiler already knows:
///
/// - `let payload: Any = ["a": 1]`
/// - `let items: AnyObject = [1, 2, 3]`
///
/// Divergence from upstream: upstream also flags known-key object literals
/// annotated as open `Record`s. Swift's equivalent — `let headers:
/// [String: String] = ["Content-Type": "..."]` — is idiomatic for small typed
/// constants and is deliberately not flagged; production use confirmed the
/// slice produced only noise.
public final class NoKnownValueWideningRule: SlopRule {
    override public class var id: String { "no-known-value-widening" }
    override public class var summary: String {
        "Rejects explicit broad annotations that discard known evidence from literals, including known dictionary keys."
    }

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
                let annotation = binding.typeAnnotation,
                let initializer = binding.initializer?.value
            else { continue }

            if SimplifiedType.of(TypeSyntax(annotation.type)).isAnyLike,
                Self.hasKnownEvidence(initializer)
            {
                report(
                    annotation.type,
                    message:
                        "The explicit \(annotation.type.trimmedDescription) annotation on \(pattern.identifier.text) discards known type evidence. Keep inference, or use a named contract."
                )
            }
        }
    }

    /// Mirrors upstream's `isKnownEvidenceExpression`: expressions whose shape
    /// the compiler already knows without any annotation.
    static func hasKnownEvidence(_ expression: ExprSyntax) -> Bool {
        switch Syntax(expression).as(SyntaxEnum.self) {
        case .stringLiteralExpr, .integerLiteralExpr, .floatLiteralExpr,
            .booleanLiteralExpr, .regexLiteralExpr,
            .arrayExpr, .dictionaryExpr, .closureExpr:
            return true
        default:
            return false
        }
    }
}

/// Port of upstream `no-widen-then-assert`.
///
/// Rejects local flows that widen a known value into an `Any`-shaped binding
/// and later cast the binding back to a narrower type:
///
/// ```swift
/// let loaded = loadUser()
/// let stored: Any = loaded
/// let user = stored as! User
/// ```
///
/// Divergence from upstream: upstream is function-boundary scoped via the
/// scope manager; this port is source-order scoped across the file, which
/// also catches class-level `var stored: Any` properties cast inside methods.
public final class NoWidenThenAssertRule: SlopRule {
    override public class var id: String { "no-widen-then-assert" }
    override public class var summary: String {
        "Rejects bindings that widen a known value to Any and later cast it back to a narrower type."
    }

    private var widenedBindings: [String: Int] = [:]
    /// Every identifier declaration (variables and function parameters) with
    /// its position, used to detect shadowing between a widened binding and a
    /// later cast.
    private var identifierDeclarations: [(name: String, offset: Int)] = []

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: VariableDeclSyntax) {
        for binding in node.bindings {
            guard let pattern = binding.pattern.as(IdentifierPatternSyntax.self) else { continue }

            let annotationIsBroad =
                binding.typeAnnotation.map {
                    SimplifiedType.of(TypeSyntax($0.type)).isAnyLike
                } ?? false

            let initializerIsBroadCast: Bool
            if let initializer = binding.initializer?.value,
                let cast = initializer.as(AsExprSyntax.self)
            {
                initializerIsBroadCast = SimplifiedType.of(cast.type).isAnyLike
            } else {
                initializerIsBroadCast = false
            }

            guard annotationIsBroad || initializerIsBroadCast else { continue }
            widenedBindings[pattern.identifier.text] =
                node.positionAfterSkippingLeadingTrivia.utf8Offset
        }
        recordIdentifierDeclarations(in: node)
    }

    /// Parameters must be recorded pre-order: post-order fires after the
    /// body's casts have already been visited.
    public override func visit(_ node: FunctionDeclSyntax) -> SyntaxVisitorContinueKind {
        let offset = node.positionAfterSkippingLeadingTrivia.utf8Offset
        for parameter in node.signature.parameterClause.parameters {
            // `first second: Type` binds `second`; bare `first: Type` binds `first`.
            let name = parameter.secondName?.text ?? parameter.firstName.text
            identifierDeclarations.append((name, offset))
        }
        return .visitChildren
    }

    private func recordIdentifierDeclarations(in node: VariableDeclSyntax) {
        let offset = node.positionAfterSkippingLeadingTrivia.utf8Offset
        for binding in node.bindings {
            guard let pattern = binding.pattern.as(IdentifierPatternSyntax.self) else { continue }
            identifierDeclarations.append((pattern.identifier.text, offset))
        }
    }

    public override func visitPost(_ node: AsExprSyntax) {
        // The cast must narrow; casting to Any is the widening side.
        guard !SimplifiedType.of(node.type).isAnyLike else { return }
        guard let operand = unwrappedOperand(node.expression),
            let declarationOffset = widenedBindings[operand]
        else { return }
        guard node.positionAfterSkippingLeadingTrivia.utf8Offset > declarationOffset else { return }

        // A same-named declaration between the widened binding and this cast
        // means the operand refers to something else; do not fire.
        let castOffset = node.positionAfterSkippingLeadingTrivia.utf8Offset
        let isShadowed = identifierDeclarations.contains { declaration in
            declaration.name == operand
                && declaration.offset > declarationOffset
                && declaration.offset < castOffset
        }
        guard !isShadowed else { return }

        report(
            node.asKeyword,
            message:
                "Binding \"\(operand)\" discards type evidence and later recreates it with a cast. Keep the precise type from initialization through use; parse boundary input once."
        )
    }

    private func unwrappedOperand(_ expression: ExprSyntax) -> String? {
        switch Syntax(expression).as(SyntaxEnum.self) {
        case .declReferenceExpr(let reference):
            return reference.baseName.text
        case .tupleExpr(let tuple):
            // Parenthesized expressions parse as one-element tuples.
            guard tuple.elements.count == 1, let only = tuple.elements.first else {
                return nil
            }
            return unwrappedOperand(only.expression)
        default:
            return nil
        }
    }
}
