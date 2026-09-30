import SwiftSyntax

/// Direct port of upstream `no-shape-in-symbol-names`.
///
/// Bans the case-insensitive substring "shape" in declaration names. "Shape"
/// describes structure rather than ownership; name symbols for their domain
/// role.
public final class NoShapeInSymbolNamesRule: SlopRule {
    override public class var id: String { "no-shape-in-symbol-names" }
    override public class var summary: String {
        "Rejects \"shape\" in declaration names; name symbols for their domain role."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    private func check(_ name: TokenSyntax) {
        guard name.text.lowercased().contains("shape") else { return }
        report(
            name,
            message:
                "Rename \"\(name.text)\" for its domain role; \"shape\" describes structure rather than ownership."
        )
    }

    override public func visitPost(_ node: ClassDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: ActorDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: StructDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: EnumDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: ProtocolDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: TypeAliasDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: AssociatedTypeDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: FunctionDeclSyntax) { check(node.name) }
    override public func visitPost(_ node: MacroDeclSyntax) { check(node.name) }

    override public func visitPost(_ node: VariableDeclSyntax) {
        for binding in node.bindings {
            if let pattern = binding.pattern.as(IdentifierPatternSyntax.self) {
                check(pattern.identifier)
            }
        }
    }

    override public func visitPost(_ node: EnumCaseDeclSyntax) {
        for element in node.elements {
            check(element.name)
        }
    }
}

/// Port of upstream `no-reflect-get`.
///
/// Rejects key-value coding and selector dispatch in favor of typed property
/// access: `value(forKey:)`, `value(forKeyPath:)`, `setValue(_:forKey:)`,
/// `setValue(_:forKeyPath:)`, and `perform(_:)`.
public final class NoKeyValueCodingRule: SlopRule {
    override public class var id: String { "no-key-value-coding" }
    override public class var summary: String {
        "Rejects key-value coding and selector dispatch in favor of typed property access."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: FunctionCallExprSyntax) {
        guard let member = node.calledExpression.as(MemberAccessExprSyntax.self) else { return }
        let name = member.declName.baseName.text
        let labels = Set(node.arguments.compactMap { $0.label?.text })

        switch name {
        case "value" where labels == ["forKey"] || labels == ["forKeyPath"]:
            break
        case "setValue" where labels.contains("forKey") || labels.contains("forKeyPath"):
            break
        case "setNilValueForKey" where labels.isEmpty:
            // NSObject.setNilValueForKey(_:) takes a single unlabeled argument.
            break
        case "perform" where labels.isEmpty:
            // Only selector dispatch counts; typed APIs like Vision's
            // handler.perform([requests]) share the name but take arrays.
            guard Self.isSelectorArgument(node.arguments.first?.expression) else { return }
        default:
            return
        }

        report(
            member.declName.baseName,
            message:
                "Key-value coding and selector dispatch bypass the type system; a typo or rename compiles and crashes at runtime. Call the typed accessor, or parse untrusted input at the boundary."
        )
    }
    /// `#selector(...)` expansions are the signature of dynamic selector
    /// dispatch; typed array arguments (Vision, XCTest) are not.
    private static func isSelectorArgument(_ expression: ExprSyntax?) -> Bool {
        guard let expression else { return false }
        if let macro = expression.as(MacroExpansionExprSyntax.self) {
            return macro.macroName.text == "selector"
        }
        return false
    }
}

/// Port of upstream `no-runtime-typeof`.
///
/// Rejects runtime type sniffing — `String(describing: type(of: x))` string
/// contracts and `type(of: x) == Dog.self` dynamic type comparisons. Model
/// variants in the type system or decode at the boundary.
///
/// Escape hatch: exact-class checks are sometimes deliberate (performance
/// fast paths, diagnostics). A preceding `// SAFETY:` comment documents that
/// choice inline instead of disabling the rule globally.
public final class NoRuntimeTypeSniffingRule: SlopRule {
    override public class var id: String { "no-runtime-type-sniffing" }
    override public class var summary: String {
        "Rejects `String(describing: type(of:))` sniffing and `type(of:) == T.self` comparisons."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: InfixOperatorExprSyntax) {
        let leftIsTypeOf = Self.isTypeOfCall(node.leftOperand)
        let rightIsTypeOf = Self.isTypeOfCall(node.rightOperand)
        guard leftIsTypeOf || rightIsTypeOf else { return }

        // Both sides being `type(of:)` calls is still inspection narrowing;
        // a single call must be compared against a metatype to count.
        if leftIsTypeOf && rightIsTypeOf {
            if hasPrecedingSafetyComment(for: node.operator) { return }
            report(
                node.operator,
                message:
                    "Comparing dynamic types narrows by inspection instead of contract. Model variants as an enum or protocol and switch on the value, not its runtime type — or add a // SAFETY: comment documenting a deliberate check such as an exact-class fast path."
            )
            return
        }

        let other = leftIsTypeOf ? node.rightOperand : node.leftOperand
        guard Self.isMetatypeExpression(other) else { return }
        if hasPrecedingSafetyComment(for: node.operator) { return }
        report(
            node.operator,
            message:
                "Comparing dynamic types narrows by inspection instead of contract. Model variants as an enum or protocol and switch on the value, not its runtime type."
        )
    }

    public override func visitPost(_ node: FunctionCallExprSyntax) {
        guard let called = node.calledExpression.as(DeclReferenceExprSyntax.self),
            called.baseName.text == "String"
        else { return }
        for argument in node.arguments {
            guard argument.label?.text == "describing" || argument.label?.text == "reflecting",
                let inner = argument.expression.as(FunctionCallExprSyntax.self),
                let innerCalled = inner.calledExpression.as(DeclReferenceExprSyntax.self),
                innerCalled.baseName.text == "type"
            else { continue }
            if hasPrecedingSafetyComment(for: argument.expression) { continue }
            report(
                argument.expression,
                message:
                    "Runtime type sniffing turns dynamic type into a string contract. Model variants as an enum or protocol, or decode the real shape at the boundary."
            )
        }
    }

    private static func isTypeOfCall(_ expression: ExprSyntax) -> Bool {
        guard let call = expression.as(FunctionCallExprSyntax.self),
            let called = call.calledExpression.as(DeclReferenceExprSyntax.self),
            called.baseName.text == "type"
        else { return false }
        return true
    }

    private static func isMetatypeExpression(_ expression: ExprSyntax) -> Bool {
        switch Syntax(expression).as(SyntaxEnum.self) {
        case .memberAccessExpr(let member):
            return member.declName.baseName.text == "self"
        case .metatypeType:
            return true
        default:
            return false
        }
    }
}
