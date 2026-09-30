import SwiftSyntax

/// Port of upstream `no-object-parameters`.
///
/// Rejects `Any`/`AnyObject` function parameters. `any Error` (optionally
/// optional) is the accepted error-channel convention and is exempt.
public final class NoAnyParametersRule: SlopRule {
    override public class var id: String { "no-any-parameters" }
    override public class var summary: String {
        "Rejects `Any` and `AnyObject` inputs on functions; model the real contract instead."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: FunctionParameterSyntax) {
        let simplified = SimplifiedType.of(node.type)
        guard simplified.isAnyLike else { return }
        report(
            node.type,
            message:
                "Parameter is typed \(node.type.trimmedDescription); that hides the real contract from every caller. Name the concrete input type, or introduce one."
        )
    }
}

/// Port of upstream `no-unknown-returns`.
///
/// Rejects functions whose declared return type is `Any`/`AnyObject`.
public final class NoAnyReturnsRule: SlopRule {
    override public class var id: String { "no-any-returns" }
    override public class var summary: String {
        "Rejects function contracts that return `Any` or `AnyObject`."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: FunctionDeclSyntax) {
        guard let returnClause = node.signature.returnClause else { return }
        let simplified = SimplifiedType.of(returnClause.type)
        guard simplified.isAnyLike else { return }
        report(
            returnClause.arrow,
            message:
                "\(node.name.text)() declares it returns \(returnClause.type.trimmedDescription); callers learn nothing they can rely on. Return a concrete or opaque type."
        )
    }
}

/// Port of upstream `no-unknown-type-aliases`.
///
/// Rejects aliases that merely conceal `Any`: `typealias Payload = Any`.
public final class NoAnyTypeAliasesRule: SlopRule {
    override public class var id: String { "no-any-typealiases" }
    override public class var summary: String {
        "Rejects type aliases whose underlying type is `Any` or `AnyObject`."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: TypeAliasDeclSyntax) {
        guard SimplifiedType.of(node.initializer.value).isAnyLike else { return }
        report(
            node.name,
            message:
                "`typealias \(node.name.text)` conceals `Any` behind a friendlier spelling without adding evidence. Define a concrete model instead."
        )
    }
}

/// Port of upstream `no-unsafe-dictionary-type`.
///
/// Rejects dictionary value contracts based on `Any`/`AnyObject`, such as
/// `[String: Any]`.
public final class NoAnyDictionaryValueRule: SlopRule {
    override public class var id: String { "no-any-dictionary-value" }
    override public class var summary: String {
        "Rejects dictionaries whose values are `Any`/`AnyObject`, such as `[String: Any]`."
    }

    public required init(
        fileName: String,
        sourceText: String,
        converter: SourceLocationConverter
    ) {
        super.init(fileName: fileName, sourceText: sourceText, converter: converter)
    }

    public override func visitPost(_ node: DictionaryTypeSyntax) {
        guard SimplifiedType.of(node.value).isAnyLike else { return }
        report(
            node.colon,
            message:
                "Dictionary values are typed \(node.value.trimmedDescription); every reader must cast to recover the real shape. Model entries as an enum or struct."
        )
    }
}
