import Foundation
import Jinja

/// The checkpoint uses Python JSON keyword arguments beyond Jinja's built-in
/// filter. Supply their semantics in the rendering environment, keeping the
/// released template intact.
struct Inkling2ChatTemplate {
    let template: Template

    init(source: String) throws { template = try Template(source) }

    func render(messages: [ChatMessage], tools: [ToolDefinition]? = nil, effort: Double) throws -> String {
        guard effort.isFinite, (0...0.99).contains(effort) else {
            throw ChatRequestIssue("reasoning_effort", "must be between 0 and 0.99")
        }
        let environment = Environment()
        environment["tojson"] = .function { args, kwargs, _ in
            guard args.count == 1, Set(kwargs.keys).isSubset(of: ["sort_keys", "separators", "ensure_ascii", "indent"]) else {
                throw ChatRequestIssue("chat_template", "unsupported JSON filter arguments")
            }
            var options = Jinja.JSON.DumpsOptions(ensureASCII: false)
            if let sort = kwargs["sort_keys"] {
                guard case .boolean(let value) = sort else { throw ChatRequestIssue("chat_template", "sort_keys must be boolean") }
                options.sortKeys = value
            }
            if let separators = kwargs["separators"] {
                guard case .array(let values) = separators, values.count == 2,
                      case .string(let item) = values[0], case .string(let key) = values[1] else {
                    throw ChatRequestIssue("chat_template", "separators must contain two strings")
                }
                options.separators = (item, key)
            }
            if let ascii = kwargs["ensure_ascii"] {
                guard case .boolean(let value) = ascii else { throw ChatRequestIssue("chat_template", "ensure_ascii must be boolean") }
                options.ensureASCII = value
            }
            if let indent = kwargs["indent"], indent != .null {
                guard case .int(let value) = indent else { throw ChatRequestIssue("chat_template", "indent must be an integer") }
                options.indent = value
            }
            return .string(try Jinja.JSON.dumps(args[0], options: options))
        }
        let context: [String: Value] = try [
            "messages": Value(any: MuseGlimmerTokenizerAndTemplate.renderMessages(messages)),
            "tools": Value(any: tools?.map { $0.toToolSpec() } ?? []),
            "reasoning_effort": .double(effort), "add_generation_prompt": true,
        ]
        return try template.render(context, environment: environment)
    }
}
