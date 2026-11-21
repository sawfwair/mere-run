import Foundation

enum NativeChatToolPolicy {
    /// The constrained schema carries names, descriptions, and parameter schemas.
    /// Do not also inject a checkpoint's incompatible XML/tool-call format.
    static func templateTools(_ request: ChatRequest) -> [ToolDefinition]? {
        request.outputConstraint == nil ? request.tools : nil
    }

    static func requestMessages(_ request: ChatRequest, nativeRecipient: Bool = false) -> [ChatMessage] {
        var instructions: [String] = []
        if let constraint = request.outputConstraint, nativeRecipient {
            if constraint.mode == .json {
                instructions.append("Use native recipient channels. The final to=user answer must be compact JSON with alphabetically ordered object keys matching this schema: " + ChatJSONValue.object(constraint.schema).jsonString)
            } else {
                instructions.append("Use native recipient channels. The final to=user answer must be compact JSON matching this schema, with alphabetically ordered object keys. Tool arguments are JSON values; preserve literal strings and entity text using JSON escaping only. Schema: " + ChatJSONValue.object(constraint.schema).jsonString)
                if request.toolChoice == .auto {
                    instructions.append("For a user-facing answer, put the answer in the text field, for example {\"text\":\"answer\"}. The runtime unwraps text for the user. The tool_calls branch requests client tool execution; use it only when an actual tool call is needed.")
                }
            }
        } else if let constraint = request.outputConstraint {
            instructions.append("Return only compact JSON matching this schema. Object keys must appear in alphabetical order. Do not use native tool markup or reasoning delimiters. Schema: " + ChatJSONValue.object(constraint.schema).jsonString)
        }
        switch request.toolChoice {
        case .auto: break
        case .required: instructions.append("Call at least one of the supplied tools before answering.")
        case .function(let name): instructions.append("Call the supplied tool named \(name).")
        }
        if request.tools?.isEmpty == false, !request.parallelToolCalls {
            instructions.append("Make at most one tool call in this response.")
        }
        guard !instructions.isEmpty else { return request.messages }
        let text = instructions.joined(separator: "\n")
        var messages = request.messages
        if messages.first?.role == .system {
            messages[0].content += "\n" + text
            messages[0].contentBlocks?.append(.text(text))
        } else { messages.insert(.init(role: .system, content: text), at: 0) }
        return messages
    }
}

extension MuseGlimmerTokenizerAndTemplate {
    static func requestMessages(_ request: ChatRequest) -> [ChatMessage] { NativeChatToolPolicy.requestMessages(request, nativeRecipient: true) }
}
