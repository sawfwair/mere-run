import Foundation

public struct GLM47ChatTemplate {
    public static func render(
        messages: [ChatMessage],
        tools: [[String: Any]]? = nil,
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = false
    ) -> String {
        var prompt = "[gMASK]<sop>"

        if let tools, !tools.isEmpty {
            prompt.append("\n<|system|>\n# Tools\n\n")
            prompt.append("You may call one or more functions to assist with the user query.\n\n")
            prompt.append("You are provided with function signatures within <tools></tools> XML tags:\n")
            prompt.append("<tools>\n")
            for tool in tools {
                if let json = jsonString(tool) {
                    prompt.append(json)
                }
                prompt.append("\n")
            }
            prompt.append("</tools>\n\n")
            prompt.append("For each function call, output the function name and arguments within the following XML format:\n")
            prompt.append("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>")
        }

        for message in messages {
            switch message.role {
            case .user:
                prompt.append("\n<|user|>")
                prompt.append(message.content)
            case .assistant:
                prompt.append("\n<|assistant|>")
                prompt.append(enableThinking ? "<think>" : "</think>")
                if !message.content.isEmpty {
                    prompt.append(message.content)
                }
            case .system:
                prompt.append("\n<|system|>")
                prompt.append(message.content)
            case .tool:
                prompt.append("\n<|observation|><tool_response>")
                prompt.append(message.content)
                prompt.append("</tool_response>")
            }
        }

        if addGenerationPrompt {
            prompt.append("\n<|assistant|>")
            prompt.append(enableThinking ? "<think>" : "</think>")
        }

        return prompt
    }

    private static func jsonString(_ value: [String: Any]) -> String? {
        guard JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: []) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }
}
