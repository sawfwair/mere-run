import Foundation
import MereRunCore

enum StructuredImagePromptAdapterError: LocalizedError {
    case invalidMaxTokens(Int)
    case invalidModelRoot(String)
    case invalidCaptionJSON(String)

    var errorDescription: String? {
        switch self {
        case .invalidMaxTokens(let value):
            return "--structured-prompt-max-tokens must be >= 1. Received: \(value)."
        case .invalidModelRoot(let path):
            return "Structured prompt model root not found: \(path)"
        case .invalidCaptionJSON(let detail):
            return "Structured prompt adapter did not produce valid caption JSON. \(detail)"
        }
    }
}

enum StructuredImagePromptAdapter {
    static let defaultModelID = Gemma4Resources.twelveB4BitModelId
    static let defaultMaxTokens = 2_048
    static let recommendedImagePromptTokens = 2_048

    static func backendDescription(for modelID: String) -> String {
        let normalizedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedModelID == "text-agent-deepseek-v4-flash" {
            return "external DeepSeek V4 Flash bridge"
        }
        if ManagedModelCatalog.spec(for: normalizedModelID)?.validationKind == .codegenGGUF {
            return "llama.cpp/GGUF"
        }
        return NativeMLXRuntime.backendDescription
    }

    static func expand(
        prompt: String,
        modelID: String,
        modelRoot: String?,
        maxTokens: Int,
        progressHandler: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        guard maxTokens > 0 else {
            throw StructuredImagePromptAdapterError.invalidMaxTokens(maxTokens)
        }

        let session = try makeChatSession(modelID: modelID, modelRoot: modelRoot)
        do {
            let expanded = try await expand(
                prompt: prompt,
                maxTokens: maxTokens,
                session: session,
                progressHandler: progressHandler
            )
            await session.unload()
            return expanded
        } catch {
            await session.unload()
            throw error
        }
    }

    @discardableResult
    static func validateCaptionJSON(_ rawJSON: String) throws -> StructuredImageCaption {
        guard let data = rawJSON.data(using: .utf8), !data.isEmpty else {
            throw StructuredImagePromptAdapterError.invalidCaptionJSON("Output was empty.")
        }
        do {
            return try JSONDecoder().decode(StructuredImageCaption.self, from: data)
        } catch {
            throw StructuredImagePromptAdapterError.invalidCaptionJSON(error.localizedDescription)
        }
    }

    static func normalizedCaptionJSON(from rawJSON: String, fallbackPrompt: String) throws -> String {
        guard !containsGeneratedSpecialTokenSpill(rawJSON) else {
            throw StructuredImagePromptAdapterError.invalidCaptionJSON(
                "Output contained generated multimodal special tokens."
            )
        }
        let candidate = cleanedJSONCandidate(from: rawJSON)
        do {
            return try normalizedCaptionJSONCandidate(candidate, fallbackPrompt: fallbackPrompt)
        } catch {
            if let repaired = repairedJSONPrefixCandidate(from: candidate) {
                do {
                    return try normalizedCaptionJSONCandidate(repaired, fallbackPrompt: fallbackPrompt)
                } catch {
                    if let salvaged = salvagedCaptionJSON(from: candidate, fallbackPrompt: fallbackPrompt) {
                        return salvaged
                    }
                    throw StructuredImagePromptAdapterError.invalidCaptionJSON(
                        validationFailureDetail(error: error, candidate: repaired)
                    )
                }
            }
            if let salvaged = salvagedCaptionJSON(from: candidate, fallbackPrompt: fallbackPrompt) {
                return salvaged
            }
            throw StructuredImagePromptAdapterError.invalidCaptionJSON(
                validationFailureDetail(error: error, candidate: candidate)
            )
        }
    }

    private static func normalizedCaptionJSONCandidate(_ candidate: String, fallbackPrompt: String) throws -> String {
        do {
            let caption = try validateCaptionJSON(candidate)
            return try encodeCaptionJSON(ensuringTextRender(caption, prompt: fallbackPrompt))
        } catch {
            guard let data = candidate.data(using: .utf8), !data.isEmpty else {
                throw StructuredImagePromptAdapterError.invalidCaptionJSON("Output was empty.")
            }
            let flexibleCaption = try JSONDecoder().decode(FlexibleStructuredImageCaption.self, from: data)
            let normalizedCaption = ensuringTextRender(
                flexibleCaption.normalized(fallbackPrompt: fallbackPrompt),
                prompt: fallbackPrompt
            )
            let normalizedJSON = try encodeCaptionJSON(normalizedCaption)
            _ = try validateCaptionJSON(normalizedJSON)
            return normalizedJSON
        }
    }

    /// Deterministic safety net: if the adapter model left "text render" empty but the
    /// original prompt asks for visible text in quotes, copy those strings in verbatim.
    /// Without this, requested glyphs silently degrade into garbled texture.
    static func ensuringTextRender(_ caption: StructuredImageCaption, prompt: String) -> StructuredImageCaption {
        guard caption.textRender.isEmpty else { return caption }
        let quoted = quotedStrings(in: prompt)
        guard !quoted.isEmpty else { return caption }
        let entries = quoted.prefix(5).map { text in
            StructuredImageTextRender(
                text: text,
                location: "as described in the prompt",
                size: "medium",
                color: "unspecified",
                font: "unspecified",
                appearanceDetails: nil
            )
        }
        return StructuredImageCaption(
            shortDescription: caption.shortDescription,
            objects: caption.objects,
            backgroundSetting: caption.backgroundSetting,
            lighting: caption.lighting,
            aesthetics: caption.aesthetics,
            photographicCharacteristics: caption.photographicCharacteristics,
            styleMedium: caption.styleMedium,
            textRender: Array(entries),
            context: caption.context,
            artisticStyle: caption.artisticStyle
        )
    }

    static func deterministicCaptionJSON(for prompt: String) throws -> String {
        let description = prompt.nonEmpty ?? "Requested image"
        let caption = StructuredImageCaption(
            shortDescription: description,
            objects: [StructuredImageObject.defaultObject(description: description)],
            backgroundSetting: "Background consistent with the requested image.",
            lighting: .defaultLighting(description: nil),
            aesthetics: .defaultAesthetics(description: nil),
            photographicCharacteristics: .defaultPhotographicCharacteristics(description: nil),
            styleMedium: "photograph",
            textRender: [],
            context: "Text-to-image generation prompt.",
            artisticStyle: "natural realism"
        )
        return try encodeCaptionJSON(ensuringTextRender(caption, prompt: prompt))
    }

    static func containsGeneratedSpecialTokenSpill(_ response: String) -> Bool {
        let patterns = [
            #"<\|?(?:image|audio|video)[^>]{0,64}>"#,
            #"<(?:start|end)_of_(?:image|audio|video)>"#,
            #"<(?:boi|eoi|boa|eoa|bov|eov)\|?>"#,
        ]
        return patterns.contains { pattern in
            response.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
        }
    }

    /// Extracts double-, curly-, and single-quoted strings. Single quotes must sit on
    /// word boundaries so apostrophes ("the sign's letters") do not produce matches.
    static func quotedStrings(in prompt: String) -> [String] {
        let patterns = [
            #""([^"\n]{1,120})""#,
            #"“([^”\n]{1,120})”"#,
            #"(?<=^|[\s(])'([^'\n]{1,120})'(?=$|[\s,.!?;:)])"#,
        ]
        var seen = Set<String>()
        var results: [String] = []
        let range = NSRange(prompt.startIndex..., in: prompt)
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            for match in regex.matches(in: prompt, range: range) {
                guard match.numberOfRanges > 1,
                      let captureRange = Range(match.range(at: 1), in: prompt) else { continue }
                let text = String(prompt[captureRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty, seen.insert(text.lowercased()).inserted else { continue }
                results.append(text)
            }
        }
        return results
    }

    static func cleanedJSONCandidate(from response: String) -> String {
        var cleaned = response.replacingOccurrences(
            of: #"(?is)<\|channel>final\s*"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"(?is)<\|channel>thought\b.*?<channel\|>\s*"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"(?is)<\|channel>thought\b.*\z"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: #"(?is)<\|channel>[a-z_]+\s*|<channel\|>"#,
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*?</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?is)<think>.*\\z",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "(?i)</think>",
            with: "",
            options: .regularExpression
        )
        cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)

        if cleaned.hasPrefix("```") {
            cleaned = cleaned.replacingOccurrences(
                of: #"(?is)^```(?:json)?\s*|\s*```$"#,
                with: "",
                options: .regularExpression
            )
            cleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let firstBrace = cleaned.firstIndex(of: "{"),
           let lastBrace = cleaned.lastIndex(of: "}"),
           firstBrace <= lastBrace {
            let balancedCandidate = String(cleaned[firstBrace...lastBrace])
            if isParseableJSON(balancedCandidate) {
                return balancedCandidate
            }
            return String(cleaned[firstBrace...])
        }
        if let firstBrace = cleaned.firstIndex(of: "{") {
            return String(cleaned[firstBrace...])
        }
        return cleaned
    }

    static func repairedJSONPrefixCandidate(from candidate: String) -> String? {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              (trimmed.first == "{" || trimmed.first == "["),
              !isParseableJSON(trimmed) else {
            return nil
        }

        var output = ""
        output.reserveCapacity(trimmed.count + 8)
        var stack: [Character] = []
        var inString = false
        var escaping = false
        var completed = false

        func appendEscapedControl(_ scalar: UnicodeScalar) {
            switch scalar.value {
            case 0x08:
                output += "\\b"
            case 0x09:
                output += "\\t"
            case 0x0A:
                output += "\\n"
            case 0x0C:
                output += "\\f"
            case 0x0D:
                output += "\\r"
            case 0x00...0x1F:
                output += String(format: "\\u%04x", scalar.value)
            default:
                output.append(Character(scalar))
            }
        }

        scan: for scalar in trimmed.unicodeScalars {
            if completed {
                if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                    output.append(Character(scalar))
                    continue
                }
                break
            }
            if inString {
                if escaping {
                    if #"\"/bfnrtu"#.unicodeScalars.contains(scalar) {
                        output.append("\\")
                    } else {
                        output += "\\\\"
                    }
                    appendEscapedControl(scalar)
                    escaping = false
                    continue
                }
                if scalar == "\\" {
                    escaping = true
                    continue
                }
                if scalar == "\"" {
                    inString = false
                    output.append("\"")
                    continue
                }
                appendEscapedControl(scalar)
                continue
            }

            switch scalar {
            case "\"":
                inString = true
                output.append("\"")
            case "{":
                stack.append("}")
                output.append("{")
            case "[":
                stack.append("]")
                output.append("[")
            case "}", "]":
                guard stack.last == Character(scalar) else { break scan }
                output = output.removingTrailingCommaAndWhitespace()
                _ = stack.popLast()
                output.append(Character(scalar))
                completed = stack.isEmpty
            default:
                output.append(Character(scalar))
            }
        }

        if escaping {
            output += "\\\\"
        }
        if inString {
            output.append("\"")
        }
        while let closing = stack.popLast() {
            output = output.removingTrailingCommaAndWhitespace()
            output.append(closing)
        }
        output = output.removingDanglingJSONCommas()

        guard output != trimmed,
              isParseableJSON(output) else {
            return nil
        }
        return output
    }

    private static func salvagedCaptionJSON(from candidate: String, fallbackPrompt: String) -> String? {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.first == "{",
              !isParseableJSON(trimmed),
              hasStructuredCaptionSignal(trimmed) else {
            return nil
        }

        let defaults = (
            lighting: StructuredImageLighting.defaultLighting(description: nil),
            aesthetics: StructuredImageAesthetics.defaultAesthetics(description: nil),
            photo: StructuredImagePhotographicCharacteristics.defaultPhotographicCharacteristics(description: nil)
        )
        let objectDescriptions = salvagedObjectDescriptions(from: trimmed, fallbackPrompt: fallbackPrompt)
        let caption = StructuredImageCaption(
            shortDescription: firstJSONStringField(
                in: trimmed,
                keys: ["short description", "short_description"]
            ) ?? fallbackPrompt,
            objects: Array(objectDescriptions.prefix(5)).map(StructuredImageObject.defaultObject(description:)),
            backgroundSetting: firstJSONStringField(
                in: trimmed,
                keys: ["background setting", "background_setting"]
            ) ?? "Minimal background consistent with the requested image.",
            lighting: StructuredImageLighting(
                conditions: firstJSONStringField(in: trimmed, keys: ["conditions", "lighting"])
                    ?? defaults.lighting.conditions,
                direction: firstJSONStringField(in: trimmed, keys: ["direction"]) ?? defaults.lighting.direction,
                shadows: firstJSONStringField(in: trimmed, keys: ["shadows"])
            ),
            aesthetics: StructuredImageAesthetics(
                composition: firstJSONStringField(in: trimmed, keys: ["composition"])
                    ?? defaults.aesthetics.composition,
                colorScheme: firstJSONStringField(in: trimmed, keys: ["color scheme", "color_scheme"])
                    ?? defaults.aesthetics.colorScheme,
                moodAtmosphere: firstJSONStringField(in: trimmed, keys: ["mood atmosphere", "mood_atmosphere"])
                    ?? defaults.aesthetics.moodAtmosphere
            ),
            photographicCharacteristics: StructuredImagePhotographicCharacteristics(
                depthOfField: firstJSONStringField(in: trimmed, keys: ["depth of field", "depth_of_field"])
                    ?? defaults.photo.depthOfField,
                focus: firstJSONStringField(in: trimmed, keys: ["focus"]) ?? defaults.photo.focus,
                cameraAngle: firstJSONStringField(in: trimmed, keys: ["camera angle", "camera_angle"])
                    ?? defaults.photo.cameraAngle,
                lensFocalLength: firstJSONStringField(in: trimmed, keys: ["lens focal length", "lens_focal_length"])
                    ?? defaults.photo.lensFocalLength
            ),
            styleMedium: firstJSONStringField(in: trimmed, keys: ["style medium", "style_medium"]) ?? "photograph",
            textRender: salvagedTextRender(from: trimmed, fallbackPrompt: fallbackPrompt),
            context: firstJSONStringField(in: trimmed, keys: ["context"]) ?? "Text-to-image generation prompt.",
            artisticStyle: firstJSONStringField(in: trimmed, keys: ["artistic style", "artistic_style"])
                ?? "natural realism"
        )

        guard let encoded = try? encodeCaptionJSON(caption),
              (try? validateCaptionJSON(encoded)) != nil else {
            return nil
        }
        return encoded
    }

    private static func hasStructuredCaptionSignal(_ candidate: String) -> Bool {
        [
            #""objects"\s*:"#,
            #""text[_ ]render"\s*:"#,
            #""short[_ ]description"\s*:"#,
            #""background[_ ]setting"\s*:"#,
        ].contains { pattern in
            candidate.range(of: pattern, options: .regularExpression) != nil
        }
    }

    private static func salvagedObjectDescriptions(from candidate: String, fallbackPrompt: String) -> [String] {
        var descriptions = allJSONStringFields(in: candidate, key: "description")
        if descriptions.isEmpty,
           let firstObjectString = firstJSONStringArrayElement(in: candidate, key: "objects") {
            descriptions.append(firstObjectString)
        }
        if descriptions.isEmpty {
            descriptions.append(fallbackPrompt)
        }
        return uniqueNonEmptyStrings(descriptions)
    }

    private static func salvagedTextRender(from candidate: String, fallbackPrompt: String) -> [StructuredImageTextRender] {
        let texts = uniqueNonEmptyStrings(
            allJSONStringFields(in: candidate, key: "text") + quotedStrings(in: fallbackPrompt)
        ).prefix(5)
        return texts.map { text in
            StructuredImageTextRender(
                text: text,
                location: "as described in the prompt",
                size: "medium",
                color: "unspecified",
                font: "unspecified",
                appearanceDetails: nil
            )
        }
    }

    private static func firstJSONStringField(in candidate: String, keys: [String]) -> String? {
        keys.lazy.compactMap { key in
            allJSONStringFields(in: candidate, key: key).first
        }.first
    }

    private static func allJSONStringFields(in candidate: String, key: String) -> [String] {
        let escapedKey = NSRegularExpression.escapedPattern(for: key)
        let pattern = #""\#(escapedKey)"\s*:\s*"((?:\\.|[^"\\])*)""#
        return regexCaptures(in: candidate, pattern: pattern)
    }

    private static func firstJSONStringArrayElement(in candidate: String, key: String) -> String? {
        let escapedKey = NSRegularExpression.escapedPattern(for: key)
        let pattern = #""\#(escapedKey)"\s*:\s*\[\s*"((?:\\.|[^"\\])*)""#
        return regexCaptures(in: candidate, pattern: pattern).first
    }

    private static func regexCaptures(in candidate: String, pattern: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(candidate.startIndex..., in: candidate)
        return regex.matches(in: candidate, range: range).compactMap { match in
            guard match.numberOfRanges > 1,
                  let captureRange = Range(match.range(at: 1), in: candidate) else {
                return nil
            }
            return decodedJSONStringFragment(String(candidate[captureRange]))?.nonEmpty
        }
    }

    private static func decodedJSONStringFragment(_ fragment: String) -> String? {
        let literal = "\"\(fragment)\""
        if let data = literal.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(String.self, from: data) {
            return decoded
        }
        return fragment.replacingOccurrences(of: #"\\(.)"#, with: "$1", options: .regularExpression)
    }

    private static func uniqueNonEmptyStrings(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed.count <= 200 else { continue }
            guard seen.insert(trimmed.lowercased()).inserted else { continue }
            result.append(trimmed)
        }
        return result
    }

    private static func expand(
        prompt: String,
        maxTokens: Int,
        session: ChatSession,
        progressHandler: (@Sendable (String) -> Void)?
    ) async throws -> String {
        let initialMessages = [
            ChatMessage(role: .system, content: structuredCaptionSystemPrompt),
            ChatMessage(
                role: .user,
                content: """
                Convert this text-to-image prompt into the required structured JSON caption.

                Prompt:
                \(prompt)
                """
            ),
        ]
        var messages = initialMessages
        var lastDetail = "No response was generated."

        for attempt in 1...3 {
            progressHandler?("running structured prompt adapter attempt \(attempt)/3")
            let request = ChatRequest(
                messages: messages,
                maxTokens: maxTokens,
                temperature: 0.2,
                topP: 0.9,
                showThinking: false,
                requiresJSON: true
            )
            let response = try await session.chat(request, nil)
            if let debugRoot = ProcessInfo.processInfo.environment["MERERUN_STRUCTURED_PROMPT_DEBUG"] {
                let url = URL(fileURLWithPath: debugRoot).appendingPathComponent("adapter_attempt_\(attempt).txt")
                try? response.response.write(to: url, atomically: true, encoding: .utf8)
            }
            do {
                return try normalizedCaptionJSON(from: response.response, fallbackPrompt: prompt)
            } catch {
                let candidate = cleanedJSONCandidate(from: response.response)
                lastDetail = validationFailureDetail(error: error, candidate: candidate)
                if isParseableJSON(candidate) {
                    // Near miss: valid JSON, wrong shape. Keep the exchange and correct it.
                    messages.append(ChatMessage(role: .assistant, content: response.response))
                    messages.append(ChatMessage(
                        role: .user,
                        content: """
                        The previous response was invalid for this task:
                        \(lastDetail)

                        Return only one valid JSON object matching the exact template from the system message. Do not use "name" or "attributes" as object keys. "text render" must be an array; use [] when no text should appear. Use null only for optional nested values.
                        """
                    ))
                } else {
                    // Gross failure (token salad / truncation / empty): feeding it back as
                    // context would poison every remaining attempt, and a corrupted runtime
                    // state would survive into them. Reload the model and retry fresh.
                    progressHandler?("adapter output was not JSON; reloading model for a fresh attempt")
                    await session.unload()
                    messages = initialMessages
                }
            }
        }

        throw StructuredImagePromptAdapterError.invalidCaptionJSON(lastDetail)
    }

    static func isParseableJSON(_ candidate: String) -> Bool {
        guard let data = candidate.data(using: .utf8), !data.isEmpty else { return false }
        return (try? JSONDecoder().decode(OpenAIJSONValue.self, from: data)) != nil
    }

    private static func makeChatSession(modelID: String, modelRoot: String?) throws -> ChatSession {
        let normalizedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let normalizedRoot = try modelRoot.flatMap { rawRoot -> String in
            let root = URL(fileURLWithPath: rawRoot).standardizedFileURL
            guard FileManager.default.fileExists(atPath: root.path) else {
                throw StructuredImagePromptAdapterError.invalidModelRoot(root.path)
            }
            return root.path
        }

        if normalizedModelID == Psi3ChatResources.defaultModelId {
            let generator = Psi3ChatGenerator(modelId: Psi3ChatResources.defaultModelId)
            return ChatSession(
                chat: { request, progress in
                    try await generator.chat(request, modelPath: normalizedRoot, progressHandler: progress)
                },
                unload: { await generator.unload() }
            )
        }

        if Gemma4Resources.handles(modelSpec: normalizedModelID) {
            let effectiveModelID = normalizedModelID.isEmpty ? Gemma4Resources.defaultModelId : normalizedModelID
            let generator = Gemma4Generator(modelId: effectiveModelID)
            return ChatSession(
                chat: { request, progress in
                    try await generator.chat(request, modelPath: normalizedRoot, progressHandler: progress)
                },
                unload: { await generator.unload() }
            )
        }

        if ManagedModelCatalog.spec(for: normalizedModelID)?.validationKind == .codegenGGUF {
            let generator = CodeGenGenerator(modelId: normalizedModelID)
            return ChatSession(
                chat: { request, progress in
                    try await generator.chat(request, modelPath: normalizedRoot, progressHandler: progress)
                },
                unload: { await generator.unload() }
            )
        }

        if LFM2Resources.handles(modelSpec: normalizedModelID) {
            let effectiveModelID = normalizedModelID.isEmpty ? LFM2Resources.defaultModelId : normalizedModelID
            let generator = LFM2Generator(modelId: effectiveModelID)
            return ChatSession(
                chat: { request, progress in
                    try await generator.chat(request, modelPath: normalizedRoot, progressHandler: progress)
                },
                unload: { await generator.unload() }
            )
        }

        if normalizedModelID == "text-agent-deepseek-v4-flash" {
            let generator = DeepseekV4FlashGenerator()
            return ChatSession(
                chat: { request, progress in
                    try await generator.chat(request, modelPath: normalizedRoot, progressHandler: progress)
                },
                unload: { await generator.shutdown() }
            )
        }

        let effectiveModelID = normalizedModelID.isEmpty ? Q35Resources.defaultModelId : normalizedModelID
        let generator = Q35Generator(modelId: effectiveModelID)
        return ChatSession(
            chat: { request, progress in
                try await generator.chat(request, modelPath: normalizedRoot, progressHandler: progress)
            },
            unload: { await generator.unload() }
        )
    }

    private static let structuredCaptionSystemPrompt = """
    Convert the user's text-to-image prompt into one valid JSON object for an image model.
    Output only JSON. No markdown, code fences, explanation, or extra text.
    Preserve all explicit constraints. Add plausible visual detail only when it does not contradict the prompt.

    Required top-level keys:
    "short description", "objects", "background setting", "lighting", "aesthetics",
    "photographic characteristics", "style medium", "artistic style", "context", "text render".

    "objects" should describe up to 5 prominent subjects. Prefer object keys:
    "description", "location", "relative size", "shape and color", "texture",
    "appearance details", "relationship", "orientation", "pose", "expression",
    "clothing", "action", "gender", "skin tone and texture", "number of objects".

    "lighting" should cover conditions, direction, and shadows.
    "aesthetics" should cover composition, color scheme, and mood atmosphere.
    "photographic characteristics" should cover depth of field, focus, camera angle, and lens focal length.
    "text render" must be an array of objects with keys:
    "text", "location", "size", "color", "font", "appearance details".
    Any words the image must display — strings in quotes, or text introduced by phrases
    like "reads", "says", "labeled", "titled", or "captioned" — MUST be copied verbatim
    (preserving case) into "text render", one entry per distinct string.
    Describe the surface carrying the text (sign, screen, label) in "objects", but put
    the exact strings only in "text render". Use [] only when no visible text is requested.
    """
}

private func encodeCaptionJSON(_ caption: StructuredImageCaption) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
    let data = try encoder.encode(caption)
    return String(decoding: data, as: UTF8.self)
}

private func validationFailureDetail(error: Error, candidate: String) -> String {
    let base: String
    if case StructuredImagePromptAdapterError.invalidCaptionJSON(let detail) = error {
        base = detail
    } else {
        base = error.localizedDescription
    }
    let collapsed = candidate
        .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !collapsed.isEmpty else { return base }
    var detail = "\(base) Candidate length: \(collapsed.count) characters. Candidate preview: \(collapsed.prefix(700))"
    if collapsed.count > 900 {
        detail += " […] Candidate tail: \(collapsed.suffix(200))"
    }
    return detail
}

private struct FlexibleStructuredImageCaption: Decodable {
    let shortDescription: String?
    let objects: [FlexibleStructuredImageObject]
    let backgroundSetting: String?
    let lighting: FlexibleStructuredImageLighting?
    let lightingText: String?
    let aesthetics: FlexibleStructuredImageAesthetics?
    let aestheticsText: String?
    let photographicCharacteristics: FlexibleStructuredImagePhotographicCharacteristics?
    let photographicCharacteristicsText: String?
    let styleMedium: String?
    let artisticStyle: String?
    let context: String?
    let textRender: [FlexibleStructuredImageTextRender]

    enum CodingKeys: String, CodingKey {
        case shortDescription = "short description"
        case shortDescriptionSnake = "short_description"
        case objects
        case backgroundSetting = "background setting"
        case backgroundSettingSnake = "background_setting"
        case lighting
        case aesthetics
        case photographicCharacteristics = "photographic characteristics"
        case photographicCharacteristicsSnake = "photographic_characteristics"
        case styleMedium = "style medium"
        case styleMediumSnake = "style_medium"
        case artisticStyle = "artistic style"
        case artisticStyleSnake = "artistic_style"
        case context
        case textRender = "text render"
        case textRenderSnake = "text_render"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        shortDescription = container.flexibleString(forKeys: [.shortDescription, .shortDescriptionSnake])
        objects = container.flexibleStructuredImageObjects(forKey: .objects)
        backgroundSetting = container.flexibleString(forKeys: [.backgroundSetting, .backgroundSettingSnake])
        lighting = container.flexibleObject(FlexibleStructuredImageLighting.self, forKey: .lighting)
        lightingText = container.flexibleString(forKey: .lighting)
        aesthetics = container.flexibleObject(FlexibleStructuredImageAesthetics.self, forKey: .aesthetics)
        aestheticsText = container.flexibleString(forKey: .aesthetics)
        photographicCharacteristics = container.flexibleObject(
            FlexibleStructuredImagePhotographicCharacteristics.self,
            forKeys: [.photographicCharacteristics, .photographicCharacteristicsSnake]
        )
        photographicCharacteristicsText = container.flexibleString(
            forKeys: [.photographicCharacteristics, .photographicCharacteristicsSnake]
        )
        styleMedium = container.flexibleString(forKeys: [.styleMedium, .styleMediumSnake])
        artisticStyle = container.flexibleString(forKeys: [.artisticStyle, .artisticStyleSnake])
        context = container.flexibleString(forKey: .context)
        textRender = container.flexibleArray([FlexibleStructuredImageTextRender].self, forKeys: [.textRender, .textRenderSnake])
    }

    func normalized(fallbackPrompt: String) -> StructuredImageCaption {
        let normalizedObjects = objects.prefix(5).enumerated().map { offset, object in
            object.normalized(index: offset)
        }
        let finalObjects = normalizedObjects.isEmpty
            ? [StructuredImageObject.defaultObject(description: fallbackPrompt)]
            : Array(normalizedObjects)

        return StructuredImageCaption(
            shortDescription: shortDescription.nonEmpty ?? fallbackPrompt,
            objects: finalObjects,
            backgroundSetting: backgroundSetting.nonEmpty ?? "Minimal background consistent with the requested image.",
            lighting: lighting?.normalized ?? .defaultLighting(description: lightingText),
            aesthetics: aesthetics?.normalized ?? .defaultAesthetics(description: aestheticsText),
            photographicCharacteristics: photographicCharacteristics?.normalized
                ?? .defaultPhotographicCharacteristics(description: photographicCharacteristicsText),
            styleMedium: styleMedium.nonEmpty ?? "photograph",
            textRender: Array(textRender.prefix(5).map(\.normalized)),
            context: context.nonEmpty ?? "Text-to-image generation prompt.",
            artisticStyle: artisticStyle.nonEmpty ?? "natural realism"
        )
    }
}

private struct FlexibleStructuredImageObject: Decodable {
    let description: String?
    let name: String?
    let attributes: [String]
    let location: String?
    let relationship: String?
    let relativeSize: String?
    let shapeAndColor: String?
    let texture: String?
    let appearanceDetails: String?
    let numberOfObjects: Int?
    let pose: String?
    let expression: String?
    let clothing: String?
    let action: String?
    let gender: String?
    let skinToneAndTexture: String?
    let orientation: String?

    enum CodingKeys: String, CodingKey {
        case description
        case name
        case attributes
        case location
        case relationship
        case relativeSize = "relative size"
        case shapeAndColor = "shape and color"
        case texture
        case appearanceDetails = "appearance details"
        case numberOfObjects = "number of objects"
        case pose
        case expression
        case clothing
        case action
        case gender
        case skinToneAndTexture = "skin tone and texture"
        case orientation
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        description = container.flexibleString(forKey: .description)
        name = container.flexibleString(forKey: .name)
        attributes = container.flexibleStringArray(forKey: .attributes)
        location = container.flexibleString(forKey: .location)
        relationship = container.flexibleString(forKey: .relationship)
        relativeSize = container.flexibleString(forKey: .relativeSize)
        shapeAndColor = container.flexibleString(forKey: .shapeAndColor)
        texture = container.flexibleString(forKey: .texture)
        appearanceDetails = container.flexibleString(forKey: .appearanceDetails)
        numberOfObjects = container.flexibleInt(forKey: .numberOfObjects)
        pose = container.flexibleString(forKey: .pose)
        expression = container.flexibleString(forKey: .expression)
        clothing = container.flexibleString(forKey: .clothing)
        action = container.flexibleString(forKey: .action)
        gender = container.flexibleString(forKey: .gender)
        skinToneAndTexture = container.flexibleString(forKey: .skinToneAndTexture)
        orientation = container.flexibleString(forKey: .orientation)
    }

    init(description: String) {
        self.description = description
        self.name = nil
        self.attributes = []
        self.location = nil
        self.relationship = nil
        self.relativeSize = nil
        self.shapeAndColor = nil
        self.texture = nil
        self.appearanceDetails = nil
        self.numberOfObjects = nil
        self.pose = nil
        self.expression = nil
        self.clothing = nil
        self.action = nil
        self.gender = nil
        self.skinToneAndTexture = nil
        self.orientation = nil
    }

    func normalized(index: Int) -> StructuredImageObject {
        let attributeSummary = attributes.nonEmptyJoined
        let objectName = name.nonEmpty
        let objectDescription = description.nonEmpty
            ?? [objectName, attributeSummary].compactMap(\.nonEmpty).joined(separator: ", ").nonEmpty
            ?? "Prominent image subject"
        let shapeSummary = shapeAndColor.nonEmpty
            ?? [objectName, attributeSummary].compactMap(\.nonEmpty).joined(separator: ", ").nonEmpty
            ?? "visually prominent subject"

        return StructuredImageObject(
            description: objectDescription,
            location: location.nonEmpty ?? (index == 0 ? "center foreground" : "supporting area"),
            relationship: relationship.nonEmpty ?? "part of the requested scene",
            relativeSize: relativeSize.nonEmpty ?? inferredRelativeSize(from: attributes),
            shapeAndColor: shapeSummary,
            texture: texture.nonEmpty ?? inferredTexture(from: attributes),
            appearanceDetails: appearanceDetails.nonEmpty ?? attributeSummary,
            numberOfObjects: numberOfObjects,
            pose: pose.nonEmpty,
            expression: expression.nonEmpty,
            clothing: clothing.nonEmpty,
            action: action.nonEmpty,
            gender: gender.nonEmpty,
            skinToneAndTexture: skinToneAndTexture.nonEmpty,
            orientation: orientation.nonEmpty
        )
    }

    private func inferredRelativeSize(from attributes: [String]) -> String {
        let lowercased = attributes.joined(separator: " ").lowercased()
        if lowercased.contains("small") { return "small within frame" }
        if lowercased.contains("large") { return "large within frame" }
        return "medium within frame"
    }

    private func inferredTexture(from attributes: [String]) -> String? {
        let lowercased = attributes.map { $0.lowercased() }
        if lowercased.contains(where: { $0.contains("matte") }) { return "matte surface" }
        if lowercased.contains(where: { $0.contains("gloss") }) { return "glossy surface" }
        if lowercased.contains(where: { $0.contains("metal") }) { return "metallic texture" }
        return nil
    }
}

private struct FlexibleStructuredImageLighting: Decodable {
    let conditions: String?
    let direction: String?
    let shadows: String?

    enum CodingKeys: String, CodingKey {
        case conditions
        case direction
        case shadows
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        conditions = container.flexibleString(forKey: .conditions)
        direction = container.flexibleString(forKey: .direction)
        shadows = container.flexibleString(forKey: .shadows)
    }

    var normalized: StructuredImageLighting {
        let defaults = StructuredImageLighting.defaultLighting(description: nil)
        return StructuredImageLighting(
            conditions: conditions.nonEmpty ?? defaults.conditions,
            direction: direction.nonEmpty ?? defaults.direction,
            shadows: shadows.nonEmpty
        )
    }
}

private struct FlexibleStructuredImageAesthetics: Decodable {
    let composition: String?
    let colorScheme: String?
    let moodAtmosphere: String?

    enum CodingKeys: String, CodingKey {
        case composition
        case colorScheme = "color scheme"
        case colorSchemeSnake = "color_scheme"
        case moodAtmosphere = "mood atmosphere"
        case moodAtmosphereSnake = "mood_atmosphere"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        composition = container.flexibleString(forKey: .composition)
        colorScheme = container.flexibleString(forKeys: [.colorScheme, .colorSchemeSnake])
        moodAtmosphere = container.flexibleString(forKeys: [.moodAtmosphere, .moodAtmosphereSnake])
    }

    var normalized: StructuredImageAesthetics {
        let defaults = StructuredImageAesthetics.defaultAesthetics(description: nil)
        return StructuredImageAesthetics(
            composition: composition.nonEmpty ?? defaults.composition,
            colorScheme: colorScheme.nonEmpty ?? defaults.colorScheme,
            moodAtmosphere: moodAtmosphere.nonEmpty ?? defaults.moodAtmosphere
        )
    }
}

private struct FlexibleStructuredImagePhotographicCharacteristics: Decodable {
    let depthOfField: String?
    let focus: String?
    let cameraAngle: String?
    let lensFocalLength: String?

    enum CodingKeys: String, CodingKey {
        case depthOfField = "depth of field"
        case depthOfFieldSnake = "depth_of_field"
        case focus
        case cameraAngle = "camera angle"
        case cameraAngleSnake = "camera_angle"
        case lensFocalLength = "lens focal length"
        case lensFocalLengthSnake = "lens_focal_length"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        depthOfField = container.flexibleString(forKeys: [.depthOfField, .depthOfFieldSnake])
        focus = container.flexibleString(forKey: .focus)
        cameraAngle = container.flexibleString(forKeys: [.cameraAngle, .cameraAngleSnake])
        lensFocalLength = container.flexibleString(forKeys: [.lensFocalLength, .lensFocalLengthSnake])
    }

    var normalized: StructuredImagePhotographicCharacteristics {
        let defaults = StructuredImagePhotographicCharacteristics.defaultPhotographicCharacteristics(description: nil)
        return StructuredImagePhotographicCharacteristics(
            depthOfField: depthOfField.nonEmpty ?? defaults.depthOfField,
            focus: focus.nonEmpty ?? defaults.focus,
            cameraAngle: cameraAngle.nonEmpty ?? defaults.cameraAngle,
            lensFocalLength: lensFocalLength.nonEmpty ?? defaults.lensFocalLength
        )
    }
}

private struct FlexibleStructuredImageTextRender: Decodable {
    let text: String?
    let location: String?
    let size: String?
    let color: String?
    let font: String?
    let appearanceDetails: String?

    enum CodingKeys: String, CodingKey {
        case text
        case location
        case size
        case color
        case font
        case appearanceDetails = "appearance details"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = container.flexibleString(forKey: .text)
        location = container.flexibleString(forKey: .location)
        size = container.flexibleString(forKey: .size)
        color = container.flexibleString(forKey: .color)
        font = container.flexibleString(forKey: .font)
        appearanceDetails = container.flexibleString(forKey: .appearanceDetails)
    }

    var normalized: StructuredImageTextRender {
        StructuredImageTextRender(
            text: text.nonEmpty ?? "",
            location: location.nonEmpty ?? "unspecified",
            size: size.nonEmpty ?? "medium",
            color: color.nonEmpty ?? "unspecified",
            font: font.nonEmpty ?? "unspecified",
            appearanceDetails: appearanceDetails.nonEmpty
        )
    }
}

private extension StructuredImageObject {
    static func defaultObject(description: String) -> StructuredImageObject {
        StructuredImageObject(
            description: description,
            location: "center foreground",
            relationship: "primary subject of the requested image",
            relativeSize: "medium within frame",
            shapeAndColor: description,
            texture: nil,
            appearanceDetails: nil,
            numberOfObjects: nil,
            pose: nil,
            expression: nil,
            clothing: nil,
            action: nil,
            gender: nil,
            skinToneAndTexture: nil,
            orientation: nil
        )
    }
}

private extension StructuredImageLighting {
    static func defaultLighting(description: String?) -> StructuredImageLighting {
        StructuredImageLighting(
            conditions: description.nonEmpty ?? "natural lighting consistent with the requested image",
            direction: "unspecified direction",
            shadows: nil
        )
    }
}

private extension StructuredImageAesthetics {
    static func defaultAesthetics(description: String?) -> StructuredImageAesthetics {
        StructuredImageAesthetics(
            composition: description.nonEmpty ?? "clear composition centered on the main subject",
            colorScheme: "colors consistent with the requested image",
            moodAtmosphere: "natural atmosphere"
        )
    }
}

private extension StructuredImagePhotographicCharacteristics {
    static func defaultPhotographicCharacteristics(description: String?) -> StructuredImagePhotographicCharacteristics {
        StructuredImagePhotographicCharacteristics(
            depthOfField: description.nonEmpty ?? "moderate depth of field",
            focus: "sharp focus on the primary subject",
            cameraAngle: "eye-level",
            lensFocalLength: "normal lens"
        )
    }
}

private extension KeyedDecodingContainer {
    func flexibleString(forKey key: Key) -> String? {
        if let string = try? decode(String.self, forKey: key) {
            return string.nonEmpty
        }
        if let int = try? decode(Int.self, forKey: key) {
            return String(int)
        }
        if let double = try? decode(Double.self, forKey: key) {
            return String(double)
        }
        if let bool = try? decode(Bool.self, forKey: key) {
            return String(bool)
        }
        return nil
    }

    func flexibleString(forKeys keys: [Key]) -> String? {
        keys.lazy.compactMap { flexibleString(forKey: $0) }.first
    }

    func flexibleInt(forKey key: Key) -> Int? {
        if let int = try? decode(Int.self, forKey: key) {
            return int
        }
        if let string = flexibleString(forKey: key) {
            return Int(string)
        }
        return nil
    }

    func flexibleObject<T: Decodable>(_ type: T.Type, forKey key: Key) -> T? {
        try? decode(type, forKey: key)
    }

    func flexibleObject<T: Decodable>(_ type: T.Type, forKeys keys: [Key]) -> T? {
        keys.lazy.compactMap { flexibleObject(type, forKey: $0) }.first
    }

    func flexibleArray<T: Decodable>(_ type: [T].Type, forKey key: Key) -> [T] {
        (try? decode(type, forKey: key)) ?? []
    }

    func flexibleArray<T: Decodable>(_ type: [T].Type, forKeys keys: [Key]) -> [T] {
        keys.lazy.map { flexibleArray(type, forKey: $0) }.first { !$0.isEmpty } ?? []
    }

    func flexibleStringArray(forKey key: Key) -> [String] {
        if let values = try? decode([String].self, forKey: key) {
            return values.compactMap(\.nonEmpty)
        }
        if let value = flexibleString(forKey: key) {
            return [value]
        }
        return []
    }

    func flexibleStructuredImageObjects(forKey key: Key) -> [FlexibleStructuredImageObject] {
        if let objects = try? decode([FlexibleStructuredImageObject].self, forKey: key) {
            return objects
        }
        if let strings = try? decode([String].self, forKey: key) {
            return strings.compactMap { value in
                value.nonEmpty.map(FlexibleStructuredImageObject.init(description:))
            }
        }
        if let value = flexibleString(forKey: key) {
            return [FlexibleStructuredImageObject(description: value)]
        }
        return []
    }
}

private extension Optional where Wrapped == String {
    var nonEmpty: String? {
        self?.nonEmpty
    }
}

private extension String {
    var nonEmpty: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    func removingTrailingCommaAndWhitespace() -> String {
        var result = self
        while let last = result.last, last.isWhitespace {
            result.removeLast()
        }
        if result.last == "," {
            result.removeLast()
        }
        return result
    }

    func removingDanglingJSONCommas() -> String {
        var current = self
        while true {
            let next = current.replacingOccurrences(
                of: #",(\s*[}\]])"#,
                with: "$1",
                options: .regularExpression
            )
            guard next != current else { return current }
            current = next
        }
    }
}

private extension Array where Element == String {
    var nonEmptyJoined: String? {
        let joined = compactMap(\.nonEmpty).joined(separator: ", ")
        return joined.nonEmpty
    }
}

private struct ChatSession {
    let chat: (ChatRequest, (@Sendable (ChatProgress) -> Void)?) async throws -> ChatResponse
    let unload: () async -> Void
}

struct StructuredImageCaption: Codable, Equatable {
    let shortDescription: String
    let objects: [StructuredImageObject]
    let backgroundSetting: String
    let lighting: StructuredImageLighting
    let aesthetics: StructuredImageAesthetics
    let photographicCharacteristics: StructuredImagePhotographicCharacteristics
    let styleMedium: String
    let textRender: [StructuredImageTextRender]
    let context: String
    let artisticStyle: String

    enum CodingKeys: String, CodingKey {
        case shortDescription = "short description"
        case objects
        case backgroundSetting = "background setting"
        case lighting
        case aesthetics
        case photographicCharacteristics = "photographic characteristics"
        case styleMedium = "style medium"
        case textRender = "text render"
        case context
        case artisticStyle = "artistic style"
    }
}

struct StructuredImageObject: Codable, Equatable {
    let description: String
    let location: String
    let relationship: String
    let relativeSize: String
    let shapeAndColor: String
    let texture: String?
    let appearanceDetails: String?
    let numberOfObjects: Int?
    let pose: String?
    let expression: String?
    let clothing: String?
    let action: String?
    let gender: String?
    let skinToneAndTexture: String?
    let orientation: String?

    enum CodingKeys: String, CodingKey {
        case description
        case location
        case relationship
        case relativeSize = "relative size"
        case shapeAndColor = "shape and color"
        case texture
        case appearanceDetails = "appearance details"
        case numberOfObjects = "number of objects"
        case pose
        case expression
        case clothing
        case action
        case gender
        case skinToneAndTexture = "skin tone and texture"
        case orientation
    }
}

struct StructuredImageLighting: Codable, Equatable {
    let conditions: String
    let direction: String
    let shadows: String?
}

struct StructuredImageAesthetics: Codable, Equatable {
    let composition: String
    let colorScheme: String
    let moodAtmosphere: String

    enum CodingKeys: String, CodingKey {
        case composition
        case colorScheme = "color scheme"
        case moodAtmosphere = "mood atmosphere"
    }
}

struct StructuredImagePhotographicCharacteristics: Codable, Equatable {
    let depthOfField: String
    let focus: String
    let cameraAngle: String
    let lensFocalLength: String

    enum CodingKeys: String, CodingKey {
        case depthOfField = "depth of field"
        case focus
        case cameraAngle = "camera angle"
        case lensFocalLength = "lens focal length"
    }
}

struct StructuredImageTextRender: Codable, Equatable {
    let text: String
    let location: String
    let size: String
    let color: String
    let font: String
    let appearanceDetails: String?

    enum CodingKeys: String, CodingKey {
        case text
        case location
        case size
        case color
        case font
        case appearanceDetails = "appearance details"
    }
}
