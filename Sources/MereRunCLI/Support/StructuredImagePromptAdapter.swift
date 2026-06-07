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
        let candidate = cleanedJSONCandidate(from: rawJSON)
        do {
            let caption = try validateCaptionJSON(candidate)
            return try encodeCaptionJSON(caption)
        } catch {
            guard let data = candidate.data(using: .utf8), !data.isEmpty else {
                throw StructuredImagePromptAdapterError.invalidCaptionJSON("Output was empty.")
            }
            do {
                let flexibleCaption = try JSONDecoder().decode(FlexibleStructuredImageCaption.self, from: data)
                let normalizedCaption = flexibleCaption.normalized(fallbackPrompt: fallbackPrompt)
                let normalizedJSON = try encodeCaptionJSON(normalizedCaption)
                _ = try validateCaptionJSON(normalizedJSON)
                return normalizedJSON
            } catch {
                throw StructuredImagePromptAdapterError.invalidCaptionJSON(
                    validationFailureDetail(error: error, candidate: candidate)
                )
            }
        }
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
            return String(cleaned[firstBrace...lastBrace])
        }
        return cleaned
    }

    private static func expand(
        prompt: String,
        maxTokens: Int,
        session: ChatSession,
        progressHandler: (@Sendable (String) -> Void)?
    ) async throws -> String {
        var messages = [
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
            do {
                return try normalizedCaptionJSON(from: response.response, fallbackPrompt: prompt)
            } catch {
                lastDetail = validationFailureDetail(error: error, candidate: cleanedJSONCandidate(from: response.response))
                messages.append(ChatMessage(role: .assistant, content: response.response))
                messages.append(ChatMessage(
                    role: .user,
                    content: """
                    The previous response was invalid for this task:
                    \(lastDetail)

                    Return only one valid JSON object matching the exact template from the system message. Do not use "name" or "attributes" as object keys. "text render" must be an array; use [] when no text should appear. Use null only for optional nested values.
                    """
                ))
            }
        }

        throw StructuredImagePromptAdapterError.invalidCaptionJSON(lastDetail)
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
    "text render" must be an array; use [] when no visible text is requested.
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
    let preview = candidate
        .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .prefix(700)
    guard !preview.isEmpty else { return base }
    return "\(base) Candidate preview: \(preview)"
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
        case objects
        case backgroundSetting = "background setting"
        case lighting
        case aesthetics
        case photographicCharacteristics = "photographic characteristics"
        case styleMedium = "style medium"
        case artisticStyle = "artistic style"
        case context
        case textRender = "text render"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        shortDescription = container.flexibleString(forKey: .shortDescription)
        objects = container.flexibleStructuredImageObjects(forKey: .objects)
        backgroundSetting = container.flexibleString(forKey: .backgroundSetting)
        lighting = container.flexibleObject(FlexibleStructuredImageLighting.self, forKey: .lighting)
        lightingText = container.flexibleString(forKey: .lighting)
        aesthetics = container.flexibleObject(FlexibleStructuredImageAesthetics.self, forKey: .aesthetics)
        aestheticsText = container.flexibleString(forKey: .aesthetics)
        photographicCharacteristics = container.flexibleObject(
            FlexibleStructuredImagePhotographicCharacteristics.self,
            forKey: .photographicCharacteristics
        )
        photographicCharacteristicsText = container.flexibleString(forKey: .photographicCharacteristics)
        styleMedium = container.flexibleString(forKey: .styleMedium)
        artisticStyle = container.flexibleString(forKey: .artisticStyle)
        context = container.flexibleString(forKey: .context)
        textRender = container.flexibleArray([FlexibleStructuredImageTextRender].self, forKey: .textRender)
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
        case moodAtmosphere = "mood atmosphere"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        composition = container.flexibleString(forKey: .composition)
        colorScheme = container.flexibleString(forKey: .colorScheme)
        moodAtmosphere = container.flexibleString(forKey: .moodAtmosphere)
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
        case focus
        case cameraAngle = "camera angle"
        case lensFocalLength = "lens focal length"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        depthOfField = container.flexibleString(forKey: .depthOfField)
        focus = container.flexibleString(forKey: .focus)
        cameraAngle = container.flexibleString(forKey: .cameraAngle)
        lensFocalLength = container.flexibleString(forKey: .lensFocalLength)
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

    func flexibleArray<T: Decodable>(_ type: [T].Type, forKey key: Key) -> [T] {
        (try? decode(type, forKey: key)) ?? []
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
