import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public enum Cosmos3TokenizerError: LocalizedError, Sendable {
    case missingEndToken
    case missingVisionStartToken

    public var errorDescription: String? {
        switch self {
        case .missingEndToken:
            return "Cosmos3 tokenizer does not define an EOS token."
        case .missingVisionStartToken:
            return "Cosmos3 tokenizer does not define <|vision_start|>."
        }
    }
}

public struct Cosmos3PromptPair: Hashable, Sendable {
    public let conditionalTokenIDs: [Int]
    public let unconditionalTokenIDs: [Int]

    public init(conditionalTokenIDs: [Int], unconditionalTokenIDs: [Int]) {
        self.conditionalTokenIDs = conditionalTokenIDs
        self.unconditionalTokenIDs = unconditionalTokenIDs
    }
}

public final class Cosmos3Tokenizer: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    public let endTokenID: Int
    public let visionStartTokenID: Int

    public init(tokenizer: any Tokenizer) throws {
        guard let endTokenID = tokenizer.eosTokenId else {
            throw Cosmos3TokenizerError.missingEndToken
        }
        guard let visionStartTokenID = tokenizer.convertTokenToId("<|vision_start|>") else {
            throw Cosmos3TokenizerError.missingVisionStartToken
        }
        self.tokenizer = tokenizer
        self.endTokenID = endTokenID
        self.visionStartTokenID = visionStartTokenID
    }

    public static func load(
        resources: Cosmos3Resources,
        hubApi: HubApi = .shared
    ) async throws -> Cosmos3Tokenizer {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: resources.tokenizerRootURL,
            hubApi: hubApi
        )
        return try Cosmos3Tokenizer(tokenizer: tokenizer)
    }

    public func encode(
        prompt: String,
        negativePrompt: String = "",
        numFrames: Int,
        height: Int,
        width: Int,
        fps: Float,
        action: Cosmos3ActionCondition? = nil,
        useSystemPrompt: Bool = false,
        systemPrompt: String? = nil,
        addResolutionTemplate: Bool = true,
        addDurationTemplate: Bool = true
    ) throws -> Cosmos3PromptPair {
        let pair = try Self.renderPrompts(
            prompt: prompt,
            negativePrompt: negativePrompt,
            numFrames: numFrames,
            height: height,
            width: width,
            fps: fps,
            action: action,
            addResolutionTemplate: addResolutionTemplate,
            addDurationTemplate: addDurationTemplate
        )
        return Cosmos3PromptPair(
            conditionalTokenIDs: try encodeChat(
                pair.conditional,
                isImage: numFrames == 1,
                useSystemPrompt: useSystemPrompt,
                systemPrompt: systemPrompt
            ),
            unconditionalTokenIDs: try encodeChat(
                pair.unconditional,
                isImage: numFrames == 1,
                useSystemPrompt: useSystemPrompt,
                systemPrompt: systemPrompt
            )
        )
    }

    public static func renderPrompts(
        prompt: String,
        negativePrompt: String,
        numFrames: Int,
        height: Int,
        width: Int,
        fps: Float,
        action: Cosmos3ActionCondition?,
        addResolutionTemplate: Bool,
        addDurationTemplate: Bool
    ) throws -> (conditional: String, unconditional: String) {
        if let action {
            return (
                try actionJSONPrompt(
                    description: prompt,
                    viewpoint: action.viewpoint,
                    numFrames: numFrames,
                    fps: fps,
                    height: height,
                    width: width
                ),
                negativePrompt
            )
        }
        let isImage = numFrames == 1
        var conditional = prompt
        var unconditional = negativePrompt
        if !isImage && addDurationTemplate {
            conditional = appendSentence(
                conditional,
                String(
                    format: "The video is %.1f seconds long and is of %.0f FPS.",
                    Float(numFrames) / fps,
                    fps
                )
            )
            unconditional = appendSentence(
                unconditional,
                String(
                    format: "The video is not %.1f seconds long and is not of %.0f FPS.",
                    Float(numFrames) / fps,
                    fps
                )
            )
        }
        if addResolutionTemplate {
            conditional = appendSentence(
                conditional,
                "This \(isImage ? "image" : "video") is of \(height)x\(width) resolution."
            )
            unconditional = appendSentence(
                unconditional,
                "This \(isImage ? "image" : "video") is not of \(height)x\(width) resolution."
            )
        }
        return (conditional, unconditional)
    }

    public static func actionJSONPrompt(
        description: String,
        viewpoint: Cosmos3ActionViewpoint,
        numFrames: Int,
        fps: Float,
        height: Int,
        width: Int
    ) throws -> String {
        let durationSeconds = fps > 0 ? Double(numFrames) / Double(fps) : 0
        let duration = durationSeconds.isFinite && durationSeconds >= 0
            ? Int(durationSeconds)
            : 0
        let actionEnd = durationSeconds.isFinite && durationSeconds >= 0
            ? Int(durationSeconds.rounded(.toNearestOrEven))
            : 0
        let minutes = actionEnd / 60
        let seconds = actionEnd % 60
        var normalizedDescription = description.trimmingCharacters(in: .whitespacesAndNewlines)
        if !normalizedDescription.isEmpty,
           !normalizedDescription.hasSuffix("."),
           !normalizedDescription.hasSuffix("!"),
           !normalizedDescription.hasSuffix("?") {
            normalizedDescription += "."
        }
        let framing = try jsonString(viewpoint.framingPrompt)
        let actionDescription = try jsonString(normalizedDescription)
        let aspect = try jsonString(nearestAspectRatio(width: width, height: height))
        return "{"
            + "\"cinematography\": {\"framing\": \(framing)}, "
            + "\"actions\": [{\"time\": \"0:00-\(minutes):\(String(format: "%02d", seconds))\", "
            + "\"description\": \(actionDescription)}], "
            + "\"duration\": \"\(duration)s\", "
            + "\"fps\": \(String(format: "%.1f", fps)), "
            + "\"resolution\": {\"H\": \(height), \"W\": \(width)}, "
            + "\"aspect_ratio\": \(aspect)"
            + "}"
    }

    private func encodeChat(
        _ text: String,
        isImage: Bool,
        useSystemPrompt: Bool,
        systemPrompt: String?
    ) throws -> [Int] {
        var messages: [Message] = []
        if useSystemPrompt || systemPrompt != nil {
            messages.append([
                "role": "system",
                "content": systemPrompt ?? (isImage
                    ? "You are a helpful assistant who will generate images from a give prompt."
                    : "You are a helpful assistant who will generate videos from a give prompt."),
            ])
        }
        messages.append(["role": "user", "content": text])
        var tokenIDs = try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
        tokenIDs.append(endTokenID)
        tokenIDs.append(visionStartTokenID)
        return tokenIDs
    }

    private static func appendSentence(_ base: String, _ addition: String) -> String {
        var trimmed = base
        while trimmed.hasSuffix(".") {
            trimmed.removeLast()
        }
        return trimmed.isEmpty ? addition : "\(trimmed). \(addition)"
    }

    private static func nearestAspectRatio(width: Int, height: Int) -> String {
        let candidates: [(String, Double)] = [
            ("1,1", 1),
            ("4,3", 4.0 / 3.0),
            ("3,4", 3.0 / 4.0),
            ("16,9", 16.0 / 9.0),
            ("9,16", 9.0 / 16.0),
        ]
        let ratio = height > 0 ? Double(width) / Double(height) : 1
        return candidates.min { abs($0.1 - ratio) < abs($1.1 - ratio) }!.0
    }

    private static func jsonString(_ value: String) throws -> String {
        let data = try JSONEncoder().encode(value)
        return String(decoding: data, as: UTF8.self)
    }
}
