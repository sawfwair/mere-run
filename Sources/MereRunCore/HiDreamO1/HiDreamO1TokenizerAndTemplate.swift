import Foundation
@preconcurrency import Hub
@preconcurrency import Tokenizers

public final class HiDreamO1TokenizerAndTemplate {
    public static let boiToken = "<|boi_token|>"
    public static let tmsToken = "<|tms_token|>"
    public static let tmsTokenId = 151_673
    public static let imagePadTokenId = 151_655

    public let tokenizer: any Tokenizer
    public let maxLength: Int

    public init(tokenizer: any Tokenizer, maxLength: Int) {
        self.tokenizer = tokenizer
        self.maxLength = maxLength
    }

    public static func load(
        from resources: HiDreamO1Resources,
        maxLengthOverride: Int? = nil,
        hubApi: HubApi = .shared
    ) async throws -> HiDreamO1TokenizerAndTemplate {
        let tokenizer = try await AutoTokenizer.from(modelFolder: resources.rootURL, hubApi: hubApi)
        let configuredMaxLength = Self.readMaxLength(from: resources.tokenizerConfigURL)
        return HiDreamO1TokenizerAndTemplate(
            tokenizer: tokenizer,
            maxLength: maxLengthOverride ?? configuredMaxLength ?? 131_072
        )
    }

    public func encodeTextToImagePrompt(_ prompt: String) throws -> [Int] {
        try encodePrompt(messages: Self.messages(prompt: prompt, referenceImageCount: 0))
    }

    public func encodeReferencePrompt(_ prompt: String, referenceImageCount: Int) throws -> [Int] {
        try encodePrompt(messages: Self.messages(prompt: prompt, referenceImageCount: referenceImageCount))
    }

    public func encodeReferencePrompt(_ prompt: String, referenceImageTokenCounts: [Int]) throws -> [Int] {
        var encoded = try encodeReferencePrompt(prompt, referenceImageCount: referenceImageTokenCounts.count)
        var imageIndex = 0
        var expanded: [Int] = []
        expanded.reserveCapacity(encoded.count + referenceImageTokenCounts.reduce(0, +))
        for token in encoded {
            guard token == Self.imagePadTokenId, imageIndex < referenceImageTokenCounts.count else {
                expanded.append(token)
                continue
            }
            expanded.append(
                contentsOf: Array(
                    repeating: token,
                    count: max(1, referenceImageTokenCounts[imageIndex])
                )
            )
            imageIndex += 1
        }
        if expanded.count > maxLength {
            encoded = Array(expanded.suffix(maxLength))
        } else {
            encoded = expanded
        }
        return encoded
    }

    public static func messages(prompt: String, referenceImageCount: Int) -> [Message] {
        if referenceImageCount <= 0 {
            return [["role": "user", "content": prompt]]
        }

        var content: [[String: String]] = Array(repeating: ["type": "image"], count: referenceImageCount)
        content.append(["type": "text", "text": prompt])
        return [["role": "user", "content": content]]
    }

    private func encodePrompt(messages: [Message]) throws -> [Int] {
        var encoded = try tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
        encoded.append(contentsOf: tokenizer.encode(text: Self.boiToken + Self.tmsToken))
        if encoded.count > maxLength {
            encoded = Array(encoded.suffix(maxLength))
        }
        return encoded
    }

    private static func readMaxLength(from url: URL) -> Int? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let configured = object["model_max_length"] as? NSNumber else {
            return nil
        }
        let value = configured.intValue
        return value > 0 ? value : nil
    }
}
