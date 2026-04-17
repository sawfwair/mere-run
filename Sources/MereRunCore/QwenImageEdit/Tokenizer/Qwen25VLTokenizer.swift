import Foundation
import MLX

/// Tokenizer adapter for Qwen2.5-VL used in Qwen-Image-Edit.
/// Wraps the base QwenTokenizer with image-editing specific formatting.
public final class Qwen25VLTokenizer {
    private let baseTokenizer: QwenTokenizer

    public var padTokenId: Int { baseTokenizer.padTokenId }
    public var imageTokenId: Int? { baseTokenizer.imageTokenId }
    public var visionStartTokenId: Int? { baseTokenizer.visionStartTokenId }
    public var visionEndTokenId: Int? { baseTokenizer.visionEndTokenId }

    public init(baseTokenizer: QwenTokenizer) {
        self.baseTokenizer = baseTokenizer
    }

    /// Load tokenizer from directory
    public static func load(from directory: URL, maxLength: Int? = nil) throws -> Qwen25VLTokenizer {
        let base = try QwenTokenizer.load(from: directory, maxLengthOverride: maxLength)
        return Qwen25VLTokenizer(baseTokenizer: base)
    }

    // MARK: - Encoding for Image Editing

    /// Encode an editing prompt with optional image placeholders
    /// - Parameters:
    ///   - prompt: The editing instruction (e.g., "Change the color to blue")
    ///   - numImageTokens: Number of image tokens to insert for the input image
    ///   - maxLength: Maximum sequence length
    /// - Returns: Token batch ready for the model
    public func encodeForEditing(
        prompt: String,
        numImageTokens: Int = 256,
        maxLength: Int? = nil
    ) -> QwenTokenBatch {
        // Format: <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>\n{prompt}
        let formattedPrompt = formatEditingPrompt(prompt: prompt, numImageTokens: numImageTokens)
        return baseTokenizer.encodePlain(prompts: [formattedPrompt], maxLength: maxLength)
    }

    /// Encode prompt and negative prompt for CFG
    /// - Parameters:
    ///   - prompt: The editing instruction
    ///   - negativePrompt: What to avoid
    ///   - numImageTokens: Number of image tokens
    ///   - maxLength: Maximum sequence length
    /// - Returns: Token batch with [negative, positive] ordering for CFG
    public func encodeForEditingWithCFG(
        prompt: String,
        negativePrompt: String,
        numImageTokens: Int = 256,
        maxLength: Int? = nil
    ) -> QwenTokenBatch {
        let positiveFormatted = formatEditingPrompt(prompt: prompt, numImageTokens: numImageTokens)
        let negativeFormatted = formatEditingPrompt(prompt: negativePrompt, numImageTokens: numImageTokens)
        return baseTokenizer.encodePlain(prompts: [negativeFormatted, positiveFormatted], maxLength: maxLength)
    }

    /// Encode for generation (text-to-image, no input image)
    public func encodeForGeneration(
        prompt: String,
        negativePrompt: String? = nil,
        maxLength: Int? = nil
    ) -> QwenTokenBatch {
        if let negative = negativePrompt {
            return baseTokenizer.encodePlain(prompts: [negative, prompt], maxLength: maxLength)
        }
        return baseTokenizer.encodePlain(prompts: [prompt], maxLength: maxLength)
    }

    // MARK: - Formatting

    private func formatEditingPrompt(prompt: String, numImageTokens: Int) -> String {
        var result = ""

        // Add vision tokens if we have them
        if visionStartTokenId != nil, imageTokenId != nil, visionEndTokenId != nil {
            // The actual tokens will be handled by the model, but we need placeholders
            // Format: <|vision_start|><|image_pad|>...<|vision_end|>
            result += "<|vision_start|>"
            result += String(repeating: "<|image_pad|>", count: numImageTokens)
            result += "<|vision_end|>\n"
        }

        result += prompt
        return result
    }

    /// Calculate number of image tokens for a given image size
    /// Based on spatial merge size and patch size
    public static func imageTokenCount(
        imageHeight: Int,
        imageWidth: Int,
        patchSize: Int = 14,
        spatialMergeSize: Int = 2
    ) -> Int {
        let patchH = imageHeight / patchSize
        let patchW = imageWidth / patchSize
        let mergedH = patchH / spatialMergeSize
        let mergedW = patchW / spatialMergeSize
        return mergedH * mergedW
    }
}

// MARK: - Chat Templates for Editing

extension Qwen25VLTokenizer {
    /// Standard editing prompt template
    public static let editingSystemPrompt = """
        You are an image editing assistant. Analyze the input image and apply the requested modifications while preserving unaffected regions.
        """

    /// Format a complete editing conversation
    public func formatEditingConversation(
        systemPrompt: String = editingSystemPrompt,
        userPrompt: String,
        hasInputImage: Bool = true
    ) -> [[String: Any]] {
        var messages: [[String: Any]] = []

        // System message
        messages.append([
            "role": "system",
            "content": systemPrompt
        ])

        // User message with optional image
        var userContent = ""
        if hasInputImage {
            userContent += "<|vision_start|><|image_pad|><|vision_end|>\n"
        }
        userContent += userPrompt

        messages.append([
            "role": "user",
            "content": userContent
        ])

        return messages
    }
}
