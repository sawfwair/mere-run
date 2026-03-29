import Foundation
import MLX
@preconcurrency import Tokenizers
@preconcurrency import Hub

public enum QwenTokenizerError: Error {
  case directoryNotFound(URL)
  case fileNotFound(URL)
  case padTokenMissing
  case padTokenNotInVocabulary(String)
}

public struct QwenTokenBatch {
  public let inputIds: MLXArray
  public let attentionMask: MLXArray

  public init(inputIds: MLXArray, attentionMask: MLXArray) {
    self.inputIds = inputIds
    self.attentionMask = attentionMask
  }
}

public final class QwenTokenizer {
  private let encodeFunction: @Sendable (String) -> [Int]
  private let prefixTokens: [Int]
  private let suffixTokens: [Int]
  private let tokenizer: Tokenizer

  public let padTokenId: Int
  public let maxLength: Int
  public let imageTokenId: Int?
  public let visionStartTokenId: Int?
  public let visionEndTokenId: Int?

  public var templateTokenCount: Int {
    prefixTokens.count
  }

  /// Tokenize text without padding - returns raw token IDs.
  public func encodeText(_ text: String) -> [Int] {
    encodeFunction(text)
  }

  public init(
    padTokenId: Int,
    maxLength: Int,
    prefixTokens: [Int],
    suffixTokens: [Int],
    tokenizer: Tokenizer,
    imageTokenId: Int? = nil,
    visionStartTokenId: Int? = nil,
    visionEndTokenId: Int? = nil,
    encode: @escaping @Sendable (String) -> [Int]
  ) {
    self.padTokenId = padTokenId
    self.maxLength = maxLength
    self.prefixTokens = prefixTokens
    self.suffixTokens = suffixTokens
    self.tokenizer = tokenizer
    self.imageTokenId = imageTokenId
    self.visionStartTokenId = visionStartTokenId
    self.visionEndTokenId = visionEndTokenId
    self.encodeFunction = encode
  }

  public static func load(
    from directory: URL,
    maxLengthOverride: Int? = nil,
    hubApi: HubApi = .shared
  ) throws -> QwenTokenizer {
    let tokenizerDirectory = resolveTokenizerDirectory(directory)
    let tokenizerConfigURL = tokenizerDirectory.appending(path: "tokenizer_config.json")
    let tokenizerDataURL = tokenizerDirectory.appending(path: "tokenizer.json")

    guard FileManager.default.fileExists(atPath: tokenizerDirectory.path) else {
      throw QwenTokenizerError.directoryNotFound(tokenizerDirectory)
    }
    guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
      throw QwenTokenizerError.fileNotFound(tokenizerConfigURL)
    }

    let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
    let tokenizerConfig = try normalizedTokenizerConfig(
      data: tokenizerConfigData,
      url: tokenizerConfigURL,
      overrideTokenizerClass: "Qwen2Tokenizer"
    )
    let addedTokensURL = tokenizerDirectory.appending(path: "added_tokens.json")
    var addedTokens: [String: Int] = [:]
    if FileManager.default.fileExists(atPath: addedTokensURL.path) {
      if let addedData = try? Data(contentsOf: addedTokensURL),
         let addedObject = try? JSONSerialization.jsonObject(with: addedData, options: []) as? [String: Any] {
        for (token, value) in addedObject {
          if let index = value as? Int {
            addedTokens[token] = index
          }
        }
      }
    }

    let tokenizer: Tokenizer
    if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
      let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
      tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    } else {

      let vocabURL = tokenizerDirectory.appending(path: "vocab.json")
      let mergesURL = tokenizerDirectory.appending(path: "merges.txt")
      guard FileManager.default.fileExists(atPath: vocabURL.path),
            FileManager.default.fileExists(atPath: mergesURL.path) else {
        throw QwenTokenizerError.fileNotFound(tokenizerDataURL)
      }
      let tokenizerData = try makeBPETokenizerData(vocabURL: vocabURL, mergesURL: mergesURL)
      tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    let padTokenNode = tokenizerConfig["pad_token"]
    let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
    guard let padToken = padTokenString else {
      throw QwenTokenizerError.padTokenMissing
    }

    guard let padId = tokenizer.convertTokenToId(padToken) ??
      tokenizer.eosTokenId ??
      tokenizer.bosTokenId
    else {
      throw QwenTokenizerError.padTokenNotInVocabulary(padToken)
    }

    let resolvedMaxLength = maxLengthOverride ?? tokenizerConfig["model_max_length"].integer(or: 131_072)

    let prefixTokens = tokenizer.encode(text: promptPrefix)
    let suffixTokens = tokenizer.encode(text: promptSuffix)
    return QwenTokenizer(
      padTokenId: padId,
      maxLength: resolvedMaxLength,
      prefixTokens: prefixTokens,
      suffixTokens: suffixTokens,
      tokenizer: tokenizer,
      imageTokenId: addedTokens["<|image_pad|>"],
      visionStartTokenId: addedTokens["<|vision_start|>"],
      visionEndTokenId: addedTokens["<|vision_end|>"]
    ) { text in
      tokenizer.encode(text: text)
    }
  }

  static func normalizedTokenizerConfig(
    data: Data,
    url: URL,
    overrideTokenizerClass: String
  ) throws -> Config {
    let parsed = try JSONSerialization.jsonObject(with: data, options: [])
    guard var dict = parsed as? [String: Any] else {
      throw QwenTokenizerError.fileNotFound(url)
    }

    if let tokenizerClass = dict["tokenizer_class"] as? String,
       tokenizerClass == "TokenizersBackend" {
      dict["tokenizer_class"] = overrideTokenizerClass
    }

    let nsDict: [NSString: Any] = dict.reduce(into: [:]) { partialResult, pair in
      partialResult[pair.key as NSString] = pair.value
    }
    return Config(nsDict)
  }

  private static func makeBPETokenizerData(vocabURL: URL, mergesURL: URL) throws -> Config {
    let vocabData = try Data(contentsOf: vocabURL)
    guard let vocabObject = try JSONSerialization.jsonObject(with: vocabData, options: []) as? [String: Any] else {
      throw QwenTokenizerError.fileNotFound(vocabURL)
    }
    var vocab: [String: Int] = [:]
    vocab.reserveCapacity(vocabObject.count)
    for (k, v) in vocabObject {
      if let i = v as? Int { vocab[k] = i }
    }

    let tokenizerDir = vocabURL.deletingLastPathComponent()
    let addedTokensURL = tokenizerDir.appending(path: "added_tokens.json")
    var addedTokensMap: [String: Int] = [:]
    if FileManager.default.fileExists(atPath: addedTokensURL.path) {
      if let addedData = try? Data(contentsOf: addedTokensURL),
         let added = try? JSONSerialization.jsonObject(with: addedData, options: []) as? [String: Any] {
        for (k, v) in added {
          if let i = v as? Int {
            vocab[k] = i
            addedTokensMap[k] = i
          }
        }
      }
    }

    let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
    let merges: [String] = mergesText
      .components(separatedBy: .newlines)
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty && !$0.hasPrefix("#") }

    var tokenizerDict: [String: Any] = [
      "model": [
        "vocab": vocab,
        "merges": merges
      ],
      "preTokenizer": [
        "type": "ByteLevel",
        "addPrefixSpace": false,
        "trimOffsets": true,
        "useRegex": true
      ],
      "decoder": [
        "type": "ByteLevel"
      ]
    ]

    if !addedTokensMap.isEmpty {
      var addedList: [[String: Any]] = []
      addedList.reserveCapacity(addedTokensMap.count)
      for (tok, id) in addedTokensMap {
        addedList.append([
          "id": id,
          "content": tok,
          "lstrip": false,
          "rstrip": false,
          "special": true
        ])
      }
      tokenizerDict["addedTokens"] = addedList
    }
    let data = try JSONSerialization.data(withJSONObject: tokenizerDict, options: [])
    let tokenizerData = try JSONDecoder().decode(Config.self, from: data)
    return tokenizerData
  }

  public func encode(
    prompts: [String],
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    precondition(!prompts.isEmpty, "At least one prompt must be provided.")

    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    precondition(targetLength > 0, "Maximum sequence length must be positive.")

    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      var tokens = assembleTokens(for: prompt)
      tokens = Self.trim(tokens, maxLength: targetLength, prefixCount: prefixTokens.count, suffixCount: suffixTokens.count)
      let (ids, mask) = Self.prepareSequence(
        tokens: tokens,
        padTokenId: padTokenId,
        maxLength: targetLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, targetLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func encode(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    if let negativePrompt {
      return encode(prompts: [negativePrompt, prompt], maxLength: maxLength)
    } else {
      return encode(prompts: [prompt], maxLength: maxLength)
    }
  }

  public func encodeChat(
    prompts: [String],
    maxLength: Int? = nil
  ) throws -> QwenTokenBatch {
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      let messages: [[String: Any]] = [
        ["role": "user", "content": prompt]
      ]
      let tokens = try tokenizer.applyChatTemplate(messages: messages)
      let trimmed = Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
      let (ids, mask) = Self.prepareSequence(
        tokens: trimmed,
        padTokenId: padTokenId,
        maxLength: targetLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, targetLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  /// Direct tokenization without chat template - used by Z-Image pipeline
  public func encodePlain(
    prompts: [String],
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      // Direct tokenization without any chat template or prefix/suffix
      let tokens = encodeFunction(prompt)
      let trimmed = Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
      let (ids, mask) = Self.prepareSequence(
        tokens: trimmed,
        padTokenId: padTokenId,
        maxLength: targetLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, targetLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  private func assembleTokens(for prompt: String) -> [Int] {
    let contentTokens = encodeFunction(prompt)
    return prefixTokens + contentTokens + suffixTokens
  }

  public func decode(tokens: [Int]) -> String {
    tokenizer.decode(tokens: tokens)
  }

  public var eosTokenId: Int? {
    tokenizer.eosTokenId
  }

  public func encodeChatForGeneration(
    messages: [[String: Any]],
    maxLength: Int? = nil
  ) throws -> [Int] {
    var text = ""
    for message in messages {
      guard let role = message["role"] as? String,
            let content = message["content"] as? String else {
        continue
      }
      text += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
    }
    text += "<|im_start|>assistant\n"

    let tokens = encodeFunction(text)
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    return Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
  }

  private static func trim(
    _ tokens: [Int],
    maxLength: Int,
    prefixCount: Int,
    suffixCount: Int
  ) -> [Int] {
    guard tokens.count > maxLength else { return tokens }
    if prefixCount + suffixCount >= maxLength {
      return Array(tokens.prefix(maxLength))
    }

    let prefix = Array(tokens.prefix(min(prefixCount, tokens.count)))
    let suffix = suffixCount > 0 ? Array(tokens.suffix(min(suffixCount, maxLength - prefix.count))) : []

    let contentStart = min(prefixCount, tokens.count)
    let contentEnd = max(contentStart, tokens.count - suffixCount)
    let content: [Int]
    if contentEnd > contentStart {
      content = Array(tokens[contentStart..<contentEnd])
    } else {
      content = []
    }

    let availableForContent = max(0, maxLength - prefix.count - suffix.count)
    let trimmedContent = Array(content.prefix(availableForContent))
    return prefix + trimmedContent + suffix
  }

  private static func prepareSequence(
    tokens: [Int],
    padTokenId: Int,
    maxLength: Int
  ) -> ([Int], [Int]) {
    let truncated = Array(tokens.prefix(maxLength))
    let paddingCount = max(0, maxLength - truncated.count)

    var padded = truncated
    if paddingCount > 0 {
      padded.append(contentsOf: Array(repeating: padTokenId, count: paddingCount))
    }

    var attention = Array(repeating: 1, count: truncated.count)
    if paddingCount > 0 {
      attention.append(contentsOf: Array(repeating: 0, count: paddingCount))
    }

    return (padded, attention)
  }

  private static func resolveTokenizerDirectory(_ directory: URL) -> URL {
    let tokenizerPath = directory.appending(path: "tokenizer", directoryHint: .isDirectory)
    if FileManager.default.fileExists(atPath: tokenizerPath.path) {
      return tokenizerPath
    }
    return directory
  }

  private static let promptPrefix: String = """
<|im_start|>system
Describe the key features of the input image (color, shape, size, texture, objects, background),
then explain how the user's text instruction should alter or modify the image.
Generate a new image that meets the user's requirements while maintaining consistency with the
original input where appropriate.<|im_end|>
<|im_start|>user
"""

  private static let promptSuffix: String = """
<|im_end|>
<|im_start|>assistant
"""
}
