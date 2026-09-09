import Foundation
import MLX
import MLXRandom

public struct PromptEnhanceConfig {
  public var maxNewTokens: Int
  public var temperature: Float
  public var topP: Float
  public var repetitionPenalty: Float?
  public var repetitionContextSize: Int
  public var eosTokenId: Int
  public var stopTokenIds: Set<Int>

  public init(
    maxNewTokens: Int = 512,
    temperature: Float = 0.7,
    topP: Float = 0.9,
    repetitionPenalty: Float? = 1.05,
    repetitionContextSize: Int = 20,
    eosTokenId: Int = 151645,
    stopTokenIds: Set<Int> = [151645, 151643]
  ) {
    self.maxNewTokens = maxNewTokens
    self.temperature = temperature
    self.topP = topP
    self.repetitionPenalty = repetitionPenalty
    self.repetitionContextSize = repetitionContextSize
    self.eosTokenId = eosTokenId
    self.stopTokenIds = stopTokenIds
  }
}

extension QwenTextEncoder {
  public func generate(
    inputIds: MLXArray,
    config: PromptEnhanceConfig = .init()
  ) -> [Int] {
    return generate(inputIds: inputIds, config: config, tokenCallback: nil)
  }

  public func generate(
    inputIds: MLXArray,
    config: PromptEnhanceConfig = .init(),
    tokenCallback: ((Int) -> Bool)? = nil
  ) -> [Int] {
    let cache: [KVCache] = (0..<configuration.numHiddenLayers).map { _ in
      KVCacheSimple(step: 256)
    }

    MLX.eval(inputIds)
    var tokens = inputIds.asArray(Int32.self).map { Int($0) }
    let inputLength = tokens.count

    let logits = encoder.forwardCausal(inputIds: inputIds, cache: cache, lastPositionOnly: true)
    MLX.eval(logits)
    let generationConfig = GenerationConfig(
      maxTokens: config.maxNewTokens,
      temperature: config.temperature,
      topP: config.topP,
      repetitionPenalty: config.repetitionPenalty,
      repetitionContextSize: config.repetitionContextSize
    )
    var eosTokens = config.stopTokenIds
    eosTokens.insert(config.eosTokenId)
    let result = AutoregressiveDecodeEngine.decodeStateful(
      AutoregressiveDecodeRequest(
        initialLogits: logits,
        generationConfig: generationConfig,
        eosTokens: eosTokens,
        tokenBudget: config.maxNewTokens,
        historySeedTokens: tokens
      ),
      stepForward: { nextInput in
        self.encoder.forwardCausal(inputIds: nextInput, cache: cache)
      },
      shouldContinue: tokenCallback
    )

    tokens.append(contentsOf: result.generatedTokens)
    return Array(tokens.dropFirst(inputLength))
  }

  public func enhancePrompt(
    _ prompt: String,
    tokenizer: QwenTokenizer,
    config: PromptEnhanceConfig = .init()
  ) throws -> String {
    let userMessage = "用户输入 prompt: \(prompt)"
    let messages: [[String: Any]] = [
      ["role": "system", "content": Self.peSystemPrompt],
      ["role": "user", "content": userMessage]
    ]

    let tokens = try tokenizer.encodeChatForGeneration(messages: messages, maxLength: tokenizer.maxLength)
    let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

    var effectiveConfig = config
    if let eosId = tokenizer.eosTokenId {
      effectiveConfig.eosTokenId = eosId
    }

    let generatedTokens = generate(inputIds: inputIds, config: effectiveConfig)
    var result = tokenizer.decode(tokens: generatedTokens)

    if let thinkEndRange = result.range(of: "</think>") {
      result = String(result[thinkEndRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    } else if result.contains("<think>") {
      result = ""
    }

    return result
  }

  /// Official Z-Image PE (Prompt Enhancer) system prompt
  /// Mirrors the upstream `pe.py` reference implementation.
  static let peSystemPrompt: String = """
你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。

你的工作流程严格遵循一个逻辑序列：

首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。

接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。

然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。

最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。

你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。

仅严格输出最终的修改后的prompt，不要输出任何其他内容。
"""
}
