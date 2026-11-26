import Foundation
import MLX
import MereRunInkling2Model
@preconcurrency import Tokenizers

enum Inkling2MiniLoRATraining {
    static func train(_ plan: TextLoRATrainingPlan, config: TextLoRATrainingConfig, metadata: [String: String],
                      progress: (@Sendable (ChatProgress) -> Void)?,
                      trainingProgress: (@Sendable (TextLoRATrainingProgress) -> Void)?) async throws -> TextLoRATrainingReport {
        let options = plan.options
        guard let path = options.modelPath ?? Inkling2Resources.localModelRoot(for: options.model)
                ?? ManagedModelResolver.resolveInstalledModel(id: options.model)?.path else {
            throw TextLoRATrainingIssue("Install \(options.model) or pass its checkpoint directory with --model-path.")
        }
        let root = URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
        progress?(ChatProgress(stage: .loadingModel, message: "Opening Inkling 2 Mini checkpoint"))
        let modelConfig = try Inkling2Configuration.decode(Data(contentsOf: root.appendingPathComponent("config.json")))
        guard modelConfig.modelType == "inkling2_loop" else {
            throw TextLoRATrainingIssue("Inkling 2 Mini training requires an inkling2_loop checkpoint.")
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: root)
        let template = try Inkling2ChatTemplate(source: String(contentsOf: root.appendingPathComponent("chat_template.jinja"), encoding: .utf8))
        let length = min(options.maxSequenceLength, modelConfig.textConfig.modelMaxLength)
        func tokenize(_ examples: [TextSFTExample]) throws -> [TextSFTTokenizedExample] {
            try examples.map { example in
                guard example.messages.last?.role == .assistant else {
                    throw TextLoRATrainingIssue("Inkling 2 Mini SFT examples must end with an assistant message.")
                }
                func encode(_ messages: [ChatMessage]) throws -> [Int] {
                    let rendered = try template.render(messages: messages, tools: example.tools, effort: options.reasoningEffort)
                    return tokenizer.encode(text: rendered, addSpecialTokens: false)
                }
                let prefix = try encode(Array(example.messages.dropLast()))
                let target = try TextSFTTrainingBatchBuilder.nativeAssistantTarget(
                    prefixTokenIds: prefix, fullConversationTokenIds: encode(example.messages), maxGenerationPromptBoundaryTokens: 0
                )
                return try TextSFTTrainingBatchBuilder.shiftedTargetExample(
                    prefixTokenIds: prefix, targetTokenIds: target, maxSequenceLength: length
                )
            }
        }
        progress?(ChatProgress(stage: .encoding, message: "Tokenizing Inkling 2 Mini assistant replies"))
        let examples = try tokenize(plan.dataset.examples)
        let evaluation = try tokenize(plan.evaluationDataset?.examples ?? [])
        var metadata = metadata
        metadata["base_model"] = options.model
        metadata["model_root"] = root.path
        metadata["format"] = TextLoRATrainingManifest.inkling2MiniFormat
        metadata["max_sequence_length"] = String(length)
        metadata["reasoning_effort"] = String(options.reasoningEffort)

        return try Stream.withNewDefaultStream {
            let model = try Inkling2Model(config: modelConfig, weights: Inkling2Weights(root: root))
            let project = try model.makeOutputHead()
            let layer = Inkling2MiniLoRALayer(config: modelConfig.textConfig, rank: options.rank,
                                              alpha: options.alpha ?? Float(options.rank), seed: config.seed)
            progress?(ChatProgress(stage: .generating, message: "Training Inkling 2 Mini output-head adapter"))
            return try TextLoRATrainer.train(
                model: Inkling2MiniTrainingHead(layer), loraLayers: ["unembed": layer],
                examples: examples, evaluationExamples: evaluation, config: config,
                outputURL: plan.outputURL, metadata: metadata, progressHandler: trainingProgress,
                prepareInputs: model.frozenHiddenStates,
                gatheredForward: { head, hidden, positions in
                    project(hidden.reshaped(-1, hidden.dim(-1))[positions], head.unembed.callAsFunction)
                }
            ) { head, hidden in project(hidden, head.unembed.callAsFunction) }
        }
    }
}
