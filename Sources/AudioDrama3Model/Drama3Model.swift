#if !os(iOS)
import Foundation
import MLX
import MLXNN
import MLXRandom
import MereRunQwenModel

public final class Drama3Model: Module {
    @ModuleInfo var backbone: Drama3Backbone
    @ModuleInfo var tower: Drama3Tower
    @ModuleInfo(key: "lm_head") var head: Linear
    public let config: Drama3Config

    public init(config: Drama3Config) throws {
        self.config = config
        _backbone.wrappedValue = Drama3Backbone(config: try config.backboneConfig())
        _tower.wrappedValue = Drama3Tower(config)
        _head.wrappedValue = Linear(config.text.hiddenSize, config.text.vocabSize, bias: false)
    }

    public static func sample(_ logits: MLXArray, temperature: Float, topK: Int = 30, topP: Float = 0.9) -> Int {
        draw(samplingCandidates(logits, temperature: temperature, topK: topK, topP: topP))
    }

    package static func semanticCandidates(_ logits: MLXArray, temperature: Float,
                                           redraw: Bool = false) -> (ids: MLXArray, logits: MLXArray) {
        // Upstream 444001b0: top-k 30 on the main draw can stall after the first
        // phrase. Only the RAS redraw and acoustic heads retain that cap.
        samplingCandidates(logits, temperature: redraw ? 1 : temperature,
            topK: redraw ? 30 : logits.size, topP: redraw ? 0.95 : 0.9, keepCrossingToken: true)
    }

    package static func samplingCandidates(_ logits: MLXArray, temperature: Float, topK: Int = 30,
                                           topP: Float = 0.9, keepCrossingToken: Bool = false)
        -> (ids: MLXArray, logits: MLXArray) {
        let scaled = logits.asType(.float32) / max(temperature, 1e-5)
        let ids = MLX.argSort(-scaled)
        let ordered = scaled[ids]
        let probability = MLX.softmax(ordered)
        let cumulative = MLX.cumsum(probability)
        let cutoff = keepCrossingToken
            ? MLX.concatenated([MLXArray.zeros([1]), cumulative[..<(logits.size - 1)]]) : cumulative
        let ranks = MLXArray(0..<logits.size)
        let dropped = MLX.logicalAnd(ranks .> 0, MLX.logicalOr(cutoff .> topP, ranks .>= topK))
        return (ids, MLX.where(dropped, MLXArray(-Float.infinity), ordered))
    }

    private static func draw(_ candidates: (ids: MLXArray, logits: MLXArray)) -> Int {
        let choice = MLXRandom.categorical(candidates.logits).item(Int.self)
        return candidates.ids[choice].item(Int.self)
    }

    /// Prefill the tower using *next* prompt frames. All thirteen tower caches
    /// persist across acoustic time, rather than restarting for each codebook.
    public func generate(prompt: [[Int32]], maxFrames: Int = 750, temperature: Float = 1,
                         seed: UInt64? = nil, onFrame: ([Int32]) throws -> Void = { _ in }) throws -> [[Int32]] {
        guard !prompt.isEmpty, prompt.allSatisfy({ $0.count == 11 }), maxFrames > 0,
              temperature.isFinite, temperature >= 0 else { throw Drama3Error.invalid("invalid generation parameters") }
        guard prompt.count + maxFrames <= config.text.maxPositionEmbeddings else {
            throw Drama3Error.invalid("prompt and generation exceed the context window")
        }
        if let seed { MLXRandom.seed(seed) }
        let cache = backbone.makeCache(), towerCache = tower.makeCache()
        var lastHidden: MLXArray?
        // Keep prefill bounded. The final prompt row is left for decode_step.
        for start in stride(from: 0, to: prompt.count, by: 64) {
            try Task.checkCancellation()
            let end = min(start + 64, prompt.count)
            let frames = MLXArray(Array(prompt[start..<end]).flatMap { $0 }).reshaped(1, end - start, 11)
            let input = tower.backboneInput(frames, text: backbone.embeddings(frames[.ellipsis, 0]))
            let hidden = backbone(input, cache: cache)
            let teacherEnd = min(end, prompt.count - 1)
            if teacherEnd > start {
                let n = teacherEnd - start
                let next = MLXArray(Array(prompt[(start + 1)...teacherEnd]).flatMap { $0 }).reshaped(1, n, 11)
                tower.prefill(hidden: hidden[0..., 0..<n, 0...], input: frames[0..., 0..<n, 0...], next: next, cache: towerCache)
            }
            lastHidden = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
            MLX.eval(lastHidden!)
            Memory.clearCache()
        }
        guard var hidden = lastHidden else { throw Drama3Error.invalid("missing prompt states") }
        var previous = prompt.last!
        var output: [[Int32]] = [], history: [Int] = []
        var completed = false
        let candidates = [config.eos] + Array(config.semanticStart...config.semanticEnd)
        let candidateIDs = MLXArray(candidates.map(Int32.init))
        for _ in 0..<maxFrames {
            try Task.checkCancellation()
            let logits = head(hidden).reshaped(-1)[candidateIDs]
            let first = candidates[Self.draw(Self.semanticCandidates(logits, temperature: temperature))]
            let redraw = candidates[Self.draw(Self.semanticCandidates(logits, temperature: temperature, redraw: true))]
            let main = history.suffix(16).contains(first) && (config.semanticStart...config.semanticEnd).contains(first)
                ? redraw : first
            let input = MLXArray(previous).reshaped(1, 1, 11)
            let next = tower.decode(hidden: hidden, input: input, main: main,
                sample: { Self.sample($0, temperature: temperature) }, cache: towerCache)
            if main == config.eos { completed = true; break }
            output.append(Array(next.dropFirst()))
            history.append(main)
            try onFrame(Array(next.dropFirst()))
            let frames = MLXArray(next).reshaped(1, 1, 11)
            hidden = backbone(tower.backboneInput(frames, text: backbone.embeddings(frames[.ellipsis, 0])), cache: cache)
            MLX.eval(hidden)
            // The checkpoint stays active; completed temporary projections do not
            // need a RAM-sized allocator cache between acoustic frames.
            Memory.clearCache()
            previous = next
        }
        guard completed else { throw Drama3Error.invalid("generation reached the acoustic frame limit; shorten the text") }
        guard !output.isEmpty else { throw Drama3Error.invalid("model generated no audio frames") }
        return (0..<10).map { book in output.map { $0[book] } }
    }
}

#endif
