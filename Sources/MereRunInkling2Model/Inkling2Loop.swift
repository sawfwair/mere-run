import MLX

struct Inkling2ResidualState {
    var partial: MLXArray
    var blocks: [MLXArray] = []

    mutating func guide(from reference: Self, strength: Float, startBlock: Int) {
        func extrapolate(_ current: MLXArray, _ earlier: MLXArray) -> MLXArray {
            let value = current.asType(.float32)
            return (value + strength * (value - earlier.asType(.float32))).asType(current.dtype)
        }
        partial = extrapolate(partial, reference.partial)
        for index in startBlock..<blocks.count {
            blocks[index] = extrapolate(blocks[index], reference.blocks[index])
        }
    }
}

struct Inkling2LoopMessages {
    let coreStart: Int
    let blocks: [MLXArray]
    let partial: MLXArray
    let queries: MLXArray
    let keyValueQueries: MLXArray
    let fusedDecode: Bool
    let batched: [MLXArray]?

    func read(layer: Int) -> [MLXArray] {
        let row = 2 * (layer - coreStart)
        if let batched {
            let kv = queries.dim(0) + row
            return [batched[row], batched[kv], batched[kv + 1], batched[row + 1]]
        }
        let query = stacked([queries[row], keyValueQueries[row], keyValueQueries[row + 1], queries[row + 1]], axis: 0)
        return Inkling2Math.residual(blocks, partial: partial, query: query, fusedDecode: fusedDecode)
    }
}

extension Inkling2Model {
    func runLoop(_ loop: Inkling2LoopConfiguration, residual: inout Inkling2ResidualState, cache: Inkling2Cache,
                 capture: Inkling2DSparkFeatures?) throws {
        let layerCount = config.textConfig.numHiddenLayers
        try runLayers(0..<loop.coreStart, residual: &residual, cache: cache, capture: capture)
        let prelude = residual
        try runLayers(loop.layers, residual: &residual, cache: cache, capture: capture)
        let reference = loopCDStrength > 0 ? residual : nil

        let queries = try weights.read(prefix + "middle_cycle_cross_pass_queries")
        let keyValueQueries = try weights.read(prefix + "middle_cycle_cross_pass_kv_queries")
        let fusedDecode = usePackedExperts && (prelude.partial.dim(1) == 1 || cache.isVerifying)
        let messageQueries = fusedDecode && cache.isVerifying ? concatenated([queries, keyValueQueries], axis: 0) : nil
        for pass in 1..<loop.numPasses {
            // The first core boundary repeats the prelude's partial state. Only
            // subsequent boundaries and the core's final state provide messages.
            let messageBlocks = Array(residual.blocks.dropFirst(prelude.blocks.count + 1))
            let batched = messageQueries.map {
                Inkling2Math.residual(messageBlocks, partial: residual.partial, query: $0, fusedDecode: true)
            }
            let messages = Inkling2LoopMessages(
                coreStart: loop.coreStart, blocks: messageBlocks, partial: residual.partial,
                queries: queries, keyValueQueries: keyValueQueries, fusedDecode: fusedDecode, batched: batched
            )
            residual = prelude
            let cacheOffset = layerCount + (pass - 1) * loop.layers.count - loop.coreStart
            try runLayers(loop.layers, residual: &residual, cache: cache, cacheOffset: cacheOffset, previousPass: messages,
                          visitOffset: pass * loop.layers.count, capture: capture)
        }
        // The coda reads the core's residual blocks and final partial state.
        // The prelude and first core boundary are shared across passes.
        if let reference {
            residual.guide(from: reference, strength: loopCDStrength, startBlock: prelude.blocks.count + 1)
        }
        try runLayers(loop.coreEnd..<layerCount, residual: &residual, cache: cache,
                      visitOffset: (loop.numPasses - 1) * loop.layers.count, capture: capture)
    }

    func normalizeLoopInputs(
        _ inputs: [MLXArray], messages: [MLXArray], prefix: String, controllers: [String], norms: [String], dtype: DType
    ) throws -> [MLXArray] {
        let keys = controllers.map { prefix + "controllers." + $0 }
        #if os(macOS)
        if usePackedExperts, config.textConfig.hiddenSize <= 4096,
           Set(keys.map { weights.dtype($0 + ".down_weight") }).count == 1 {
            let group = keys[0]
            let controller: Inkling2LoopController
            if let resident = loopControllerBatches[group] {
                controller = resident
            } else {
                let down = try stacked(keys.map { try weights.read($0 + ".down_weight", retain: false) }, axis: 0)
                    .transposed(0, 2, 1)
                let projections = try keys.map { key in
                    try concatenated([weights.read(key + ".forget_weight", retain: false).asType(.float32),
                                      weights.read(key + ".gain_weight", retain: false).asType(.float32)], axis: 0)
                }
                let up = stacked(projections, axis: 0).transposed(0, 2, 1)
                let norm = try stacked(norms.map { try weights.read(prefix + $0 + ".weight") }, axis: 0)
                eval([down, up, norm])
                controller = Inkling2LoopController(down: down, up: up, normWeights: norm)
                loopControllerBatches[group] = controller
            }
            return controller.apply(inputs, messages: messages, epsilon: config.textConfig.rmsNormEps, dtype: dtype)
        }
        #endif
        return try inputs.indices.map { index in
            let mixed = try mixLoopMessage(messages[index], into: inputs[index], key: keys[index])
            return try norm(mixed, prefix + norms[index]).asType(dtype)
        }
    }

    private func mixLoopMessage(_ message: MLXArray, into input: MLXArray, key: String) throws -> MLXArray {
        let epsilon = config.textConfig.rmsNormEps
        let normalized = concatenated([Inkling2Math.rms(input, epsilon: epsilon),
                                       Inkling2Math.rms(message, epsilon: epsilon)], axis: -1)
        let down = try weights.read(key + ".down_weight")
        let hidden = matmul(normalized.asType(down.dtype), down.T).asType(.float32)
        if usePackedExperts {
            let projection: MLXArray
            if let resident = controllerProjections[key] {
                projection = resident
            } else {
                projection = try concatenated([weights.read(key + ".forget_weight").asType(.float32),
                                               weights.read(key + ".gain_weight").asType(.float32)], axis: 0)
                eval(projection)
                controllerProjections[key] = projection
            }
            let width = input.dim(-1)
            let gates = tanh(matmul(hidden, projection.T))
            return (1 - gates[.ellipsis, ..<width]) * input.asType(.float32) + gates[.ellipsis, width...] * message
        }
        let forget = try tanh(matmul(hidden, weights.read(key + ".forget_weight").asType(.float32).T))
        let gain = try tanh(matmul(hidden, weights.read(key + ".gain_weight").asType(.float32).T))
        return (1 - forget) * input.asType(.float32) + gain * message
    }
}
