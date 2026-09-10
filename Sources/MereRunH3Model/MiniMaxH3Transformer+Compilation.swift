import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

extension MiniMaxH3Transformer {
    func compiledBlockForward(
        for block: MiniMaxH3TransformerBlock,
        compressionGate: MiniMaxH3FastH3CompressionGate? = nil
    ) -> MiniMaxH3CompiledBlockForwards {
        if let compiledBlockRunner, let compiledBlockForwards {
            updateCompiledBlockRunner(compiledBlockRunner, from: block)
            return compiledBlockForwards
        }

        let runner = makeCompiledBlockRunner(from: block)
        updateCompiledBlockRunner(runner, from: block)
        let attentionProjection = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.attentionProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )
        }
        let fastH3AttentionProjection: MiniMaxH3CompiledBlockForward? = compressionGate.map { gate in
            let gateStorage = gate.storage
            let gateParameterCount = gate.parameters.count
            return MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
                let hasCachedModulation = inputs.count == 6 + gateParameterCount
                let gateStart = hasCachedModulation ? 6 : 5
                return runner.fastH3AttentionProjection(
                    inputs[0],
                    timeEmbedding: inputs[1],
                    adaLNIndices: inputs[2],
                    rope: MiniMaxH3RotaryEmbedding(cosine: inputs[3], sine: inputs[4]),
                    cachedModulation: hasCachedModulation ? inputs[5] : nil,
                    compressionGateStorage: gateStorage,
                    compressionGateParameters: Array(inputs[gateStart...])
                )
            }
        }
        let attentionOutput = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            [runner.attentionProjectionResidual(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2]
            )]
        }
        let feedForwardProjection = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.feedForwardProjection(
                inputs[0],
                timeEmbedding: inputs[1],
                adaLNIndices: inputs[2],
                cachedModulation: inputs.count == 4 ? inputs[3] : nil
            )
        }
        let feedForwardOutput = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            [runner.feedForwardProjectionResidual(
                inputs[0],
                projected: inputs[1],
                gate: inputs[2]
            )]
        }
        let postAttentionProjection = MLX.compile(
            inputs: [runner]
        ) { (inputs: [MLXArray]) -> [MLXArray] in
            runner.postAttentionProjection(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )
        }
        let postAttention = MLX.compile(inputs: [runner]) { (inputs: [MLXArray]) -> [MLXArray] in
            [runner.postAttention(
                inputs[0],
                attended: inputs[1],
                gate: inputs[2],
                timeEmbedding: inputs[3],
                adaLNIndices: inputs[4],
                cachedModulation: inputs.count == 6 ? inputs[5] : nil
            )]
        }
        let forwards = MiniMaxH3CompiledBlockForwards(
            attentionProjection: attentionProjection,
            fastH3AttentionProjection: fastH3AttentionProjection,
            attentionOutput: attentionOutput,
            feedForwardProjection: feedForwardProjection,
            feedForwardOutput: feedForwardOutput,
            postAttentionProjection: postAttentionProjection,
            postAttention: postAttention
        )
        compiledBlockRunner = runner
        compiledBlockForwards = forwards
        return forwards
    }

    func updateCompiledBlockRunner(
        _ runner: MiniMaxH3TransformerBlock,
        from block: MiniMaxH3TransformerBlock
    ) {
        let parameters = block.parameters().flattened().filter { path, _ in
            block.includesAdaLN || !path.hasPrefix("adaln_proj.")
        }
        runner.update(parameters: ModuleParameters.unflattened(parameters))
    }

    func makeCompiledBlockRunner(
        from block: MiniMaxH3TransformerBlock
    ) -> MiniMaxH3TransformerBlock {
        let runner = MiniMaxH3TransformerBlock(
            configuration: configuration,
            includeAdaLN: block.includesAdaLN
        )
        runner.exactKernelMode = block.exactKernelMode
        runner.enabledExactKernelStages = block.enabledExactKernelStages
        runner.exactKernelDispatchHandler = block.exactKernelDispatchHandler
        runner.exactKernelFallbackHandler = block.exactKernelFallbackHandler
        let replacements: [(String, Module)] = block.leafModules().flattened().compactMap { path, module in
            if let lora = module as? MiniMaxH3RuntimeQuantizedQKVLoRALinear {
                let base = QuantizedLinear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    scales: MLXArray.zeros(lora.scales.shape, dtype: lora.scales.dtype),
                    biases: lora.biases.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    groupSize: lora.groupSize,
                    bits: lora.bits,
                    mode: lora.mode,
                    globalScale: lora.globalScale.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeQuantizedQKVLoRALinear(
                        base: base,
                        queryDown: MLXArray.zeros(lora.queryDown.shape, dtype: lora.queryDown.dtype),
                        queryUp: MLXArray.zeros(lora.queryUp.shape, dtype: lora.queryUp.dtype),
                        keyDown: MLXArray.zeros(lora.keyDown.shape, dtype: lora.keyDown.dtype),
                        keyUp: MLXArray.zeros(lora.keyUp.shape, dtype: lora.keyUp.dtype),
                        valueDown: MLXArray.zeros(lora.valueDown.shape, dtype: lora.valueDown.dtype),
                        valueUp: MLXArray.zeros(lora.valueUp.shape, dtype: lora.valueUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            if let lora = module as? MiniMaxH3RuntimeQuantizedLoRALinear {
                let base = QuantizedLinear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    scales: MLXArray.zeros(lora.scales.shape, dtype: lora.scales.dtype),
                    biases: lora.biases.map { MLXArray.zeros($0.shape, dtype: $0.dtype) },
                    groupSize: lora.groupSize,
                    bits: lora.bits,
                    mode: lora.mode,
                    globalScale: lora.globalScale.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeQuantizedLoRALinear(
                        base: base,
                        loraDown: MLXArray.zeros(lora.loraDown.shape, dtype: lora.loraDown.dtype),
                        loraUp: MLXArray.zeros(lora.loraUp.shape, dtype: lora.loraUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            if let lora = module as? MiniMaxH3RuntimeQKVLoRALinear {
                let base = Linear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeQKVLoRALinear(
                        base: base,
                        queryDown: MLXArray.zeros(lora.queryDown.shape, dtype: lora.queryDown.dtype),
                        queryUp: MLXArray.zeros(lora.queryUp.shape, dtype: lora.queryUp.dtype),
                        keyDown: MLXArray.zeros(lora.keyDown.shape, dtype: lora.keyDown.dtype),
                        keyUp: MLXArray.zeros(lora.keyUp.shape, dtype: lora.keyUp.dtype),
                        valueDown: MLXArray.zeros(lora.valueDown.shape, dtype: lora.valueDown.dtype),
                        valueUp: MLXArray.zeros(lora.valueUp.shape, dtype: lora.valueUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            if let lora = module as? MiniMaxH3RuntimeLoRALinear {
                let base = Linear(
                    weight: MLXArray.zeros(lora.weight.shape, dtype: lora.weight.dtype),
                    bias: lora.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
                )
                return (
                    path,
                    MiniMaxH3RuntimeLoRALinear(
                        base: base,
                        loraDown: MLXArray.zeros(lora.loraDown.shape, dtype: lora.loraDown.dtype),
                        loraUp: MLXArray.zeros(lora.loraUp.shape, dtype: lora.loraUp.dtype),
                        strength: lora.strength
                    )
                )
            }
            guard let quantized = module as? QuantizedLinear else { return nil }
            let weight = MLXArray.zeros(quantized.weight.shape, dtype: quantized.weight.dtype)
            let bias = quantized.bias.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            let scales = MLXArray.zeros(quantized.scales.shape, dtype: quantized.scales.dtype)
            let biases = quantized.biases.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            let globalScale = quantized.globalScale.map { MLXArray.zeros($0.shape, dtype: $0.dtype) }
            let clone: Module
            if let portable = quantized as? PortableQuantizedLinear {
                let portableClone = PortableQuantizedLinear(
                    weight: weight,
                    bias: bias,
                    scales: scales,
                    biases: biases,
                    groupSize: portable.groupSize,
                    bits: portable.bits,
                    mode: portable.mode,
                    globalScale: globalScale
                )
                portableClone.cacheDenseFallbackWeight = portable.cacheDenseFallbackWeight
                portableClone.useUncachedDenseFallback = portable.useUncachedDenseFallback
                clone = portableClone
            } else {
                clone = QuantizedLinear(
                    weight: weight,
                    bias: bias,
                    scales: scales,
                    biases: biases,
                    groupSize: quantized.groupSize,
                    bits: quantized.bits,
                    mode: quantized.mode,
                    globalScale: globalScale
                )
            }
            return (path, clone)
        }
        runner.update(modules: ModuleChildren.unflattened(replacements))
        return runner
    }

}
