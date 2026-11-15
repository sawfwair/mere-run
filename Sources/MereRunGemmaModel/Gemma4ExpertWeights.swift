import MLX
import MLXNN

extension Gemma4TextCausalLM {
    package static var supportsMetalExpertVerification: Bool { Gemma4MetalExperts.isSupported }

    package var usesDistributedMetalExpertReduction: Bool {
        languageModel.layers.contains { $0.experts?.switchGLU.downProj.metalVerification?.distributedReduction == true }
    }

    /// Install lazy packed slots before loading any checkpoint tensors.
    package func preparePackedExpertWeights(groupSize: Int, layerBits: [Int], expandedPrefill: Bool = false,
                                           metalVerification: Bool = false) throws {
        guard layerBits.count == languageModel.layers.count, [32, 64].contains(groupSize),
              layerBits.allSatisfy({ [4, 6, 8].contains($0) }),
              !expandedPrefill || (groupSize == 64 && layerBits.allSatisfy { $0 == 6 }) else {
            throw Gemma4Error.unsupportedConfiguration("Packed experts require a valid affine precision for every layer.")
        }
        let expander = expandedPrefill ? Gemma4ExpandedExpertWeights() : nil
        if metalVerification && experimentalMetalExperts == nil { experimentalMetalExperts = try Gemma4MetalExperts() }
        for (index, layer) in languageModel.layers.enumerated() {
            guard let experts = layer.experts else { continue }
            let packed = Gemma4SwitchGLU(packedConfig: languageModel.config, groupSize: groupSize,
                bits: layerBits[index], prefillExpander: expander)
            packed.gateUpProj?.metalVerification = metalVerification ? experimentalMetalExperts : nil
            packed.downProj.metalVerification = metalVerification ? experimentalMetalExperts : nil
            experts.update(modules: ModuleChildren.unflattened([("switch_glu", packed)]))
        }
    }

    /// Call once after loading the original weights and before creating a cache.
    /// Quantization replaces expert modules; routers and dense layers stay intact.
    package func prepareExperimentalExperts(groupSize: Int? = nil, bits: Int = 8, expandedPrefill: Bool = false, layerBits: [Int]? = nil, verificationGEMV: Bool = false, metalVerification: Bool = false) throws {
        guard [4, 6, 8].contains(bits), layerBits.map({ $0.count == languageModel.layers.count && $0.allSatisfy { [4, 6, 8].contains($0) } }) ?? true,
              layerBits == nil || groupSize != nil else {
            throw Gemma4Error.unsupportedConfiguration("Expert packing requires Q4, Q6, or Q8 and one precision per decoder layer.")
        }
        guard !expandedPrefill || (bits == 6 && layerBits == nil && groupSize == 64) else {
            throw Gemma4Error.unsupportedConfiguration("Expanded prefill requires uniform affine Q6 with groups of 64.")
        }
        let expander = expandedPrefill ? Gemma4ExpandedExpertWeights() : nil
        if metalVerification && experimentalMetalExperts == nil { experimentalMetalExperts = try Gemma4MetalExperts() }
        let metal = metalVerification ? experimentalMetalExperts : nil
        if groupSize != nil { Memory.clearCache() }
        for (index, layer) in languageModel.layers.enumerated() {
            guard let experts = layer.experts else { continue }
            if let groupSize {
                let packed = Gemma4SwitchGLU(packing: experts.switchGLU, groupSize: groupSize, bits: layerBits?[index] ?? bits, prefillExpander: expander)
                experts.update(modules: ModuleChildren.unflattened([("switch_glu", packed)]))
                // Release retired BF16 allocations before packing the next layer.
                // Otherwise the allocator pool can exceed the GPU working-set limit.
                Memory.clearCache()
            }
            experts.switchGLU.useGEMVForVerification = verificationGEMV
            experts.switchGLU.gateUpProj?.metalVerification = metal
            experts.switchGLU.downProj.metalVerification = metal
        }
    }

    /// Switch only dispatch; keep packed parameters and compiled shaders resident.
    package func setMetalExpertVerification(_ enabled: Bool, byteLoads: Bool = true, optimizedLayout: Bool = true,
                                            fusedActivation: Bool = false, distributedReduction: Bool? = nil) throws {
        if enabled && experimentalMetalExperts == nil { experimentalMetalExperts = try Gemma4MetalExperts() }
        experimentalMetalExperts?.byteLoads = byteLoads
        experimentalMetalExperts?.optimizedLayout = optimizedLayout
        experimentalMetalExperts?.fusedActivation = fusedActivation
        experimentalMetalExperts?.distributedReduction = distributedReduction ?? Gemma4MetalExperts.automaticDistributedReduction
        for layer in languageModel.layers {
            layer.experts?.switchGLU.gateUpProj?.metalVerification = enabled ? experimentalMetalExperts : nil
            layer.experts?.switchGLU.downProj.metalVerification = enabled ? experimentalMetalExperts : nil
        }
    }

    package func prepareForFullPrecisionExpertWeights() {
        prepareFullPrecisionExperts(languageModel)
    }
}

extension Gemma4UnifiedCausalLM {
    package func prepareForFullPrecisionExpertWeights() {
        prepareFullPrecisionExperts(languageModel)
    }
}

/// Quantized expert modules reserve scale slots for checkpoint updates. A floating-point
/// checkpoint has no such tensors; leaving the placeholders selects the quantized kernel.
private func prepareFullPrecisionExperts(_ model: Gemma4LanguageModel) {
    for layer in model.layers {
        guard let experts = layer.experts else { continue }
        let fullPrecision = Gemma4SwitchGLU(config: model.config, quantized: false)
        experts.update(modules: ModuleChildren.unflattened([("switch_glu", fullPrecision)]))
    }
}
