import MLXNN

extension Gemma4TextCausalLM {
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
