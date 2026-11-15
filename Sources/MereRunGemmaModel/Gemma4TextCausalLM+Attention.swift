extension Gemma4TextCausalLM {
    package func setExperimentalCompiledPost(_ enabled: Bool) {
        for layer in languageModel.layers { layer.experimentalCompiledPost = enabled }
    }
    package func setExperimentalHeadNorm(_ enabled: Bool) throws {
        for layer in languageModel.layers { try layer.selfAttention.setExperimentalHeadNorm(enabled) }
    }

    /// Opt-in scheduling changes; attention math, parameters and masks are unchanged.
    package func setExperimentalAttention(_ enabled: Bool) {
        languageModel.experimentalSharedAttentionMasks = enabled
        for layer in languageModel.layers { layer.selfAttention.setDirectPartialRoPE(enabled) }
    }
}
