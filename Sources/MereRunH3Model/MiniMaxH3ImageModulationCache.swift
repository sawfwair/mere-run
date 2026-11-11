import MLX

/// Released-precision outputs for the six image timesteps and fixed reference timestep.
package struct MiniMaxH3ImageModulationCache {
    package let timeEmbeddings: MLXArray
    package let blocks: [MLXArray]
    package let finalModulations: MLXArray

    package init(timeEmbeddings: MLXArray, blocks: [MLXArray], finalModulations: MLXArray) {
        self.timeEmbeddings = timeEmbeddings
        self.blocks = blocks
        self.finalModulations = finalModulations
    }

    package func matches(_ configuration: MiniMaxH3TransformerConfiguration) -> Bool {
        timeEmbeddings.shape == [6, 2, configuration.timeEmbeddingDimension]
            && blocks.count == configuration.layerCount
            && blocks.allSatisfy { $0.shape == [6, 6, 6 * configuration.hiddenSize] }
            && finalModulations.shape == [6, 2, 2 * configuration.hiddenSize]
    }
}
