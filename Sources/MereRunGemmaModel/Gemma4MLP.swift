import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    private var fusedGateUp: FusedQuantizedProjection?
    private var fusedGateUpAttempted = false

    init(config: Gemma4TextConfig, layerIndex: Int, forceKVShared: Bool = false) {
        let firstSharedIndex = config.numHiddenLayers - config.numKVSharedLayers
        let isSharedKVLayer = forceKVShared || (config.numKVSharedLayers > 0 && layerIndex >= firstSharedIndex)
        let widthMultiplier = (config.useDoubleWideMLP && isSharedKVLayer) ? 2 : 1
        let intermediate = config.intermediateSize * widthMultiplier

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let fused = resolvedFusedGateUp() {
            let parts = fused.callSplit(x)
            return downProj(geluApproximate(parts[0]) * parts[1])
        }
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }

    func resolvedFusedGateUp() -> FusedQuantizedProjection? {
        guard Gemma4FusedProjectionPolicy.enabled else { return nil }
        let sources: [Linear?] = [gateProj, upProj]
        if let fused = fusedGateUp {
            if fused.matches(sources) { return fused }
            fusedGateUp = nil
            fusedGateUpAttempted = false
        }
        if !fusedGateUpAttempted {
            fusedGateUpAttempted = true
            fusedGateUp = FusedQuantizedProjection.fuse(sources)
        }
        return fusedGateUp
    }
}
