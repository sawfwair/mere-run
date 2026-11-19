import Foundation
import MLX

package enum Q35DFlash2Calibration {
    /// Prices disposable full-accept and rejected-prefix rounds without sampling.
    /// The resulting profile stays fixed for the lifetime of the loaded model.
    package static func measure(
        model: Q35Model, draft: Q35DFlash2Model, initialLogits: MLXArray,
        layerCaches: [Q35LayerCache?], draftCache: [Q35DFlash2Cache],
        checkCancellation: () throws -> Void = {}
    ) throws -> Q35DFlash2Costs {
        let requested = Q35DFlash2VerificationPolicy.proposalLimit(
            model: model, trainedBlockSize: draft.config.dflash.blockSize, sampled: false
        )
        let maximum = Q35DFlash2VerificationPolicy.limitForCaches(
            requested, trainedBlockSize: draft.config.dflash.blockSize, caches: layerCaches
        )
        let wide = maximum > draft.config.dflash.blockSize - 1
        let anchor = argMax(initialLogits[0, -1, 0...]).item(Int32.self)
        var full = Array(repeating: [Double](), count: maximum + 1)
        var rejected = full
        for sweep in 0..<4 {
            let depths = Array(full.indices)
            for depth in sweep.isMultiple(of: 2) ? depths : Array(depths.reversed()) {
                let modes = wide && depth > 0 ? (sweep.isMultiple(of: 2) ? [false, true] : [true, false]) : [false]
                for reject in modes {
                    try checkCancellation()
                    let seconds = price(model: model, draft: draft, anchor: anchor, depth: depth,
                                        reject: reject, maintainContext: wide, caches: layerCaches, context: draftCache)
                    if sweep > 0 {
                        if reject { rejected[depth].append(seconds) }
                        else { full[depth].append(seconds) }
                    }
                }
            }
        }
        let fullPrices = full.map { $0.sorted()[1] }
        let rejectedPrices = wide ? [fullPrices[0]] + rejected.dropFirst().map { $0.sorted()[1] } : nil
        return Q35DFlash2Costs(secondsByDepth: fullPrices, rejectedSecondsByDepth: rejectedPrices)
    }

    private static func price(
        model: Q35Model, draft: Q35DFlash2Model, anchor: Int32, depth: Int, reject: Bool, maintainContext: Bool,
        caches base: [Q35LayerCache?], context baseContext: [Q35DFlash2Cache]
    ) -> Double {
        let caches = base.map { $0?.fork() }
        let context = baseContext.map { $0.fork() }
        let capture = depth > 0 || maintainContext ? Set(draft.config.dflash.targetLayerIDs) : []
        let started = DispatchTime.now().uptimeNanoseconds
        let input: MLXArray
        if depth == 0 { input = MLXArray([anchor]).reshaped(1, 1) }
        else {
            let proposals = draft.draft(anchor: Int(anchor), count: depth, cache: context, target: model)
            input = concatenated([MLXArray([anchor]).reshaped(1, 1), proposals], axis: 1)
        }
        let output = model.forward(input, cache: caches, targetVerify: depth > 0, captureLayerIndices: capture)
        let picks = argMax(output.logits, axis: -1)
        MLX.eval(picks)
        _ = picks.asArray(Int32.self)
        if depth > 0 {
            let kept = reject ? 1 : depth + 1
            for cache in caches.compactMap({ $0 }) {
                if reject { precondition(cache.restoreVerificationPrefix(totalTokens: depth + 1, tokenCount: kept)) }
                cache.commitVerification()
            }
        }
        if depth > 0 || maintainContext {
            let kept = reject ? 1 : depth + 1
            draft.appendTargetContext(output.capturedHiddenStates.mapValues { $0[0..., 0..<kept, 0...] }, cache: context)
            let recurrent = caches.compactMap { entry -> Q35LinearCache? in
                if case .linear(let cache) = entry { return cache }
                return nil
            }.flatMap { [$0.convState, $0.recurrentState].compactMap { $0 } }
            MLX.eval(context.flatMap(\.state) + recurrent)
        }
        return Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
    }
}
