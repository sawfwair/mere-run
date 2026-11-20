import MereRunKVCache
import MLX

extension Q35LinearCache {
    /// Replays already computed recurrent inputs along the accepted parent path.
    /// No target projection or sibling state is read during commit.
    func restoreVerificationPath(_ path: [Int]) {
        #if os(macOS)
        if let nativeVerificationReplay {
            do { installNativeState(try nativeVerificationReplay.acceptedState(path: path)); return }
            catch { preconditionFailure("Native tree state commit failed: \(error)") }
        }
        #endif
        let replay = verificationReplay!
        let tree = verificationTree!
        precondition(path == tree.paths[path.last!])
        let qkv = Q35VerificationTree.take(replay.qkv, rows: path)
        let convolution = concatenated([replay.baseConvState, qkv], axis: 1)
        convState = convolution[0..., (convolution.dim(1) - replay.convKeep)..., 0...]
        let (_, state) = q35GatedDeltaUpdate(
            q: Q35VerificationTree.take(replay.q, rows: path),
            k: Q35VerificationTree.take(replay.k, rows: path),
            v: Q35VerificationTree.take(replay.v, rows: path),
            a: Q35VerificationTree.take(replay.a, rows: path),
            b: Q35VerificationTree.take(replay.b, rows: path),
            aLog: replay.aLog, dtBias: replay.dtBias, state: replay.baseRecurrentState,
            numKeyHeads: replay.numKeyHeads, numValueHeads: replay.numValueHeads,
            valueHeadDim: replay.valueHeadDim
        )
        recurrentState = state
        commitVerification()
    }
}

extension Q35LayerCache {
    var pendingTreeState: [MLXArray] {
        switch self {
        case .linear(let cache):
            #if os(macOS)
            // These accessors join and export the Metal commit. The next
            // verifier consumes its buffers directly once they are ready.
            if Q35NativeVerificationPolicy.prepareCommit, cache.nativeState != nil { return [] }
            #endif
            return [cache.convState, cache.recurrentState].compactMap { $0 }
        case .full(let cache as KVCacheSimple):
            let state = cache.verificationRows([cache.offset - 1])
            return [state.0, state.1]
        case .full: preconditionFailure("Tree verification requires ordinary KV storage")
        }
    }

    /// The receiver is the live committed cache; candidate is disposable.
    /// Publish only accepted rows, keeping caller-owned cache identities.
    package func commitVerificationTree(candidate: Q35LayerCache, path: [Int]) {
        switch (self, candidate) {
        case let (.linear(base), .linear(verified)):
            #if os(macOS)
            if let native = verified.nativeVerificationReplay {
                do {
                    let state = try native.acceptedState(path: path)
                    verified.installNativeState(state)
                    base.installNativeState(state)
                    return
                } catch { preconditionFailure("Native tree state commit failed: \(error)") }
            }
            #endif
            verified.restoreVerificationPath(path)
            base.convState = verified.convState
            base.recurrentState = verified.recurrentState
        case let (.full(base as KVCacheSimple), .full(verified as KVCacheSimple))
            where path.enumerated().allSatisfy({ $0.element == $0.offset }):
            base.adoptVerificationPrefix(from: verified, count: path.count)
        case let (.full(base), .full(verified as KVCacheSimple)):
            let indices = path.map { base.offset + $0 }
            let state = verified.verificationRows(indices)
            _ = base.update(keys: state.0, values: state.1)
        default: preconditionFailure("Tree cache families must match")
        }
    }
}
