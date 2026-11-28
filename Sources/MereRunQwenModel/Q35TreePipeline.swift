import Foundation
import MereRunEnvironment
import MereRunKVCache
import MereRunTensor
import MLX

enum Q35TreePipeline {
    @TaskLocal static var enabled = MereRunEnvironmentSwitch.isEnabled("MERERUN_Q38_TREE_PIPELINED_KEEP", default: false)

    struct Match {
        let path: MLXArray
        let absolutePath: MLXArray
        let info: MLXArray
        let host: MLXArray
    }

    static func match(picks: MLXArray, proposal: MereRunTensor.DFlash2TreeProposal,
                      budget: Int, eos: Set<Int>, prefix: Int) -> Match {
        let result = Q35TreePipelineKernels.match(
            [picks, MLXArray(proposal.tokens.map(Int32.init)), MLXArray(proposal.parents.map(Int32.init)),
             MLXArray((eos.isEmpty ? [-1] : eos.sorted()).map(Int32.init)), MLXArray([Int32(budget), Int32(prefix)])],
            template: [("Rows", proposal.tokens.count), ("EOSCount", eos.count)],
            grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[proposal.tokens.count], [proposal.tokens.count], [2]],
            outputDTypes: [.int32, .int32, .int32])
        return Match(path: result[0], absolutePath: result[1], info: result[2],
                     host: concatenated([result[2], result[0]]))
    }

    enum Prepared {
        case linear(MLXArray, MLXArray)
        case full(original: KVCacheSimple, remapped: KVCacheSimple)

        var state: [MLXArray] {
            switch self {
            case let .linear(conv, recurrent): return [conv, recurrent]
            case .full: return []
            }
        }

        func publish(to base: Q35LayerCache, path: [Int]) {
            switch (self, base) {
            case let (.linear(conv, recurrent), .linear(cache)):
                cache.convState = conv; cache.recurrentState = recurrent
                cache.commitVerification()
            case let (.full(original, remapped), .full(cache as KVCacheSimple)):
                if path.enumerated().allSatisfy({ $0.element == $0.offset }) {
                    cache.adoptVerificationPrefix(from: original, count: path.count)
                } else {
                    MLX.asyncEval(remapped.pendingVerificationState)
                    cache.adoptVerificationPrefix(from: remapped, count: path.count)
                }
            default: preconditionFailure("Prepared and live tree caches must match")
            }
        }
    }

    /// Request-owned commit stream. Metadata and taps stay on the request's
    /// stream, so the next draft can run before target-only state work finishes.
    final class Frame {
        private let keepStream = Stream(Device.defaultDevice())

        func finish() { keepStream.synchronize() }

        func prepare(caches: [Q35LayerCache?], tree: Q35VerificationTree, match: Match,
                     prefix: Int) -> [Prepared] {
            let stream = StreamOrDevice.stream(keepStream)
            var indicesByCapacity: [Int: MLXArray] = [:]
            return caches.map { entry in
                switch entry! {
                case .linear(let cache):
                    let replay = cache.verificationReplay!
                    let gates = cache.verificationGates ?? (
                        g: Q35CompiledOperations.current.computeG(replay.aLog, replay.a, replay.dtBias),
                        beta: MLX.sigmoid(replay.b).asType(.float32))
                    let shape = [1, replay.numValueHeads, replay.valueHeadDim, replay.k.dim(3)]
                    let initial = replay.baseRecurrentState ?? MLXArray.zeros(shape, dtype: .float32)
                    let state = Q35TreePipelineKernels.recurrent(
                        [replay.k, replay.v, gates.g, gates.beta, initial, match.path, match.info],
                        template: [("Dk", replay.k.dim(3)), ("Dv", replay.valueHeadDim),
                                   ("Hk", replay.numKeyHeads), ("Hv", replay.numValueHeads)],
                        grid: (32, replay.valueHeadDim, replay.numValueHeads), threadGroup: (32, 4, 1),
                        outputShapes: [shape], outputDTypes: [.float32], stream: stream)[0]
                    let channels = replay.qkv.dim(2)
                    let conv = Q35TreePipelineKernels.convolution(
                        [replay.qkv, replay.baseConvState, match.path, match.info],
                        template: [("Channels", channels), ("Keep", replay.convKeep)],
                        grid: (channels * replay.convKeep, 1, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, replay.convKeep, channels]], outputDTypes: [replay.qkv.dtype], stream: stream)[0]
                    return .linear(conv, state)
                case .full(let cache as KVCacheSimple):
                    let capacity = cache.verificationCapacity
                    let indices: MLXArray
                    if let saved = indicesByCapacity[capacity] {
                        indices = saved
                    } else {
                        indices = concatenated([MLXArray((0..<prefix).map(Int32.init)), match.absolutePath,
                            MLXArray(((prefix + tree.count)..<capacity).map(Int32.init))], stream: stream)
                        indicesByCapacity[capacity] = indices
                    }
                    return .full(original: cache, remapped: cache.preparingVerificationRows(indices, stream: stream))
                case .full: preconditionFailure("Tree pipeline requires ordinary KV storage")
                }
            }
        }
    }
}
