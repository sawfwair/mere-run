import Foundation
import MereRunKVCache
import MereRunTensor

/// Widens only the eligible target's verification scope. Serial projections
/// retain their eight-row arithmetic and draft modules keep their own policy.
package enum Q35DFlash2VerificationPolicy {
    private static let configuredRows = ProcessInfo.processInfo.environment["MERERUN_Q38_M5_VERIFICATION_ROWS"]
        .map { $0 == "16" ? 16 : 8 }
    @TaskLocal package static var maximumRowsOverride: Int?
    package static var maximumRows: Int { maximumRowsOverride ?? configuredRows ?? (QwenDFlashDefaults.active ? 16 : 8) }
    @TaskLocal package static var requestedProposalDepth: Int? =
        ProcessInfo.processInfo.environment["MERERUN_Q38_DFLASH2_PROPOSAL_DEPTH"] == "15" ? 15 : nil

    package static func rowLimit(config: Q35Config, backend: QuantizedVerificationBackend?) -> Int {
        let text = config.textConfig
        guard backend == .m5Matrix, QuantizedVerificationBackend.m5Supported,
              PortableQuantizedLinear.m5TiledWeightsEnabled,
              !text.usesMoE, !text.isQwen4Exp, text.hiddenSize == 5120,
              text.intermediateSize == 17_408, text.numHiddenLayers == 64,
              config.quantization?.bits == 4, config.quantization?.groupSize == 64 else { return 8 }
        return maximumRows
    }

    package static func proposalLimit(trainedBlockSize: Int, verificationRows: Int, sampled: Bool) -> Int {
        guard !sampled, verificationRows == 16, requestedProposalDepth == 15 else { return trainedBlockSize - 1 }
        return 15
    }

    package static func proposalLimit(model: Q35Model, trainedBlockSize: Int, sampled: Bool) -> Int {
        proposalLimit(trainedBlockSize: trainedBlockSize, verificationRows: rowLimit(config: model.config,
            backend: (model.lmHead as? PortableQuantizedLinear)?.effectiveVerificationBackend), sampled: sampled)
    }

    /// Wide checkpoint parity covers prefix-restorable default caches. Other
    /// KV formats retain the trained block and its existing recovery path.
    package static func limitForCaches(_ maximum: Int, trainedBlockSize: Int, caches: [Q35LayerCache?]) -> Int {
        let needsRecovery = caches.contains { entry in
            if case .full(let cache) = entry { return !(cache is KVCacheSimple) }
            return false
        }
        return needsRecovery ? min(maximum, trainedBlockSize - 1) : maximum
    }
}
