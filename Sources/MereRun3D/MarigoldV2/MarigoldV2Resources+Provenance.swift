import Foundation
import MereRunCore

extension MarigoldV2Resources {
    /// Records the selected adapter file rather than attributing every local
    /// checkpoint to the managed Log-stage2 artifact.
    public func modelProvenance() throws -> GeometryModelProvenance {
        let digest = try ModelArtifactPin.fileSHA256(trainablesURL)
        let isPinnedCheckpoint = checkpoint == MarigoldV2Repository.installedCheckpoint
            && digest == MarigoldV2Repository.trainablesPin.sha256
        return GeometryModelProvenance(
            modelID: MarigoldV2Repository.modelId,
            upstreamRepository: MarigoldV2Repository.upstreamRepoId,
            upstreamRevision: isPinnedCheckpoint ? MarigoldV2Repository.upstreamRevision : "unverified-local",
            license: MarigoldV2Repository.license,
            weightsSHA256: digest
        )
    }
}
