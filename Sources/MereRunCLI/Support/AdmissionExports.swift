import Foundation
@_exported import MereRunAdmission
import MereRunCore

extension RuntimeRequestProgress {
    init(_ progress: ChatProgress) {
        let stage: Stage
        switch progress.stage {
        case .loadingModel: stage = .loadingModel
        case .encoding: stage = .encoding
        case .generating: stage = .generating
        }
        self.init(stage: stage, message: progress.message)
    }
}

extension RuntimeRequestAdmissionLease {
    func observe(_ progress: ChatProgress) {
        observe(RuntimeRequestProgress(progress))
    }
}

extension RuntimeRequestAdmission {
    func recordProgress(id: UUID, sequence: UInt64, progress: ChatProgress, observedAt: Date = Date()) {
        recordProgress(id: id, sequence: sequence, progress: RuntimeRequestProgress(progress), observedAt: observedAt)
    }
}
