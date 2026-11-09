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
        let message: String?
        switch progress.event {
        case .text(let text), .reasoning(let text): message = text
        default: message = progress.decodedText == nil ? progress.message : nil
        }
        self.init(stage: stage, message: message)
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
