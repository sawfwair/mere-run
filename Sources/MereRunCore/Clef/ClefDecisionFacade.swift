import Foundation

/// Select the checkpoint's native thinker; dense Clef and Omni retain distinct media/attention policies.
public final class ClefDecisionOperation {
    private enum Runtime {
        case dense(ClefDenseDecisionOperation)
        case omni(ClefOmniDecisionOperation)
    }
    private let runtime: Runtime
    public let modelID: String

    public init(root: URL, modelID: String = ClefCatalog.modelID) throws {
        self.modelID = modelID
        if try ClefCatalog.isOmni(root: root) {
            runtime = .omni(try ClefOmniDecisionOperation(root: root, modelID: modelID))
        } else { runtime = .dense(try ClefDenseDecisionOperation(root: root, modelID: modelID)) }
    }

    public func prepare(_ request: ClefDecisionRequest) throws -> ClefDecisionPlan {
        switch runtime {
        case .dense(let operation): try operation.prepare(request)
        case .omni(let operation): try operation.prepare(request)
        }
    }

    public func predict(_ request: ClefDecisionRequest) throws -> ClefDecisionResponse {
        switch runtime {
        case .dense(let operation): try operation.predict(request)
        case .omni(let operation): try operation.predict(request)
        }
    }

    public func unload() {
        switch runtime {
        case .dense(let operation): operation.unload()
        case .omni(let operation): operation.unload()
        }
    }
}
