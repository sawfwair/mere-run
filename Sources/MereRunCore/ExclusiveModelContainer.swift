import Foundation

/// Actor wrapper that guarantees single-threaded access to a non-Sendable model object.
///
/// This is the same high-level pattern used in `mlx-swift-examples`, adapted for Swift 6:
/// return values must be `Sendable`, so callers cannot accidentally leak `MLXArray` across
/// concurrency boundaries.
public actor ExclusiveModelContainer<Model> {
    public enum ContainerError: LocalizedError {
        case modelDiscarded

        public var errorDescription: String? {
            switch self {
            case .modelDiscarded:
                return "Model was discarded to conserve memory."
            }
        }
    }

    public enum State {
        case discarded
        case loaded(Model)
    }

    private var state: State
    private var conserveMemory: Bool = false

    public init(model: Model) {
        self.state = .loaded(model)
    }

    public func setConserveMemory(_ enabled: Bool) {
        conserveMemory = enabled
    }

    public func discard() {
        state = .discarded
    }

    public func withModel<R: Sendable>(_ body: @Sendable (Model) throws -> R) throws -> R {
        switch state {
        case .discarded:
            throw ContainerError.modelDiscarded
        case .loaded(let model):
            return try body(model)
        }
    }

    public func withModelVoid(_ body: @Sendable (Model) throws -> Void) throws {
        try withModel { model in
            try body(model)
            return ()
        }
    }

    /// Runs a two-stage operation, optionally discarding the model between stages.
    ///
    /// `R1` never crosses an actor boundary; only `R2` is returned (and must be `Sendable`).
    public func withTwoStage<R1, R2: Sendable>(
        first: @Sendable (Model) throws -> R1,
        second: @Sendable (R1) throws -> R2
    ) throws -> R2 {
        let r1: R1
        switch state {
        case .discarded:
            throw ContainerError.modelDiscarded
        case .loaded(let model):
            r1 = try first(model)
        }

        if conserveMemory {
            state = .discarded
        }

        return try second(r1)
    }
}

