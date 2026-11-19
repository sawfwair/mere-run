#if os(macOS)
import Foundation
import MereRunTensor
import MLX

package enum Q35NativeExecutionPolicy {
    package static let enabled = ProcessInfo.processInfo.environment["MERERUN_Q38_SWIFT_METAL"] == "1"
    static let promptRows: Int = {
        let rows = Int(ProcessInfo.processInfo.environment["MERERUN_Q38_SWIFT_METAL_PROMPT_ROWS"] ?? "")
        return rows.flatMap { [16, 64, 128, 256].contains($0) ? $0 : nil } ?? 128
    }()
}

@available(macOS 26.0, *)
package final class Q35NativeExecution {
    package enum Failure: LocalizedError {
        case incompatibleCheckpoint, missingAnchor
        case stage(String, any Error)

        public var errorDescription: String? {
            switch self {
            case .incompatibleCheckpoint: "The native Swift Qwen path requires the compatible 27B Q4 target and DFlash 2 draft."
            case .missingAnchor: "Native Qwen prefill could not select the first token."
            case .stage(let stage, let error): "Native Qwen \(stage) failed: \(error.localizedDescription)"
            }
        }
    }

    private let target: Q35NativeVerification.Session
    private let draft: Q35NativeDraftProgram.Session
    private let prompt: [Int]
    private let eos: Set<Int>
    private let budget: Int
    private let retainedStart: Int
    private var anchor: Int?

    package init(model: Q35Model, drafter: Q35DFlash2Model, prompt: [Int], budget: Int, eos: Set<Int>) throws {
        precondition(!prompt.isEmpty && budget > 0)
        let retainedStart = max(0, prompt.count - drafter.config.slidingWindow)
        let capture = Set(drafter.config.dflash.targetLayerIDs)
        guard let head = model.lmHead else { throw Failure.incompatibleCheckpoint }
        let target = try Self.perform("target setup") {
            guard let session = try model.model.nativeExecutionTarget(
                head: head, capacity: prompt.count + budget + 16, capture: capture
            ) else { throw Failure.incompatibleCheckpoint }
            return session
        }
        let draft = try Self.perform("draft setup") {
            guard let session = try drafter.nativeExecutionDraft(target: model, offset: retainedStart)
            else { throw Failure.incompatibleCheckpoint }
            return session
        }
        self.target = target
        self.draft = draft
        self.prompt = prompt
        self.budget = budget
        self.eos = eos
        self.retainedStart = retainedStart
    }

    private static func perform<Result>(_ stage: String, _ work: () throws -> Result) throws -> Result {
        do { return try work() }
        catch { throw Failure.stage(stage, error) }
    }

    /// A prompt chunk publishes every causal row. The draft only absorbs the
    /// suffix it can retain. Only the final prompt row reaches the vocabulary head.
    package func prefill(seedDraft: Bool = true, progress: (Int) -> Void) throws {
        let width = Q35NativeExecutionPolicy.promptRows
        for start in stride(from: 0, to: prompt.count, by: width) {
            try Task.checkCancellation()
            let end = min(start + width, prompt.count)
            let capture = seedDraft && end > retainedStart
            let chunk = try Self.perform("prefill rows \(start)..<\(end)") {
                try target.prefill(Array(prompt[start..<end]), capture: capture, score: end == prompt.count)
            }
            if capture {
                for first in stride(from: max(start, retainedStart), to: end, by: 16) {
                    let next = min(first + 16, end)
                    try draft.stageNative(chunk.taps, count: next - first, skipping: first - start)
                    if next < prompt.count { try draft.absorbNative() }
                }
            }
            anchor = chunk.following
            progress(end)
        }
        guard anchor != nil else { throw Failure.missingAnchor }
    }

    package func decode(drafting: Bool = true, emit: (Int) -> Void) throws -> Q35DFlash2Stats {
        guard var anchor else { throw Failure.missingAnchor }
        var stats = Q35DFlash2Stats()
        stats.treeRoundsByNodeCount = Array(repeating: 0, count: 16)
        var generated = 0
        var width = Q35DFlash2TreeCostPolicy()
        while generated < budget && !eos.contains(anchor) {
            try Task.checkCancellation()
            emit(anchor)
            generated += 1
            if generated == budget { break }
            if !drafting {
                let next = try target.prefill([anchor], capture: false, score: true)
                guard let following = next.following else { throw Failure.missingAnchor }
                anchor = following
                stats.targetOnlyRounds += 1
                continue
            }
            let remaining = budget - generated
            let adaptive = Q35DFlash2TreePolicy.adaptiveWidthEnabled && Q35DFlash2TreePolicy.maximumNodes == 15
            let limit = adaptive ? width.nodeLimit(remaining: remaining) : nil
            let dimensions = Q35DFlash2TreePolicy.proposalDimensions(remaining: remaining, nodeLimit: limit)
            let started = adaptive ? DispatchTime.now().uptimeNanoseconds : 0
            let proposal = try Self.perform("draft") {
                try draft.propose(anchor: anchor, depth: dimensions.depth, nodes: dimensions.nodes)
            }
            let result = try Self.perform("verification") {
                try target.verify(proposal, budget: remaining + 1, eos: eos)
            }
            anchor = result.following!
            try draft.stageNative(result.taps, count: result.count)
            for row in result.path.dropFirst() {
                emit(proposal.tokens[row])
                generated += 1
            }
            let proposed = proposal.tokens.count - 1
            stats.rounds += 1
            stats.treeRounds += 1
            stats.draftedTokens += proposed
            stats.acceptedTokens += result.count - 1
            stats.treeRoundsByNodeCount[proposed] += 1
            if adaptive, dimensions.depth == dimensions.nodes, proposed == dimensions.nodes {
                width.record(nodes: proposed, confirmedTokens: result.count,
                             seconds: Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9)
                stats.treeRoundSecondsByNodeCount = width.secondsByNodeCount
            }
        }
        try target.finish()
        return stats
    }
    package func scoreTeacher(tokens: [Int], labels: [Int]) throws -> [(loss: Double, top1: Int)] {
        precondition(tokens.count == labels.count && tokens.count <= budget)
        let observer = try target.makeTeacher()
        return try zip(tokens, labels).map { token, label in
            try Task.checkCancellation()
            let row = try target.prefill([token], capture: false, score: true, teacher: (observer, label))
            guard let loss = row.loss, loss.isFinite, let top1 = row.following else {
                throw Failure.missingAnchor
            }
            return (loss, top1)
        }
    }
}
#endif
