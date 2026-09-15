import Foundation

/// Applies the request's thinking visibility while preserving token-boundary fragments.
final class LFM2GenerationStream: @unchecked Sendable {
    private let lock = NSLock()
    private let showThinking: Bool
    private let handler: (@Sendable (ChatProgress) -> Void)?
    private var pending = ""
    private var insideThinking = false

    init(showThinking: Bool, handler: (@Sendable (ChatProgress) -> Void)?) {
        self.showThinking = showThinking
        self.handler = handler
    }

    func accept(_ progress: ChatProgress) {
        guard progress.stage == .generating, let text = progress.message else {
            handler?(progress)
            return
        }
        lock.withLock {
            if showThinking {
                handler?(progress)
                return
            }
            pending += text
            var visible = ""
            while !pending.isEmpty {
                let tag = insideThinking ? "</think>" : "<think>"
                if let range = pending.range(of: tag, options: .caseInsensitive) {
                    if !insideThinking { visible += pending[..<range.lowerBound] }
                    pending.removeSubrange(..<range.upperBound)
                    insideThinking.toggle()
                    continue
                }
                let retained = (1..<tag.count).reversed().first {
                    pending.lowercased().hasSuffix(tag.prefix($0))
                } ?? 0
                let ready = pending.dropLast(retained)
                if !insideThinking { visible += ready }
                pending = String(pending.suffix(retained))
                break
            }
            if !visible.isEmpty { handler?(ChatProgress(stage: .generating, message: visible)) }
        }
    }

    func finish() {
        lock.withLock {
            if !insideThinking, !pending.isEmpty {
                handler?(ChatProgress(stage: .generating, message: pending))
            }
            pending = ""
        }
    }
}
