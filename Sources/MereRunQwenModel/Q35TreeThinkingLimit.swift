import Foundation

package struct Q35TreeThinkingLimit {
    private let closingToken: Int
    private let start: Int
    private let transition: [Int]

    package static func configured(
        tokenBudget: Int, requestedTokens: Int, automatic: Bool, mode: Q35SampledTreeMode?,
        environment: [String: String] = ProcessInfo.processInfo.environment, encode: (String) -> [Int]
    ) -> Self? {
        guard mode == .vocabulary else { return nil }
        let limit: Int
        if let raw = environment["MERERUN_Q38_THINKING_TOKEN_LIMIT"] {
            guard let value = Int(raw), value > 0, value < tokenBudget else { return nil }
            limit = value
        } else {
            guard automatic, requestedTokens == 16_384, tokenBudget == requestedTokens else { return nil }
            limit = 14_336
        }
        let close = encode("</think>")
        guard close.count == 1 else { return nil }
        return Self(limit: limit, tokenBudget: tokenBudget, closingToken: close[0],
                    cue: encode("\n\nI will now give my final answer.\n"), separator: encode("\n\n"))
    }

    init?(limit: Int, tokenBudget: Int, closingToken: Int, cue: [Int], separator: [Int]) {
        // Count the cue as thinking. Leave at least one position for a sampled answer.
        guard limit > 0, limit < tokenBudget, tokenBudget - limit > 1 + separator.count else { return nil }
        let cue = cue.count < limit ? cue : []
        self.closingToken = closingToken
        start = limit - cue.count
        transition = cue + [closingToken] + separator
    }

    func childBudget(remaining: Int, generated: Int) -> Int {
        min(remaining, max(0, start - generated))
    }

    func forcedToken(at index: Int) -> Int? {
        let offset = index - start
        return transition.indices.contains(offset) ? transition[offset] : nil
    }

    func finished(after token: Int, at index: Int) -> Bool {
        (index < start && token == closingToken) || index - start == transition.count - 1
    }
}
