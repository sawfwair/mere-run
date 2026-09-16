import ArgumentParser
import Foundation
import MereRunCore

extension YuE2PlanningMode: ExpressibleByArgument {}

struct YuE2CLIOptions: ParsableArguments {
    @Option(name: .customLong("score-mode"), help: "YuE2 ABC planning: full (default), melody, or off.")
    var planning: YuE2PlanningMode?

    @Option(name: .customLong("abc-file"), help: "YuE2 input ABC score; skips score generation. Requires full or melody mode.")
    var abcFile: String?

    @Option(name: .customLong("abc-output"), help: "YuE2 score output (default: <output>.abc).")
    var abcOutput: String?

    @Option(name: .customLong("abc-max-tokens"), help: "YuE2 score token budget (default: 4096).")
    var abcMaximumTokens: Int?

    @Option(name: .customLong("semantic-temperature"), help: "YuE2 music sampling temperature in 0...5 (default: 1).")
    var temperature: Float?

    @Option(name: .customLong("semantic-top-p"), help: "YuE2 music nucleus sampling in (0, 1] (default: 0.95).")
    var topP: Float?

    @Option(name: .customLong("semantic-top-k"), help: "YuE2 music top-k sampling (default: 100).")
    var topK: Int?

    @Option(name: .customLong("semantic-repetition-penalty"), help: "YuE2 frequency penalty over the last 50 music tokens (default: 1.2).")
    var repetitionPenalty: Float?

    var isSpecified: Bool {
        planning != nil || abcFile != nil || abcOutput != nil || abcMaximumTokens != nil
            || temperature != nil || topP != nil || topK != nil || repetitionPenalty != nil
    }
}
