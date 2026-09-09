import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    /// Measures target-only linear verification against an oracle sequence
    /// produced by the same loaded target. This benchmark-only API excludes
    /// draft cost and does not implement tree-shaped state.
    @_spi(Benchmark)
    public func benchmarkVerificationFrontier(
        messages: [ChatMessage],
        tokenCount: Int,
        widths: [Int],
        trials: Int
    ) throws -> [Q35VerificationFrontierResult] {
        precondition(tokenCount > 0 && !widths.isEmpty && trials > 0)
        guard let model, let tokenizerAndTemplate else {
            throw Q35Error.modelNotLoaded
        }
        let promptTokens = try tokenizerAndTemplate.encodeForGeneration(
            messages: messages,
            includeThinking: false,
            maxLength: Q35Resources.defaultContextLength
        )
        let promptInput = MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count)
        let serialCaches = makeLayerCaches(config: model.config)
        var serial = model.forward(promptInput, cache: serialCaches)
        MLX.eval(serial.logits)
        var oracleTokens: [Int] = []
        oracleTokens.reserveCapacity(tokenCount)
        for _ in 0..<tokenCount {
            let token = MLX.argMax(serial.logits[0, -1, 0...]).item(Int32.self)
            oracleTokens.append(Int(token))
            let input = MLXArray([token]).reshaped(1, 1)
            serial = model.forward(input, cache: serialCaches)
            MLX.eval(serial.logits)
        }

        var results: [Q35VerificationFrontierResult] = []
        for trial in 0..<max(1, trials) {
            let orderedWidths = trial.isMultiple(of: 2) ? widths : Array(widths.reversed())
            for width in orderedWidths {
                let effectiveWidth = max(1, min(width, tokenCount))
                let caches = makeLayerCaches(config: model.config)
                var output = model.forward(promptInput, cache: caches)
                MLX.eval(output.logits)
                var index = 0
                var passes = 0
                var seconds = 0.0
                var parity = MLX.argMax(output.logits[0, -1, 0...]).item(Int32.self)
                    == Int32(oracleTokens[0])
                while index < oracleTokens.count {
                    let end = min(index + effectiveWidth, oracleTokens.count)
                    let values = oracleTokens[index..<end].map(Int32.init)
                    let input = MLXArray(values).reshaped(1, values.count)
                    let started = Date()
                    output = model.forward(
                        input,
                        cache: caches,
                        targetVerify: values.count > 1
                    )
                    MLX.eval(output.logits)
                    seconds += Date().timeIntervalSince(started)
                    passes += 1

                    if values.count > 1 {
                        commitVerificationCaches(caches)
                    }
                    for offset in 0..<values.count where index + offset + 1 < oracleTokens.count {
                        let predicted = MLX.argMax(output.logits[0, offset, 0...]).item(Int32.self)
                        parity = parity && predicted == Int32(oracleTokens[index + offset + 1])
                    }
                    index = end
                }
                results.append(Q35VerificationFrontierResult(
                    width: effectiveWidth,
                    trial: trial,
                    verifiedTokens: oracleTokens.count,
                    verificationPasses: passes,
                    verificationSeconds: seconds,
                    verifiedTokensPerSecond: Double(oracleTokens.count) / seconds,
                    greedyOutputParity: parity,
                    activeMemoryBytes: Memory.activeMemory,
                    cacheMemoryBytes: Memory.cacheMemory
                ))
            }
        }
        return results
    }
}
