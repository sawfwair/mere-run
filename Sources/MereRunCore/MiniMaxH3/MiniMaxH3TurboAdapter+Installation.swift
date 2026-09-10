import Foundation
import MLX
import MLXNN

extension MiniMaxH3TurboAdapter {
    @discardableResult
    static func install(
        url: URL,
        into transformer: MiniMaxH3Transformer,
        strength: Float,
        expectedPairCount: Int? = nil
    ) throws -> Int {
        try installForInference(
            url: url,
            into: transformer,
            strength: strength,
            adaLNCache: nil,
            expectedPairCount: expectedPairCount
        ).pairCount
    }

    static func installForInference(
        url: URL,
        into transformer: MiniMaxH3Transformer,
        strength: Float,
        adaLNCache: MiniMaxH3AdaLNCache?,
        expectedPairCount: Int? = nil
    ) throws -> Installation {
        let sourceFormat = try sourceFormat(at: url)
        if (sourceFormat == .fastVideo || sourceFormat == .fastH3Premerged), strength != 1 {
            throw AdapterError.requiresUnitStrength(strength)
        }
        if sourceFormat == .fastH3Premerged {
            let gateCount = try installFastH3QuantizedCompressionGates(
                url: url,
                into: transformer
            )
            guard gateCount == fastVideoExpectedCompressionGateCount else {
                throw AdapterError.unexpectedAuxiliaryTensorCount(
                    kind: "quantized-compression-gate",
                    expected: fastVideoExpectedCompressionGateCount,
                    actual: gateCount
                )
            }
            Memory.clearCache()
            return Installation(pairCount: 0, adaLNCache: adaLNCache)
        }
        let inferenceRecipe = inferenceRecipe(for: url)
        let suffixes = sourceFormat.pairSuffixes
        let usesNativeFastH3Cache = sourceFormat == .fastVideo
            && adaLNCache?.sourceIdentity == fastH3SourceIdentity
        if sourceFormat == .fastVideo {
            let parameterCount = try applyFastVideoDifferences(
                url: url,
                to: transformer,
                strength: strength,
                omittingCacheCoveredParameters: usesNativeFastH3Cache
            )
            guard parameterCount == fastVideoExpectedDiffCount else {
                throw AdapterError.unexpectedAuxiliaryTensorCount(
                    kind: "difference",
                    expected: fastVideoExpectedDiffCount,
                    actual: parameterCount
                )
            }
        }
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]
        var qkvPairs: [String: [QKVBranch: LoRAPair]] = [:]
        var adaLNPairs: [String: LoRAPair] = [:]

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: suffixes.0,
            secondSuffix: suffixes.1
        ) { sourcePath, rawDown, rawUp in
            let target = try target(for: sourcePath, sourceFormat: sourceFormat)
            guard rawDown.ndim == 2,
                  rawUp.ndim == 2,
                  rawUp.dim(1) == rawDown.dim(0) else {
                throw AdapterError.invalidPairShape(
                    sourcePath,
                    a: rawDown.shape,
                    b: rawUp.shape
                )
            }

            let up = scaledUp(
                rawUp,
                down: rawDown,
                sourceFormat: sourceFormat,
                lightX2VAlpha: inferenceRecipe.lightX2VAlpha
            )
            if isAdaLNTarget(target.modulePath) {
                if usesNativeFastH3Cache { return }
                guard adaLNPairs[target.modulePath] == nil else {
                    throw AdapterError.duplicateTarget(target.modulePath)
                }
                adaLNPairs[target.modulePath] = LoRAPair(down: rawDown, up: up)
                return
            }
            if let branch = target.qkvBranch {
                var branches = qkvPairs[target.modulePath] ?? [:]
                guard branches[branch] == nil else {
                    throw AdapterError.duplicateQKVBranch(target.modulePath, branch)
                }
                branches[branch] = LoRAPair(
                    down: rawDown,
                    up: up
                )
                if branches.count == QKVBranch.allCases.count {
                    guard replacements[target.modulePath] == nil else {
                        throw AdapterError.duplicateTarget(target.modulePath)
                    }
                    let linear = try targetLinear(at: target.modulePath, in: modulesByPath)
                    let query = branches[.query]!
                    let key = branches[.key]!
                    let value = branches[.value]!
                    try validateQKV(
                        target.modulePath,
                        query: query,
                        key: key,
                        value: value,
                        base: linear
                    )
                    replacements[target.modulePath] = runtimeQKVLayer(
                        base: linear,
                        query: query,
                        key: key,
                        value: value,
                        strength: strength
                    )
                    qkvPairs.removeValue(forKey: target.modulePath)
                } else {
                    qkvPairs[target.modulePath] = branches
                }
                return
            }

            guard replacements[target.modulePath] == nil else {
                throw AdapterError.duplicateTarget(target.modulePath)
            }
            let linear = try targetLinear(at: target.modulePath, in: modulesByPath)
            let mappedUp = target.modulePath.hasSuffix(".attn.qkv_proj")
                ? MiniMaxH3ModelLoader.deinterleavedQKVOutputRows(
                    up,
                    headCount: transformer.configuration.attentionHeadCount,
                    headDimension: transformer.configuration.attentionHeadDimension
                )
                : up
            try validate(
                target.modulePath,
                down: rawDown,
                up: mappedUp,
                base: linear
            )
            replacements[target.modulePath] = runtimeLayer(
                base: linear,
                down: rawDown,
                up: mappedUp,
                strength: strength
            )
        }

        guard pairCount > 0 else { throw AdapterError.noPairs(url) }
        let requiredPairCount = expectedPairCount ?? sourceFormat.expectedPairCount
        guard pairCount == requiredPairCount else {
            throw AdapterError.unexpectedPairCount(expected: requiredPairCount, actual: pairCount)
        }

        for (modulePath, branches) in qkvPairs {
            let missing = QKVBranch.allCases.filter { branches[$0] == nil }
            throw AdapterError.incompleteQKVTarget(modulePath, missing: missing)
        }

        var resolvedAdaLNCache = adaLNCache
        if !adaLNPairs.isEmpty {
            if let adaLNCache {
                resolvedAdaLNCache = try augmentedAdaLNCache(
                    adaLNCache,
                    pairs: adaLNPairs,
                    configuration: transformer.configuration,
                    strength: strength
                )
            } else {
                for (path, pair) in adaLNPairs {
                    guard replacements[path] == nil else {
                        throw AdapterError.duplicateTarget(path)
                    }
                    let linear = try targetLinear(at: path, in: modulesByPath)
                    try validate(path, down: pair.down, up: pair.up, base: linear)
                    replacements[path] = runtimeLayer(
                        base: linear,
                        down: pair.down,
                        up: pair.up,
                        strength: strength
                    )
                }
            }
        }

        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        if sourceFormat == .fastVideo {
            let gateCount = try installFastVideoCompressionGates(url: url, into: transformer)
            guard gateCount == fastVideoExpectedCompressionGateCount else {
                throw AdapterError.unexpectedAuxiliaryTensorCount(
                    kind: "compression-gate",
                    expected: fastVideoExpectedCompressionGateCount,
                    actual: gateCount
                )
            }
        }
        transformer.exactKernelMode = .disabled
        Memory.clearCache()
        return Installation(pairCount: pairCount, adaLNCache: resolvedAdaLNCache)
    }

}
