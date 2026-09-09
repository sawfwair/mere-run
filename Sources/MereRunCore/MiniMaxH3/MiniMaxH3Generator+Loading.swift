import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

extension MiniMaxH3Generator {
    func loadDenoisingRuntime(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        sequenceLength: Int,
        weightMode: MiniMaxH3TransformerWeightMode,
        adapterURL: URL?,
        adapterStrength: Float,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> (transformer: MiniMaxH3Transformer, adaLNCache: MiniMaxH3AdaLNCache?) {
        let modelSourceIdentity = try resources.adaLNCacheSourceIdentity()
        let adapterInferenceRecipe = adapterURL.map(MiniMaxH3TurboAdapter.inferenceRecipe(for:))
        let usesPremergedFastH3Student = adapterURL.map(
            MiniMaxH3TurboAdapter.isPremergedFastH3Artifact
        ) ?? false
        let cacheKey = DenoisingRuntimeCacheKey(
            modelRoot: resources.rootURL.resolvingSymlinksInPath(),
            modelSourceIdentity: modelSourceIdentity,
            videoSigmas: videoSchedule.sigmas,
            audioSigmas: audioSchedule.sigmas,
            weightMode: weightMode,
            adapterURL: adapterURL?.resolvingSymlinksInPath(),
            adapterSHA256: try adapterURL.map {
                try ModelArtifactPin.fileSHA256($0.resolvingSymlinksInPath())
            },
            adapterStrength: adapterStrength
        )
        if retainsRuntime,
           let retainedDenoisingRuntime,
           retainedDenoisingRuntime.key == cacheKey {
            return (
                retainedDenoisingRuntime.transformer,
                retainedDenoisingRuntime.adaLNCache
            )
        }
        let adaLNCache: MiniMaxH3AdaLNCache?
        let cachedAdaLNForLoading: MiniMaxH3AdaLNCache?
        if adapterInferenceRecipe?.requiresFastH3VSA == true,
           !resources.usesShardedBF16Transformer {
            guard try [.compactBF16, .affineQ8].contains(resources.transformerStorage()),
                  let adapterURL else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "FastH3 VSA requires the compact BF16, affine Q8, or full sharded BF16 MiniMax-H3 transformer"
                )
            }
            let cacheURL = adapterURL.deletingLastPathComponent().appending(
                path: MiniMaxH3TurboAdapter.fastH3AdaLNCacheFilename
            )
            guard FileManager.default.fileExists(atPath: cacheURL.path) else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "FastH3 VSA requires its source-bound AdaLN cache at \(cacheURL.path). "
                        + "Build it with scripts/model-conversion/prepare_minimax_h3_fasth3_vsa.py."
                )
            }
            let cache = try MiniMaxH3AdaLNCache.load(
                from: cacheURL,
                configuration: .init(configuration),
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: MiniMaxH3TurboAdapter.fastH3SourceIdentity
            )
            adaLNCache = cache
            cachedAdaLNForLoading = cache
        } else if resources.usesShardedBF16Transformer {
            // A legacy full BF16 root retains the schedule-only projections.
            // Build an exact table for the requested run; compact managed roots
            // select their source-bound production cache pack below.
            adaLNCache = nil
            cachedAdaLNForLoading = nil
        } else {
            let selection = try Self.loadAdaLNCache(
                resources: resources,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule
            )
            if selection?.exact == false {
                progressHandler?(.init(
                    stage: .interpolatingAdaLNCache,
                    stepIndex: 1,
                    totalSteps: 1
                ))
            }
            adaLNCache = selection?.cache
            cachedAdaLNForLoading = selection?.cache
        }
        let transformer = try MiniMaxH3ModelLoader.loadInferenceTransformer(
            resources: resources,
            configuration: configuration,
            cachedAdaLN: cachedAdaLNForLoading,
            progressHandler: { shard in
                progressHandler?(.init(
                    stage: .loadingTransformer,
                    stepIndex: shard.shardIndex,
                    totalSteps: shard.shardCount
                ))
            }
        )
        if let adapterURL, resources.usesShardedBF16Transformer {
            try MiniMaxH3TurboAdapter.install(
                url: adapterURL,
                into: transformer,
                strength: adapterStrength
            )
        }
        var resolvedAdaLNCache: MiniMaxH3AdaLNCache?
        if resources.usesShardedBF16Transformer {
            resolvedAdaLNCache = transformer.precomputeAdaLN(
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: modelSourceIdentity
            )
            transformer.discardAdaLNWeights()
            Memory.clearCache()
        } else {
            resolvedAdaLNCache = adaLNCache
        }
        let effectiveWeightMode: MiniMaxH3TransformerWeightMode =
            usesPremergedFastH3Student && weightMode == .automatic ? .quantized : weightMode
        let shouldMaterialize = try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: effectiveWeightMode,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            estimatedResidentBytes: transformer.estimatedResidentBF16ByteCount,
            sequenceLength: sequenceLength,
            hasAdaLNCache: resolvedAdaLNCache != nil,
            isPortableMac: MiniMaxH3Host.isPortableMac
        )
        if shouldMaterialize {
            progressHandler?(.init(
                stage: .materializingTransformerBF16,
                stepIndex: 0,
                totalSteps: 1
            ))
            _ = transformer.materializeResidentBF16()
        }
        if let adapterURL, !resources.usesShardedBF16Transformer {
            let adapterTask = adapterInferenceRecipe?.task ?? .fl2va
            if adapterTask == .fl2va,
               try !resources.transformerStorage().supportsFL2VATurboAdapters {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "MiniMax-H3 FL2VA Turbo requires compact BF16 or Q8; legacy Q4 is unsupported"
                )
            }
            if adapterTask == .ref2va, !transformer.usesResidentBF16 {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "MiniMax-H3 Ref2VA Turbo requires resident BF16 weights; use --h3-weight-mode resident-bf16 on a machine with sufficient memory"
                )
            }
            let installation = try MiniMaxH3TurboAdapter.installForInference(
                url: adapterURL,
                into: transformer,
                strength: adapterStrength,
                adaLNCache: resolvedAdaLNCache
            )
            resolvedAdaLNCache = installation.adaLNCache
        }
        if retainsRuntime {
            retainedDenoisingRuntime = (cacheKey, transformer, resolvedAdaLNCache)
        }
        return (transformer, resolvedAdaLNCache)
    }

    func loadConditioner(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> QwenVLEncoder {
        let root = resources.rootURL.resolvingSymlinksInPath()
        if retainsRuntime, let retainedConditioner, retainedConditioner.root == root {
            return retainedConditioner.model
        }
        let model = try MiniMaxH3ModelLoader.loadConditioner(
            resources: resources,
            configuration: configuration,
            progressHandler: progressHandler
        )
        if retainsRuntime { retainedConditioner = (root, model) }
        return model
    }

    func loadVideoVAE(resources: MiniMaxH3Resources) throws -> MiniMaxH3VideoVAE {
        let root = resources.rootURL.resolvingSymlinksInPath()
        if retainsRuntime, let retainedVideoVAE, retainedVideoVAE.root == root {
            return retainedVideoVAE.model
        }
        let model = try MiniMaxH3ModelLoader.loadVideoVAE(resources: resources)
        if retainsRuntime { retainedVideoVAE = (root, model) }
        return model
    }

    func loadAudioVAE(resources: MiniMaxH3Resources) throws -> MiniMaxH3AudioVAE {
        let root = resources.rootURL.resolvingSymlinksInPath()
        if retainsRuntime, let retainedAudioVAE, retainedAudioVAE.root == root {
            return retainedAudioVAE.model
        }
        let model = try MiniMaxH3ModelLoader.loadAudioVAE(resources: resources)
        if retainsRuntime { retainedAudioVAE = (root, model) }
        return model
    }

}
