import AudioParakeetModel
import Foundation
import MLX
import MLXNN
import AudioCore
import AudioCodecs
import MereRunCore

extension ParakeetGenerator {
    func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: ParakeetResources.defaultModelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(
                            ASRProgress(stage: .loadingModel, message: "Downloading model... \(percent)%")
                        )
                    case .extracting:
                        progressHandler?(
                            ASRProgress(stage: .loadingModel, message: "Extracting model...")
                        )
                    }
                }
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw ParakeetError.downloadFailed(error.localizedDescription)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> ParakeetError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    func loadModel(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws {
        try await Stream.withNewDefaultStream(isolation: self) {
            try loadModelOnTaskSafeStream(
                from: rootURL,
                progressHandler: progressHandler
            )
        }
    }

    /// Model construction and weight evaluation create lazy MLX arrays that retain their
    /// originating stream. The live session decodes on a later cooperative-executor thread,
    /// so preparation must use MLX's cross-thread streams too—not only the decode call.
    private func loadModelOnTaskSafeStream(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws {
        let resources = ParakeetResources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ParakeetError.missingFiles(missing.map(\.lastPathComponent))
        }

        let config = try ParakeetModelConfig.load(from: resources.configURL)
        let includeMLXEncoder: Bool
        switch executionProvider {
        case .mlx:
            guard config.packaging == .completeMLX else {
                throw ParakeetError.coreMLProviderRequired
            }
            includeMLXEncoder = true
        case .coreML:
            includeMLXEncoder = false
        }
        let model = ParakeetModelFactory.build(
            config: config,
            includeMLXEncoder: includeMLXEncoder
        )

        switch executionProvider {
        case .mlx:
            break
        case .coreML(let artifactURL):
            guard let baseModel = model as? ParakeetBaseModel,
                  config.variant == .tdt || config.variant == .tdtCTC else {
                throw ParakeetError.unsupportedCoreMLVariant(config.variant.rawValue)
            }
            if config.packaging == .coreMLHybrid,
               rootURL.resolvingSymlinksInPath().standardizedFileURL
               != artifactURL.resolvingSymlinksInPath().standardizedFileURL {
                throw ParakeetError.coreMLHybridArtifactMismatch
            }
            let loadedArtifact = try ParakeetCoreMLManifest.load(
                artifactURL: artifactURL,
                config: config
            )
            baseModel.externalEncoder = try ParakeetCoreMLEncoder(
                manifest: loadedArtifact.manifest,
                modelURL: loadedArtifact.modelURL
            )
            if loadedArtifact.manifest.coreMLDecoder != nil,
               let tdtModel = baseModel as? ParakeetTDTModel {
                tdtModel.externalDecoder = try ParakeetCoreMLDecoder(
                    artifactURL: artifactURL,
                    manifest: loadedArtifact.manifest,
                    config: config
                )
            }
        }

        guard let module = model as? Module else {
            throw ParakeetError.modelNotLoaded
        }

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet weights..."))

        let quantization: ModelWeightsLoader.QuantizationParams?
        if let bits = config.quantizationBits, let group = config.quantizationGroupSize {
            quantization = .init(bits: bits, groupSize: group)
        } else {
            quantization = nil
        }

        let mapper: (String, MLXArray) -> [(String, MLXArray)]
        let verification: Module.VerifyUpdate
        switch executionProvider {
        case .mlx:
            mapper = { key, value in [(key, value)] }
            verification = .noUnusedKeys
        case .coreML:
            mapper = { key, value in
                key.hasPrefix("encoder.") ? [] : [(key, value)]
            }
            verification = .all
        }

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.modelIndexURL,
            singleURL: resources.modelWeightsURL,
            to: module,
            dtype: .bfloat16,
            verify: verification,
            mapper: mapper,
            quantization: quantization
        )

        // Parakeet uses BatchNorm/Dropout in the encoder stack; keep inference in eval mode.
        module.train(false)

        MLX.eval(module.parameters())

        self.model = model
        self.modelConfig = config
        self.audioPreprocessor = try ParakeetAudioPreprocessor(config: config.preprocessor)
        self.loadedModelPath = rootURL.path
    }
}
