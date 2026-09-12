import Foundation

public extension ManagedModelSpec {
    var usesPinnedGeometryArtifacts: Bool {
        switch validationKind {
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR, .instantMesh, .trellis2:
            true
        default:
            false
        }
    }

    var requiresManagedConversion: Bool {
        switch validationKind {
        case .instantMesh, .terramindFlood, .terramindFire, .tessera, .olmoEarth:
            true
        default:
            false
        }
    }

    func managedConversionGuidance(at rootURL: URL) -> String? {
        guard requiresManagedConversion else { return nil }
        if validationKind == .terramindFlood {
            let source = rootURL.appendingPathComponent(TerraMindFloodResources.sourceCheckpointFilename).path
            let configuration = rootURL.appendingPathComponent(
                TerraMindFloodResources.sourceConfigurationFilename
            ).path
            return "Pinned TerraMind Flood source downloaded at \(source). Deterministic conversion is required "
                + "before native MLX inference; run scripts/convert-terramind-flood-mlx.py "
                + "--checkpoint \"\(source)\" --configuration \"\(configuration)\" "
                + "--output \"\(rootURL.path)\" --dtype float32. FP16 is intentionally unsupported by parity evidence."
        }
        if validationKind == .terramindFire {
            let source = rootURL.appendingPathComponent(TerraMindFireResources.sourceCheckpointFilename).path
            let configuration = rootURL.appendingPathComponent(
                TerraMindFireResources.sourceConfigurationFilename
            ).path
            return "Pinned TerraMind Fire source downloaded at \(source). Deterministic conversion is required "
                + "before native MLX inference; run scripts/convert-terramind-fire-mlx.py "
                + "--checkpoint \"\(source)\" --configuration \"\(configuration)\" "
                + "--output \"\(rootURL.path)\" --dtype float32."
        }
        if validationKind == .tessera, let source = TESSERAResources.spec(for: id) {
            let checkpoint = rootURL.appendingPathComponent(source.sourceCheckpointFilename).path
            return "Pinned TESSERA v2 \(source.variant.rawValue) source downloaded at \(checkpoint). "
                + "Deterministic conversion is required before native MLX inference; run "
                + "scripts/convert-tessera-v2-mlx.py --variant \(source.variant.rawValue) "
                + "--checkpoint \"\(checkpoint)\" --output \"\(rootURL.path)\" --dtype float32."
        }
        if validationKind == .olmoEarth, let source = OlmoEarthResources.spec(for: id) {
            let weights = rootURL.appendingPathComponent(OlmoEarthResources.sourceWeightsFilename).path
            let configuration = rootURL.appendingPathComponent(
                OlmoEarthResources.sourceConfigurationFilename
            ).path
            return "Pinned OlmoEarth v1.2 \(source.variant.rawValue) source downloaded at \(weights). "
                + "Deterministic conversion is required before native MLX inference; run "
                + "scripts/convert-olmoearth-v12-mlx.py --variant \(source.variant.rawValue) "
                + "--weights \"\(weights)\" --configuration \"\(configuration)\" "
                + "--output \"\(rootURL.path)\" --dtype float32."
        }
        let source = rootURL.appendingPathComponent("instant_mesh_base.ckpt").path
        let output = rootURL.appendingPathComponent(
            InstantMeshResources.managedConvertedDirectoryName,
            isDirectory: true
        ).path
        let license = rootURL.appendingPathComponent("LICENSE").path
        return "Pinned InstantMesh source downloaded at \(source). Conversion is required before runtime; "
            + "run scripts/model-conversion/convert_instantmesh_base.py "
            + "--source \"\(source)\" --output \"\(output)\" --license-file \"\(license)\"."
    }

    func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let base = rootURL.resolvingSymlinksInPath()
        if validationKind == .lfm2 {
            return LFM2Resources.normalizedRootURL(base, fileManager: fileManager)
        }
        switch normalizationKind {
        case .none, .musicACEStep:
            return base
        case .musicACEStepLM:
            if ACEStep5HzLMResources(rootURL: base).validate(fileManager: fileManager).isEmpty {
                return base
            }
            let candidates = [
                "acestep-5Hz-lm-1.7B",
                "acestep-5hz-lm-1.7b",
                "acestep-5Hz-lm-4B",
                "acestep-5hz-lm-4b",
            ]
            return candidates
                .map { base.appendingPathComponent($0, isDirectory: true) }
                .first {
                    ACEStep5HzLMResources(rootURL: $0).validate(fileManager: fileManager).isEmpty
                } ?? base
        case .qwen3ASRNested:
            let direct = missingPathsWithoutNormalization(in: base, fileManager: fileManager)
            if direct.isEmpty {
                return base
            }
            let nested = base.appendingPathComponent(id, isDirectory: true)
            return missingPathsWithoutNormalization(in: nested, fileManager: fileManager).isEmpty ? nested : base
        case .parakeetNested:
            let direct = missingPathsWithoutNormalization(in: base, fileManager: fileManager)
            if direct.isEmpty {
                return base
            }
            let nested = base.appendingPathComponent(id, isDirectory: true)
            return missingPathsWithoutNormalization(in: nested, fileManager: fileManager).isEmpty ? nested : base
        }
    }

    func missingPaths(in rootURL: URL, fileManager: FileManager = .default) -> [URL] {
        missingPathsWithoutNormalization(
            in: normalizedRootURL(rootURL, fileManager: fileManager),
            fileManager: fileManager
        )
    }

    private func missingPathsWithoutNormalization(in rootURL: URL, fileManager: FileManager = .default) -> [URL] {
        switch validationKind {
        case .flux1:
            return Flux1Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .flux2Klein:
            return Self.missingDiffusersImagePaths(in: rootURL, fileManager: fileManager)
        case .bonsaiImage:
            return Self.missingBonsaiImagePaths(in: rootURL, fileManager: fileManager)
        case .zimageTurbo:
            return ZImageTurboResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .hidreamO1:
            return HiDreamO1Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .senseNovaU15:
            return SenseNovaU15Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .krea2:
            return Krea2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .qwenImageEdit:
            var missing = QwenImageEditResources(rootURL: rootURL).validate(fileManager: fileManager)
            if id == QwenImageEditRepository.lightning2511Id {
                let adapter = rootURL.appendingPathComponent(QwenImageEditRepository.lightningRelativePath)
                if !fileManager.fileExists(atPath: adapter.path) {
                    missing.append(adapter)
                }
            }
            return missing
        case .ideogram4SDNQ:
            return Ideogram4Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .gemma4:
            return Gemma4Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .diffusionGemma:
            return DiffusionGemmaResources(rootURL: rootURL).validate(
                fileManager: fileManager,
                requireVision: true
            )
        case .gemma4Unified:
            return Gemma4Resources(rootURL: rootURL).validate(
                fileManager: fileManager,
                requireUnifiedProcessor: true
            )
        case .gemma4MTPAssistant:
            return Gemma4MTPResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .laguna:
            return LagunaResources.missingTargetFiles(rootURL: rootURL, fileManager: fileManager)
        case .lagunaDFlash:
            return LagunaResources.missingDFlashFiles(rootURL: rootURL, fileManager: fileManager)
        case .q35:
            let resources = Q35Resources(rootURL: rootURL)
            var missing = resources.validate(fileManager: fileManager)
            if id == Q35Resources.q38TwentySevenB4BitModelId {
                missing.append(contentsOf: resources.validateQ38MTPComponent(fileManager: fileManager))
                missing.append(contentsOf: resources.validateQ38VisionComponent(fileManager: fileManager))
            }
            if id == Q35Resources.q38FlashNext3BitNativePLEModelId {
                for filename in [Q38PLEPlacement.manifestFilename, "MERERUN_PLE_PACK.json"] {
                    let url = rootURL.appendingPathComponent(filename)
                    if !fileManager.fileExists(atPath: url.path) {
                        missing.append(url)
                    }
                }
            }
            if id == Q35Resources.ornith35BMLX4BitModelId {
                missing.append(contentsOf: resources.validateOrnithVisionComponent(fileManager: fileManager))
                missing.append(contentsOf: resources.validateOrnithMTPComponent(fileManager: fileManager))
            }
            return missing
        case .q35MTPAssistant:
            return Q35Resources(rootURL: rootURL).validateOrnith35BMTPCompanion(
                fileManager: fileManager
            )
        case .lfm2:
            return LFM2Resources(rootURL: rootURL).validate(
                fileManager: fileManager,
                requireVisionProcessor: id == LFM2Resources.visionModelId
            )
        case .lfm2DSpark:
            return LFM2Resources.missingDSparkFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .inkling:
            return InklingResources.validate(rootURL: rootURL, fileManager: fileManager)
        case .museGlimmer:
            return MuseGlimmerResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .museGlimmerAssistant:
            return MuseGlimmerResources.validateAssistant(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .nemotronH:
            return NemotronHResources.missingTargetFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .nemotronHDSpark:
            return NemotronHResources.missingDSparkFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .nemotronOmni:
            return NemotronOmniResources.missingTargetFiles(
                rootURL: rootURL,
                fileManager: fileManager
            )
        case .sam31:
            return SAM31Resources(modelRootURL: rootURL).missingRequiredPaths(fileManager: fileManager)
        case .falconPerception:
            return FalconPerceptionResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .terramindFlood:
            return TerraMindFloodResources.missingSourcePaths(in: rootURL, fileManager: fileManager)
        case .terramindFire:
            return TerraMindFireResources.missingSourcePaths(in: rootURL, fileManager: fileManager)
        case .tessera:
            return TESSERAResources.missingSourcePaths(for: id, in: rootURL, fileManager: fileManager)
        case .olmoEarth:
            return OlmoEarthResources.missingSourcePaths(in: rootURL, fileManager: fileManager)
        case .insightFaceBuffaloL:
            return FaceAnalysisResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR:
            guard let pin = GeometryModelPins.pin(for: id) else { return [rootURL] }
            return Self.invalidPinnedArtifacts(pin.runtimeArtifacts, in: rootURL, fileManager: fileManager)
        case .instantMesh:
            if Self.invalidInstantMeshNativeArtifacts(in: rootURL, fileManager: fileManager).isEmpty {
                return []
            }
            guard let pin = GeometryModelPins.pin(for: id) else { return [rootURL] }
            return Self.invalidPinnedArtifacts(pin.artifacts, in: rootURL, fileManager: fileManager)
        case .trellis2:
            return Trellis2Resources.validate(rootURL: rootURL, fileManager: fileManager)
        case .qwen3TTS:
            return Self.missingQwen3TTSPaths(in: rootURL, fileManager: fileManager)
        case .qwen3ASR:
            return Self.missingQwen3ASRPaths(in: rootURL, fileManager: fileManager)
        case .parakeet:
            return Self.missingParakeetPaths(in: rootURL, fileManager: fileManager)
        case .sortformer:
            return Self.missingSortformerPaths(in: rootURL, fileManager: fileManager)
        case .qwen3Embedding:
            return Qwen3EmbeddingResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .qwen3VLEmbedding:
            return Qwen3VLEmbeddingResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .privacyFilter:
            return OpenAIPrivacyFilterResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .codegenGGUF:
            return Self.missingCodeGenPaths(
                preferredRelativePath: hubFallback?.filePath,
                in: rootURL,
                fileManager: fileManager
            )
        case .deepseekV4FlashIMatrixGGUF:
            return Self.missingDeepseekV4FlashIMatrixPaths(in: rootURL, fileManager: fileManager)
        case .lightOnOCR:
            return Self.missingLightOnOCRPaths(in: rootURL, fileManager: fileManager)
        case .aceStep:
            return Self.missingACEStepPaths(modelID: id, in: rootURL, fileManager: fileManager)
        case .aceStepLM:
            return ACEStep5HzLMResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .miniMaxMusic3:
            return MiniMaxMusic3Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .magentaRT2:
            return Self.missingMagentaRT2Paths(modelID: id, in: rootURL, fileManager: fileManager)
        case .muScriptor:
            return MuScriptorResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .roFormer:
            if let resources = try? RoFormerResources(rootURL: rootURL, modelID: id) {
                return resources.validate(fileManager: fileManager)
            }
            if let resources = try? MelBandRoFormerResources(rootURL: rootURL, modelID: id) {
                return resources.validate(fileManager: fileManager)
            }
            return [rootURL]
        case .apBWE:
            return APBWEResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .univerSR:
            return UniverSRResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .woosh:
            let checkpointsRoot = WooshResources.normalizeRoot(rootURL, fileManager: fileManager)
            let variant = WooshVariant.resolve(model: id, rootURL: checkpointsRoot, fileManager: fileManager) ?? .dflow
            return WooshModelResources(checkpointsRootURL: checkpointsRoot, variant: variant)
                .missingFiles(fileManager: fileManager)
        case .wooshClap:
            let checkpointsRoot = WooshResources.normalizeRoot(rootURL, fileManager: fileManager)
            return WooshCLAPResources(checkpointsRootURL: checkpointsRoot)
                .missingFiles(fileManager: fileManager)
        case .wooshSynchformer:
            return WooshSynchformerResources(rootURL: rootURL)
                .missingFiles(fileManager: fileManager)
        case .mmaudio:
            return MMAudioModelResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo23MLX:
            return Self.missingLTXVideo23MLXPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo23FullMLX:
            return Self.missingLTXVideo23FullMLXPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo23A2VMLX:
            return Self.missingLTXVideo23A2VMLXPaths(in: rootURL, fileManager: fileManager)
        case .ltxVideo25:
            let resources = LTX25Resources(rootURL: rootURL)
            return id == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
                ? resources.validateFull(fileManager: fileManager)
                : resources.validate(fileManager: fileManager)
        case .wan22TI2VMLX:
            return Wan2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .miniMaxH3MLX:
            return MiniMaxH3Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .cosmos3EdgeMLX:
            let resources = Cosmos3Resources(rootURL: rootURL)
            return resources.validate(fileManager: fileManager)
                + resources.validateReasoner(fileManager: fileManager)
        case .scail2MLX:
            return SCAIL2Resources(rootURL: rootURL).validate(fileManager: fileManager)
        case .dreamXCausalMLX:
            return Wan2DreamXCausalResources(rootURL: rootURL).validate(fileManager: fileManager)
        case .hfTextChat:
            return Self.missingHFTextRootPaths(in: rootURL, fileManager: fileManager)
        }
    }

    func validationMessages(in rootURL: URL, fileManager: FileManager = .default) -> [String] {
        switch validationKind {
        case .qwenImageEdit where id == QwenImageEditRepository.lightning2511Id:
            do {
                _ = try QwenImageEditRepository.lightningPin.verify(
                    in: normalizedRootURL(rootURL, fileManager: fileManager),
                    fileManager: fileManager
                )
                return missingPaths(in: rootURL, fileManager: fileManager).map {
                    "Missing required file: \($0.path)"
                }
            } catch {
                return [error.localizedDescription]
            }
        case .nemotronOmni:
            return NemotronOmniResources.validationMessages(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR:
            guard let pin = GeometryModelPins.pin(for: id) else {
                return ["Missing exact artifact pin for managed model \(id)."]
            }
            return Self.pinnedArtifactValidationMessages(
                pin.runtimeArtifacts,
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )
        case .instantMesh:
            let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
            if Self.invalidInstantMeshNativeArtifacts(in: normalized, fileManager: fileManager).isEmpty {
                return []
            }
            guard let pin = GeometryModelPins.pin(for: id) else {
                return ["Missing exact artifact pin for managed model \(id)."]
            }
            return Self.pinnedArtifactValidationMessages(
                pin.artifacts,
                in: normalized,
                fileManager: fileManager
            )
        case .trellis2:
            return Trellis2Resources.validationMessages(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )
        case .sam31:
            return SAM31Resources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .falconPerception:
            return FalconPerceptionResources.validateRoot(normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
        case .ltxVideo:
            return Self.missingLTXVideoPaths(in: normalizedRootURL(rootURL, fileManager: fileManager), fileManager: fileManager)
                .map { "Missing required LTX file: \($0.path)" }
        case .ltxVideo23MLX:
            return Self.missingLTXVideo23MLXPaths(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required LTX 2.3 MLX file: \($0.path)" }
        case .ltxVideo23FullMLX:
            return Self.missingLTXVideo23FullMLXPaths(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required LTX 2.3 full MLX file: \($0.path)" }
        case .ltxVideo23A2VMLX:
            return Self.missingLTXVideo23A2VMLXPaths(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required LTX 2.3 A2Vid MLX file: \($0.path)" }
        case .ltxVideo25:
            let resources = LTX25Resources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            )
            return (id == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
                ? resources.validateFull(fileManager: fileManager)
                : resources.validate(fileManager: fileManager))
                .map { "Missing required LTX 2.5 file: \($0.path)" }
        case .wan22TI2VMLX:
            let resources = Wan2Resources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager))
            let missing = resources.validate(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required Wan2.2 TI2V MLX file: \($0.path)" }
            }
            return []
        case .miniMaxH3MLX:
            let resources = MiniMaxH3Resources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            )
            let missing = resources.validate(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required MiniMax-H3 MLX file: \($0.path)" }
            }
            do {
                let configuration = try resources.loadConfiguration()
                let expectedTask = id == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue ? "ref2va" : "fl2va"
                guard configuration.task == expectedTask else {
                    return ["MiniMax-H3 model \(id) requires partition \(expectedTask), got \(configuration.task)."]
                }
                if id == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue {
                    let expectedQuantization = MiniMaxH3QuantizationConfiguration(
                        bits: 8,
                        groupSize: 64,
                        mode: "affine"
                    )
                    guard configuration.quantization == expectedQuantization,
                          configuration.textEncoderQuantization == expectedQuantization else {
                        return ["MiniMax-H3 Ref2VA requires MLX affine INT8/group-64 transformer and conditioner weights."]
                    }
                    return resources.validateManagedRef2VAArtifact(fileManager: fileManager)
                }
                if id == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue {
                    guard try resources.transformerStorage() == .compactBF16,
                          configuration.quantization == nil else {
                        return ["MiniMax-H3 BF16 requires an unquantized compact BF16 transformer."]
                    }
                    return resources.validateCompactCachePack(fileManager: fileManager)
                }
                if id == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue {
                    let expectedQuantization = MiniMaxH3QuantizationConfiguration(
                        bits: 8,
                        groupSize: 64,
                        mode: "affine"
                    )
                    guard try resources.transformerStorage() == .affineQ8,
                          configuration.quantization == expectedQuantization,
                          configuration.textEncoderQuantization == expectedQuantization else {
                        return ["FastH3 VSA DataFree requires MLX affine INT8/group-64 transformer and conditioner weights."]
                    }
                    return resources.validateManagedFastH3Artifact(fileManager: fileManager)
                }
                if id == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue {
                    let expectedQuantization = MiniMaxH3QuantizationConfiguration(
                        bits: 8,
                        groupSize: 64,
                        mode: "affine"
                    )
                    guard try resources.transformerStorage() == .affineQ8,
                          configuration.quantization == expectedQuantization,
                          configuration.textEncoderQuantization == expectedQuantization else {
                        return ["MiniMax-H3 Q8 requires MLX affine INT8/group-64 transformer and conditioner weights."]
                    }
                    return resources.validateCompactCachePack(fileManager: fileManager)
                }
                return []
            } catch {
                return [error.localizedDescription]
            }
        case .cosmos3EdgeMLX:
            let resources = Cosmos3Resources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            )
            let missing = resources.validate(fileManager: fileManager)
                + resources.validateReasoner(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required Cosmos3-Edge file: \($0.path)" }
            }
            do {
                _ = try resources.loadTransformerConfiguration()
                _ = try resources.loadVAEConfiguration()
                _ = try resources.loadReasonerConfiguration()
                return []
            } catch {
                return [error.localizedDescription]
            }
        case .scail2MLX:
            let resources = SCAIL2Resources(rootURL: normalizedRootURL(rootURL, fileManager: fileManager))
            let missing = resources.validate(fileManager: fileManager)
            if !missing.isEmpty {
                return missing.map { "Missing required SCAIL-2 MLX file: \($0.path)" }
            }
            do {
                _ = try resources.loadConfiguration()
                return []
            } catch {
                return [error.localizedDescription]
            }
        case .dreamXCausalMLX:
            return Wan2DreamXCausalResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            ).validate(fileManager: fileManager).map {
                "Missing required DreamX causal MLX file: \($0.path)"
            }
        case .magentaRT2:
            return Self.missingMagentaRT2Paths(
                modelID: id,
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            ).map { "Missing required Magenta RT2 file: \($0.path)" }
        case .roFormer:
            if let resources = try? RoFormerResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager),
                modelID: id
            ) {
                return resources.validationMessages(fileManager: fileManager)
            }
            if let resources = try? MelBandRoFormerResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager),
                modelID: id
            ) {
                return resources.validationMessages(fileManager: fileManager)
            }
            return ["Unsupported managed RoFormer model id: \(id)"]
        case .apBWE:
            return APBWEResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            ).validationMessages(fileManager: fileManager)
        case .univerSR:
            return UniverSRResources(
                rootURL: normalizedRootURL(rootURL, fileManager: fileManager)
            ).validationMessages(fileManager: fileManager)
        default:
            return missingPaths(in: rootURL, fileManager: fileManager).map { "Missing required file: \($0.path)" }
        }
    }

    func isManagedRootComplete(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        missingPaths(in: rootURL, fileManager: fileManager).isEmpty
            && managedSourceMatches(rootURL, fileManager: fileManager)
    }

    func isManagedRuntimeReady(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
        switch validationKind {
        case .instantMesh:
            return Self.invalidInstantMeshNativeArtifacts(
                in: normalized,
                fileManager: fileManager
            ).isEmpty
        case .terramindFlood:
            return (try? TerraMindFloodResources.inspect(normalized)) != nil
        case .terramindFire:
            return (try? TerraMindFireResources.inspect(normalized)) != nil
        case .tessera:
            return (try? TESSERAResources.inspect(normalized))?.source.modelID == id
        case .olmoEarth:
            return (try? OlmoEarthResources.inspect(normalized))?.source.modelID == id
        case .moge2, .videoDepthAnything, .depthAnything3, .tripoSR:
            guard let pin = GeometryModelPins.pin(for: id) else { return false }
            return Self.invalidPinnedArtifacts(
                pin.runtimeArtifacts,
                in: normalized,
                fileManager: fileManager
            ).isEmpty
        default:
            return isManagedRootComplete(normalized, fileManager: fileManager)
        }
    }

    func managedSourceMatches(_ rootURL: URL, fileManager: FileManager) -> Bool {
        if id == QwenImageEditRepository.lightning2511Id {
            return (try? QwenImageEditRepository.lightningPin.verify(
                in: normalizedRootURL(rootURL, fileManager: fileManager),
                fileManager: fileManager
            )) != nil
        }
        let requiresPinnedSource = id == ModelResolver.ModelID.zetaNano.rawValue
            || id == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
        guard requiresPinnedSource, let expectedRepo = upstreamRepoId else {
            return true
        }
        let normalized = normalizedRootURL(rootURL, fileManager: fileManager)
        guard let manifest = try? MereRunModelManifest.loadIfPresent(from: normalized, fileManager: fileManager),
              let installedRepo = manifest.upstreamRepoId else {
            return id != ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
                && id != ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
                && id != ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
        }

        let requiresExactRevision = id == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue
            || id == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
        if installedRepo == expectedRepo, !requiresExactRevision {
            return true
        }
        if let expectedWithRevision = upstreamRevision.map({ "\(expectedRepo)@\($0)" }),
           installedRepo == expectedWithRevision {
            return true
        }
        if upstreamRevision == nil, installedRepo == expectedRepo {
            return true
        }
        if id == ModelResolver.ModelID.zetaNano.rawValue,
           installedRepo == "\(expectedRepo)@\(ZImageTurboRepository.revision)" {
            return true
        }
        return false
    }

    func managedInstallRootURL() -> URL {
        MereRunModelPaths.modelDir(id)
    }

    func managedRuntimeURL(fileManager: FileManager = .default) -> URL? {
        switch installShape {
        case .directoryRoot, .structuredRoot:
            if let modelID {
                guard let resolved = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) else {
                    return nil
                }
                return resolved.rootURL
            }
            let root = normalizedRootURL(managedInstallRootURL(), fileManager: fileManager)
            return validateRuntimeURL(root, fileManager: fileManager).isEmpty ? root : nil
        case .singleFile(let relativePath):
            if let modelID,
               let resolved = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID),
               let externalFile = Self.findFirstGGUFFile(
                   in: resolved.rootURL,
                   fileManager: fileManager
               ),
               validateRuntimeURL(externalFile, fileManager: fileManager).isEmpty {
                return externalFile
            }
            let aliasURL = MereRunModelPaths.resolveModelFile(relativePath: relativePath) { candidate in
                self.validateRuntimeURL(candidate, fileManager: fileManager).isEmpty
            }
            if validateRuntimeURL(aliasURL, fileManager: fileManager).isEmpty {
                return aliasURL
            }

            let root = managedInstallRootURL()
            let normalizedRoot = normalizedRootURL(root, fileManager: fileManager)
            if let gguf = Self.findFirstGGUFFile(in: normalizedRoot, fileManager: fileManager),
               validateRuntimeURL(gguf, fileManager: fileManager).isEmpty {
                return gguf
            }
            return nil
        }
    }

    func validateRuntimeURL(_ url: URL, fileManager: FileManager = .default) -> [URL] {
        switch installShape {
        case .singleFile:
            switch validationKind {
            case .codegenGGUF:
                return fileManager.fileExists(atPath: url.path) ? [] : [url]
            case .deepseekV4FlashIMatrixGGUF:
                // Accept any GGUF whose name marks it as imatrix-tuned.
                if fileManager.fileExists(atPath: url.path),
                   url.lastPathComponent.lowercased().contains("imatrix") {
                    return []
                }
                return [url]
            default:
                return fileManager.fileExists(atPath: url.path) ? [] : [url]
            }
        case .directoryRoot, .structuredRoot:
            return missingPaths(in: url, fileManager: fileManager)
        }
    }

    private static func missingFiles(
        _ relativePaths: [String],
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        relativePaths
            .map { rootURL.appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func invalidPinnedArtifacts(
        _ artifacts: [ModelArtifactPin],
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        artifacts.compactMap { artifact in
            do {
                _ = try artifact.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return rootURL.appendingPathComponent(artifact.filename)
            }
        }
    }

    private static func pinnedArtifactValidationMessages(
        _ artifacts: [ModelArtifactPin],
        in rootURL: URL,
        fileManager: FileManager
    ) -> [String] {
        artifacts.compactMap { artifact in
            do {
                _ = try artifact.verify(in: rootURL, fileManager: fileManager)
                return nil
            } catch {
                return error.localizedDescription
            }
        }
    }

    private static func invalidInstantMeshNativeArtifacts(
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        let native = rootURL.appendingPathComponent(
            InstantMeshResources.managedConvertedDirectoryName,
            isDirectory: true
        )
        return invalidPinnedArtifacts(
            [
                InstantMeshResources.convertedWeightsPin,
                InstantMeshResources.convertedConfigurationPin,
                InstantMeshResources.convertedSourceManifestPin,
                InstantMeshResources.convertedLicensePin,
            ],
            in: native,
            fileManager: fileManager
        )
    }

    private static func missingQwen3TTSPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let speechTokenizerDir = rootURL.appendingPathComponent("speech_tokenizer", isDirectory: true)
        let speechTokenizerConfig = speechTokenizerDir.appendingPathComponent("config.json")
        let tokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let vocab = rootURL.appendingPathComponent("vocab.json")
        let merges = rootURL.appendingPathComponent("merges.txt")
        let tokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")

        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        if !fileManager.fileExists(atPath: speechTokenizerConfig.path) { missing.append(speechTokenizerConfig) }
        let tokenizerWeights = (try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: speechTokenizerDir,
            includingPropertiesForKeys: nil
        ))?.filter {
            $0.pathExtension == "safetensors"
        } ?? []
        if tokenizerWeights.isEmpty { missing.append(speechTokenizerDir) }
        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSON.path)
        let hasVocab = fileManager.fileExists(atPath: vocab.path)
        let hasMerges = fileManager.fileExists(atPath: merges.path)
        if !hasTokenizerJSON && !(hasVocab && hasMerges) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingQwen3ASRPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let vocab = rootURL.appendingPathComponent("vocab.json")
        let merges = rootURL.appendingPathComponent("merges.txt")
        let tokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")

        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        let hasTokenizerJSON = fileManager.fileExists(atPath: tokenizerJSON.path)
        let hasVocab = fileManager.fileExists(atPath: vocab.path)
        let hasMerges = fileManager.fileExists(atPath: merges.path)
        if !hasTokenizerJSON && !(hasVocab && hasMerges) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingParakeetPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerModel = rootURL.appendingPathComponent("tokenizer.model")
        let tokenizerVocab = rootURL.appendingPathComponent("tokenizer.vocab")
        let vocabTxt = rootURL.appendingPathComponent("vocab.txt")

        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        let hasTokenizer = fileManager.fileExists(atPath: tokenizerModel.path)
            || fileManager.fileExists(atPath: tokenizerVocab.path)
            || fileManager.fileExists(atPath: vocabTxt.path)
        if !hasTokenizer { missing.append(tokenizerModel) }
        return missing
    }

    private static func missingSortformerPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        Self.missingFiles(
            ["config.json", "model.safetensors"],
            in: rootURL,
            fileManager: fileManager
        )
    }

    private static func missingHFTextRootPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelIndexURL = rootURL.appendingPathComponent("model.safetensors.index.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle { missing.append(modelIndexURL) }
        if hasIndex {
            missing.append(contentsOf: missingShardPaths(indexURL: modelIndexURL, fileManager: fileManager))
        }
        if !fileManager.fileExists(atPath: tokenizerJSON.path) { missing.append(tokenizerJSON) }
        if !fileManager.fileExists(atPath: tokenizerConfig.path) { missing.append(tokenizerConfig) }
        return missing
    }

    private static func missingShardPaths(indexURL: URL, fileManager: FileManager) -> [URL] {
        guard let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return []
        }

        let rootURL = indexURL.deletingLastPathComponent()
        return index.shardFilenames
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingCodeGenPaths(
        preferredRelativePath: String?,
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        if let preferredRelativePath {
            let preferredURL = rootURL.appendingPathComponent(preferredRelativePath, isDirectory: false)
            if isRegularFileOrSymlinkTarget(preferredURL, fileManager: fileManager) {
                return []
            }
        }
        return findFirstGGUFFile(in: rootURL, fileManager: fileManager) == nil
            ? [rootURL.appendingPathComponent("*.gguf")]
            : []
    }

    /// DeepSeek V4 Flash explicitly prefers the imatrix-tuned GGUF (per the
    /// upstream README's "USE THE IMATRIX VERSIONS" note). A directory that
    /// only contains the legacy non-imatrix GGUF is considered *not* complete,
    /// so `mere.run model pull` will fetch the preferred variant instead of
    /// reporting "already installed."
    private static func missingDeepseekV4FlashIMatrixPaths(
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        guard let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return [rootURL.appendingPathComponent("*imatrix*.gguf")]
        }
        while let candidate = enumerator.nextObject() as? URL {
            guard candidate.pathExtension.lowercased() == "gguf",
                  candidate.lastPathComponent.lowercased().contains("imatrix") else {
                continue
            }
            if isRegularFileOrSymlinkTarget(candidate, fileManager: fileManager) {
                return []
            }
        }
        return [rootURL.appendingPathComponent("*imatrix*.gguf")]
    }

    private static func missingLightOnOCRPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let configURL = rootURL.appendingPathComponent("config.json")
        let modelWeightsURL = rootURL.appendingPathComponent("model.safetensors")
        let tokenizerURL = rootURL.appendingPathComponent("tokenizer")
        let tokenizerJSON = tokenizerURL.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = tokenizerURL.appendingPathComponent("tokenizer_config.json")
        let rootTokenizerJSON = rootURL.appendingPathComponent("tokenizer.json")
        let rootTokenizerConfig = rootURL.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: configURL.path) { missing.append(configURL) }
        if !fileManager.fileExists(atPath: modelWeightsURL.path) { missing.append(modelWeightsURL) }
        let hasTokenizer = fileManager.fileExists(atPath: tokenizerJSON.path)
            || fileManager.fileExists(atPath: rootTokenizerJSON.path)
        if !hasTokenizer { missing.append(rootTokenizerJSON) }
        let hasTokenizerConfig = fileManager.fileExists(atPath: tokenizerConfig.path)
            || fileManager.fileExists(atPath: rootTokenizerConfig.path)
        if !hasTokenizerConfig { missing.append(rootTokenizerConfig) }
        return missing
    }

    private static func missingACEStepPaths(modelID: String, in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let decoderSubdirectories: [String]
        switch modelID {
        case ModelResolver.ModelID.aceStep.rawValue:
            decoderSubdirectories = ["acestep-v15-turbo", "music-acestep-v15-turbo"]
        case ModelResolver.ModelID.aceStepXLBase.rawValue:
            decoderSubdirectories = ["acestep-v15-xl-base"]
        case ModelResolver.ModelID.aceStepXLSFT.rawValue:
            decoderSubdirectories = ["acestep-v15-xl-sft"]
        default:
            decoderSubdirectories = ["acestep-v15-xl-turbo"]
        }
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let textDir = rootURL.appendingPathComponent("Qwen3-Embedding-0.6B", isDirectory: true)
        if !decoderSubdirectories.contains(where: {
            fileManager.fileExists(atPath: rootURL.appendingPathComponent($0, isDirectory: true).path)
        }) {
            missing.append(rootURL.appendingPathComponent(decoderSubdirectories[0], isDirectory: true))
        }
        if !fileManager.fileExists(atPath: vaeDir.path) { missing.append(vaeDir) }
        if !fileManager.fileExists(atPath: textDir.path) { missing.append(textDir) }
        if modelID == ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue {
            let lmDir = rootURL.appendingPathComponent("acestep-5Hz-lm-4B", isDirectory: true)
            let lmMissing = ACEStep5HzLMResources(rootURL: lmDir).validate(fileManager: fileManager)
            if !lmMissing.isEmpty {
                missing.append(lmDir)
            }
        }
        return missing
    }

    private static func missingMagentaRT2Paths(
        modelID: String,
        in rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        let modelName = modelID == ModelResolver.ModelID.magentaRT2Base.rawValue ? "mrt2_base" : "mrt2_small"
        let relativePaths = [
            "models/\(modelName)/\(modelName).mlxfn",
            "models/\(modelName)/\(modelName)_state.safetensors",
            "resources/musiccoca/audio_preprocessor.tflite",
            "resources/musiccoca/mapper.tflite",
            "resources/musiccoca/music_encoder.tflite",
            "resources/musiccoca/pretrained_vector_quantizer.tflite",
            "resources/musiccoca/spm.model",
            "resources/musiccoca/text_encoder.tflite",
            "resources/spectrostream/decoder.safetensors",
            "resources/spectrostream/encoder.safetensors",
            "resources/spectrostream/quantizer.safetensors",
            "resources/spectrostream/spectrostream_encoder.mlxfn",
        ]
        return relativePaths
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingLTXVideoPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing: [URL] = []
        let textEncoderConfig = rootURL.appendingPathComponent("text_encoder/config.json")
        let textEncoderWeights = rootURL.appendingPathComponent("text_encoder/model.safetensors.index.json")
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        if !fileManager.fileExists(atPath: textEncoderConfig.path) { missing.append(textEncoderConfig) }
        if !fileManager.fileExists(atPath: textEncoderWeights.path) { missing.append(textEncoderWeights) }
        if !fileManager.fileExists(atPath: tokenizerDir.path) { missing.append(tokenizerDir) }
        let entries = (try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        let hasTransformer = entries.contains { $0.lastPathComponent.hasPrefix("ltx-2-19") && $0.pathExtension == "safetensors" }
        let hasUpsampler = entries.contains { $0.lastPathComponent.hasPrefix("ltx-2-spatial-upscaler") && $0.pathExtension == "safetensors" }
        if !hasTransformer { missing.append(rootURL.appendingPathComponent("ltx-2-19b-distilled.safetensors")) }
        if !hasUpsampler { missing.append(rootURL.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors")) }
        return missing
    }

    private static func missingLTXVideo23MLXPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let relativePaths = [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-distilled.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x1_5_v1_0.safetensors",
            "spatial_upscaler_x1_5_v1_0_config.json",
            "temporal_upscaler_x2_v1_0.safetensors",
            "temporal_upscaler_x2_v1_0_config.json",
        ]
        return relativePaths
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingLTXVideo23A2VMLXPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let relativePaths = [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
        ]
        return relativePaths
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    private static func missingLTXVideo23FullMLXPaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        var missing = missingLTXVideo23A2VMLXPaths(in: rootURL, fileManager: fileManager)
        let vocoder = rootURL.appendingPathComponent("vocoder.safetensors", isDirectory: false)
        if !fileManager.fileExists(atPath: vocoder.path) {
            missing.append(vocoder)
        }
        return missing
    }

    private static func missingDiffusersImagePaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoderDir = rootURL.appendingPathComponent("text_encoder", isDirectory: true)
        let transformerDir = rootURL.appendingPathComponent("transformer", isDirectory: true)
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let schedulerDir = rootURL.appendingPathComponent("scheduler", isDirectory: true)

        var missing: [URL] = []
        let required: [URL] = [
            rootURL.appendingPathComponent("model_index.json"),
            tokenizerDir.appendingPathComponent("tokenizer_config.json"),
            textEncoderDir.appendingPathComponent("config.json"),
            transformerDir.appendingPathComponent("config.json"),
            vaeDir.appendingPathComponent("config.json"),
            schedulerDir.appendingPathComponent("scheduler_config.json"),
        ]
        for path in required where !fileManager.fileExists(atPath: path.path) {
            missing.append(path)
        }

        let textWeights = textEncoderDir.appendingPathComponent("model.safetensors")
        let textWeightsIndex = textEncoderDir.appendingPathComponent("model.safetensors.index.json")
        if !fileManager.fileExists(atPath: textWeights.path) && !fileManager.fileExists(atPath: textWeightsIndex.path) {
            missing.append(textWeightsIndex)
        }

        let transformerWeights = transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        let transformerWeightsIndex = transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        if !fileManager.fileExists(atPath: transformerWeights.path) && !fileManager.fileExists(atPath: transformerWeightsIndex.path) {
            missing.append(transformerWeightsIndex)
        }

        let vaeWeights = vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if !fileManager.fileExists(atPath: vaeWeights.path) {
            missing.append(vaeWeights)
        }

        return missing
    }

    private static func missingBonsaiImagePaths(in rootURL: URL, fileManager: FileManager) -> [URL] {
        let tokenizerDir = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoderDir = rootURL.appendingPathComponent("text_encoder-mlx-4bit", isDirectory: true)
        let transformerDir = rootURL.appendingPathComponent("transformer-packed-mflux", isDirectory: true)
        let vaeDir = rootURL.appendingPathComponent("vae", isDirectory: true)
        let schedulerDir = rootURL.appendingPathComponent("scheduler", isDirectory: true)

        var missing: [URL] = []
        let required: [URL] = [
            rootURL.appendingPathComponent("manifest.json"),
            tokenizerDir.appendingPathComponent("tokenizer_config.json"),
            textEncoderDir.appendingPathComponent("config.json"),
            transformerDir.appendingPathComponent("config.json"),
            transformerDir.appendingPathComponent("quantization_config.json"),
            vaeDir.appendingPathComponent("config.json"),
            schedulerDir.appendingPathComponent("scheduler_config.json"),
        ]
        for path in required where !fileManager.fileExists(atPath: path.path) {
            missing.append(path)
        }

        let textWeights = textEncoderDir.appendingPathComponent("model.safetensors")
        let textWeightsIndex = textEncoderDir.appendingPathComponent("model.safetensors.index.json")
        if !fileManager.fileExists(atPath: textWeights.path) && !fileManager.fileExists(atPath: textWeightsIndex.path) {
            missing.append(textWeightsIndex)
        }

        let transformerWeights = transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if !fileManager.fileExists(atPath: transformerWeights.path) {
            missing.append(transformerWeights)
        }

        let vaeWeights = vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors")
        if !fileManager.fileExists(atPath: vaeWeights.path) {
            missing.append(vaeWeights)
        }

        return missing
    }

    static func findFirstGGUFFile(in rootURL: URL, fileManager: FileManager = .default) -> URL? {
        let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        while let candidate = enumerator?.nextObject() as? URL {
            guard candidate.pathExtension.lowercased() == "gguf" else { continue }
            if isRegularFileOrSymlinkTarget(candidate, fileManager: fileManager) {
                return candidate
            }
        }
        return nil
    }

    private static func isRegularFileOrSymlinkTarget(_ url: URL, fileManager: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && !isDirectory.boolValue
    }
}
