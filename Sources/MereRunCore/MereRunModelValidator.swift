import Foundation

public struct MereRunModelValidationReport: Hashable, Sendable {
    public let rootURL: URL
    public let manifest: MereRunModelManifest?
    public var warnings: [String]
    public var errors: [String]

    public init(
        rootURL: URL,
        manifest: MereRunModelManifest?,
        warnings: [String] = [],
        errors: [String] = []
    ) {
        self.rootURL = rootURL
        self.manifest = manifest
        self.warnings = warnings
        self.errors = errors
    }

    public var isValid: Bool { errors.isEmpty }
}

public enum MereRunModelValidator {
    public enum ValidationError: LocalizedError, Sendable {
        case invalidModelRoot(URL, details: [String])

        public var errorDescription: String? {
            switch self {
            case .invalidModelRoot(let url, let details):
                var lines: [String] = []
                lines.append("Invalid model directory: \(url.path)")
                if !details.isEmpty {
                    lines.append(contentsOf: details.map { "  - \($0)" })
                }
                return lines.joined(separator: "\n")
            }
        }
    }

    public static func validate(
        modelRoot rootURL: URL,
        expectedModelID: String? = nil,
        fileManager: FileManager = .default
    ) -> MereRunModelValidationReport {
        var warnings: [String] = []
        var errors: [String] = []

        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &isDir), isDir.boolValue else {
            return MereRunModelValidationReport(
                rootURL: rootURL,
                manifest: nil,
                warnings: [],
                errors: ["Missing model directory"]
            )
        }

        let manifest: MereRunModelManifest?
        do {
            manifest = try MereRunModelManifest.loadIfPresent(from: rootURL, fileManager: fileManager)
        } catch {
            manifest = nil
            errors.append("Failed to decode \(MereRunModelManifest.filename): \(error.localizedDescription)")
        }

        let supportsImagePipeline: Bool
        if let unwrappedManifest = manifest {
            let supports = Set(unwrappedManifest.supports ?? [])
            supportsImagePipeline = supports.contains(.txt2img) || supports.contains(.img2img) || supports.contains(.referenceEdit)
        } else {
            supportsImagePipeline = true
        }

        if let manifest = manifest {
            validateManifestSemantics(
                manifest,
                expectedModelID: expectedModelID,
                warnings: &warnings,
                errors: &errors
            )
        } else {
            errors.append("Missing \(MereRunModelManifest.filename). Manifests are required (no heuristic guessing).")
        }

        let spec = manifest.flatMap { ManagedModelCatalog.spec(for: $0.id) }
        let usesMFluxZImage = spec?.validationKind == .zimageTurbo
            && ZImageTurboResources(rootURL: rootURL).hasMFluxWeights(fileManager: fileManager)

        // Root marker checks: diffusers-style models should have model_index.json, but allow fallback markers.
        let modelIndex = rootURL.appendingPathComponent("model_index.json")
        let rootConfig = rootURL.appendingPathComponent("config.json")

        let hasRootMarker = fileManager.fileExists(atPath: modelIndex.path) || fileManager.fileExists(atPath: rootConfig.path)
        if !hasRootMarker && !usesMFluxZImage {
            warnings.append("Missing model root marker (expected model_index.json).")
        }

        let modelResolver = ModelResolver(fileManager: fileManager)
        let componentRefs = manifest?.components
        let usesUnifiedImageTransformer = manifest?.engine == .hidreamO1
        let transformerDir: URL?
        let textEncoderDir: URL?
        let vaeDir: URL?
        let tokenizerDir: URL?

        if spec?.validationKind == .codegenGGUF {
            errors.append(contentsOf: spec?.validationMessages(in: rootURL, fileManager: fileManager) ?? [])
            transformerDir = nil
            textEncoderDir = nil
            vaeDir = nil
            tokenizerDir = nil
        } else if spec?.validationKind == .aceStep || spec?.validationKind == .ltxVideo {
            errors.append(contentsOf: spec?.validationMessages(in: rootURL, fileManager: fileManager) ?? [])
            transformerDir = nil
            textEncoderDir = nil
            vaeDir = nil
            tokenizerDir = nil
        } else if spec?.validationKind == .sam31 {
            errors.append(contentsOf: SAM31Resources.validateRoot(rootURL, fileManager: fileManager))
            transformerDir = nil
            textEncoderDir = nil
            vaeDir = nil
            tokenizerDir = resolveComponentDirectory(
                componentRefs?.tokenizer ?? .local(path: "tokenizer"),
                modelRoot: rootURL,
                modelResolver: modelResolver,
                fileManager: fileManager,
                label: "tokenizer",
                report: { _ in }
            )
        } else if spec?.validationKind == .falconPerception {
            errors.append(contentsOf: FalconPerceptionResources.validateRoot(rootURL, fileManager: fileManager))
            transformerDir = nil
            textEncoderDir = nil
            vaeDir = nil
            tokenizerDir = resolveComponentDirectory(
                componentRefs?.tokenizer ?? .local(path: "."),
                modelRoot: rootURL,
                modelResolver: modelResolver,
                fileManager: fileManager,
                label: "tokenizer",
                report: { errors.append($0) }
            )
        } else {
            // Required components for generation/training.
            if supportsImagePipeline {
                transformerDir = resolveComponentDirectory(
                    componentRefs?.transformer ?? .local(path: "transformer"),
                    modelRoot: rootURL,
                    modelResolver: modelResolver,
                    fileManager: fileManager,
                    label: "transformer",
                    report: { errors.append($0) }
                )
            } else {
                transformerDir = nil
            }
            if let transformerDir {
                validateComponentDirectory(
                    componentDir: transformerDir,
                    label: "transformer",
                    requiredConfigFilename: usesMFluxZImage ? nil : "config.json",
                    requireWeights: true,
                    errors: &errors,
                    fileManager: fileManager
                )
            }

            textEncoderDir = resolveComponentDirectory(
                componentRefs?.textEncoder ?? .local(path: "text_encoder"),
                modelRoot: rootURL,
                modelResolver: modelResolver,
                fileManager: fileManager,
                label: "text_encoder",
                report: { errors.append($0) }
            )
            if let textEncoderDir {
                validateComponentDirectory(
                    componentDir: textEncoderDir,
                    label: "text_encoder",
                    requiredConfigFilename: usesMFluxZImage ? nil : "config.json",
                    requireWeights: true,
                    errors: &errors,
                    fileManager: fileManager
                )
            }

            if supportsImagePipeline && !usesUnifiedImageTransformer {
                vaeDir = resolveComponentDirectory(
                    componentRefs?.vae ?? .local(path: "vae"),
                    modelRoot: rootURL,
                    modelResolver: modelResolver,
                    fileManager: fileManager,
                    label: "vae",
                    report: { errors.append($0) }
                )
            } else {
                vaeDir = nil
            }
            if let vaeDir {
                validateComponentDirectory(
                    componentDir: vaeDir,
                    label: "vae",
                    requiredConfigFilename: usesMFluxZImage ? nil : "config.json",
                    requireWeights: true,
                    errors: &errors,
                    fileManager: fileManager
                )
            }

            tokenizerDir = resolveComponentDirectory(
                componentRefs?.tokenizer ?? .local(path: "tokenizer"),
                modelRoot: rootURL,
                modelResolver: modelResolver,
                fileManager: fileManager,
                label: "tokenizer",
                report: { _ in }
            )
            if let tokenizerDir {
                validateTokenizerDirectory(
                    tokenizerDir,
                    warnings: &warnings,
                    fileManager: fileManager
                )
            } else {
                errors.append("Missing tokenizer directory.")
            }

            let schedulerDir: URL?
            if supportsImagePipeline && !usesUnifiedImageTransformer {
                schedulerDir = resolveComponentDirectory(
                    componentRefs?.scheduler ?? .local(path: "scheduler"),
                    modelRoot: rootURL,
                    modelResolver: modelResolver,
                    fileManager: fileManager,
                    label: "scheduler",
                    report: { _ in }
                )
            } else {
                schedulerDir = nil
            }
            if let schedulerDir, !usesMFluxZImage {
                let schedulerConfig = schedulerDir.appendingPathComponent("scheduler_config.json")
                if !fileManager.fileExists(atPath: schedulerConfig.path) {
                    errors.append("Missing scheduler/scheduler_config.json.")
                }
            }
        }

        // Incomplete download markers (HF snapshot downloads).
        if containsIncompleteFiles(in: rootURL, fileManager: fileManager) {
            errors.append("Found incomplete download markers (*.incomplete / *.partial).")
        }
        if supportsImagePipeline, let transformerDir, containsIncompleteFiles(in: transformerDir, fileManager: fileManager) {
            errors.append("Found incomplete download markers under transformer component.")
        }
        if let textEncoderDir, containsIncompleteFiles(in: textEncoderDir, fileManager: fileManager) {
            errors.append("Found incomplete download markers under text_encoder component.")
        }
        if let vaeDir, containsIncompleteFiles(in: vaeDir, fileManager: fileManager) {
            errors.append("Found incomplete download markers under vae component.")
        }

        return MereRunModelValidationReport(rootURL: rootURL, manifest: manifest, warnings: warnings, errors: errors)
    }

    public static func assertValid(
        modelRoot rootURL: URL,
        expectedModelID: String? = nil,
        fileManager: FileManager = .default
    ) throws -> MereRunModelValidationReport {
        let report = validate(modelRoot: rootURL, expectedModelID: expectedModelID, fileManager: fileManager)
        guard report.isValid else {
            throw ValidationError.invalidModelRoot(rootURL, details: report.errors)
        }
        return report
    }

    // MARK: - Helpers

    private static func validateManifestSemantics(
        _ manifest: MereRunModelManifest,
        expectedModelID: String?,
        warnings: inout [String],
        errors: inout [String]
    ) {
        if let expectedModelID, manifest.id != expectedModelID {
            errors.append("Manifest id mismatch: expected=\(expectedModelID) found=\(manifest.id)")
        }

        if manifest.schemaVersion > MereRunModelManifest.currentSchemaVersion {
            warnings.append("Manifest schemaVersion=\(manifest.schemaVersion) is newer than this build supports (\(MereRunModelManifest.currentSchemaVersion)).")
        }

        if let inferredFamily = inferFamily(from: manifest.id), let family = manifest.family, family != inferredFamily {
            warnings.append("Manifest family mismatch: id implies \(inferredFamily.rawValue) but manifest says \(family.rawValue).")
        }
        if let inferredTier = inferTier(from: manifest.id), let tier = manifest.tier, tier != inferredTier {
            warnings.append("Manifest tier mismatch: id implies \(inferredTier.rawValue) but manifest says \(tier.rawValue).")
        }

        if let family = manifest.family, let engine = manifest.engine {
            switch family {
            case .klein where engine != .flux2Klein:
                warnings.append("Manifest engine mismatch: family=klein expects flux2-klein.")
            case .zimage where engine != .zimageTurbo:
                warnings.append("Manifest engine mismatch: family=zimage expects zimage-turbo.")
            case .hidream where engine != .hidreamO1:
                warnings.append("Manifest engine mismatch: family=hidream expects hidream-o1.")
            case .gemma where engine != .gemma4:
                warnings.append("Manifest engine mismatch: family=gemma expects gemma-4.")
            case .qwen where engine != .qwen35HybridMoE:
                warnings.append("Manifest engine mismatch: family=qwen expects qwen3.5-hybrid-moe.")
            case .sam where engine != .samSegmentation:
                warnings.append("Manifest engine mismatch: family=sam expects sam-segmentation.")
            case .falcon where engine != .falconPerception:
                warnings.append("Manifest engine mismatch: family=falcon expects falcon-perception.")
            case .tts where engine != .qwen3TTS:
                warnings.append("Manifest engine mismatch: family=tts expects qwen3-tts.")
            case .asr where engine != .qwen3ASR && engine != .parakeetASR:
                warnings.append("Manifest engine mismatch: family=asr expects qwen3-asr or parakeet-asr.")
            case .embed where engine != .qwen3Embedding:
                warnings.append("Manifest engine mismatch: family=embed expects qwen3-embedding.")
            case .privacy where engine != .openAIPrivacyFilter:
                warnings.append("Manifest engine mismatch: family=privacy expects openai-privacy-filter.")
            case .code where engine != .qwen3Coder:
                warnings.append("Manifest engine mismatch: family=code expects qwen3-coder.")
            case .ocr where engine != .lightOnOCR:
                warnings.append("Manifest engine mismatch: family=ocr expects lighton-ocr.")
            case .music where engine != .aceStep:
                warnings.append("Manifest engine mismatch: family=music expects ace-step.")
            case .video where engine != .ltxVideo:
                warnings.append("Manifest engine mismatch: family=video expects ltx-video.")
            case .psi where engine != .psiChat:
                warnings.append("Manifest engine mismatch: family=psi expects psi-chat.")
            default:
                break
            }
        }

        if let precision = manifest.precision {
            switch precision {
            case .int4, .int8:
                guard let q = manifest.quantization else {
                    errors.append("Quantized precision (\(precision.rawValue)) requires quantization metadata.")
                    break
                }
                if let bits = q.bits, !(2...8).contains(bits) {
                    errors.append("Invalid quantization.bits=\(bits) (expected 2–8).")
                }
                if q.bits == nil {
                    errors.append("Quantized precision (\(precision.rawValue)) requires quantization.bits.")
                }
                if let groupSize = q.groupSize, groupSize <= 0 {
                    errors.append("Invalid quantization.groupSize=\(groupSize) (expected > 0).")
                }
                if q.groupSize == nil {
                    errors.append("Quantized precision (\(precision.rawValue)) requires quantization.groupSize.")
                }
                if let residualRank = q.svdResidualRank, residualRank < 0 {
                    errors.append("Invalid quantization.svdResidualRank=\(residualRank) (expected >= 0).")
                }
                if let maxLayers = q.svdMaxLayers, maxLayers < 0 {
                    errors.append("Invalid quantization.svdMaxLayers=\(maxLayers) (expected >= 0).")
                }
            case .bf16, .fp16, .fp32, .unknown:
                if manifest.quantization != nil {
                    warnings.append("Manifest includes quantization metadata but precision=\(precision.rawValue).")
                }
            }
        } else {
            errors.append("Manifest missing precision.")
        }

        if manifest.engine == nil { errors.append("Manifest missing engine.") }
        if manifest.family == nil { errors.append("Manifest missing family.") }
        if manifest.tier == nil { errors.append("Manifest missing tier.") }
        if manifest.variant == nil { errors.append("Manifest missing variant.") }

        if manifest.supports == nil {
            errors.append("Manifest missing supports[] capabilities.")
        }

        let supportsImagePipeline = {
            let supports = Set(manifest.supports ?? [])
            return supports.contains(.txt2img) || supports.contains(.img2img) || supports.contains(.referenceEdit)
        }()
        let usesUnifiedImageTransformer = manifest.engine == .hidreamO1
        let supportsVisionModel = {
            let supports = Set(manifest.supports ?? [])
            return supports.contains(.visionGrounding)
                || supports.contains(.visionDetection)
                || supports.contains(.visionSegmentation)
                || supports.contains(.visionTracking)
                || manifest.family == .sam
                || manifest.family == .falcon
        }()
        let skipsComponentValidation = {
            switch manifest.engine {
            case .qwen3Coder?, .aceStep?, .ltxVideo?:
                return true
            default:
                return false
            }
        }()

        guard let components = manifest.components else {
            if skipsComponentValidation {
                return
            }
            errors.append("Manifest missing components.")
            return
        }

        if components.tokenizer == nil { errors.append("Manifest components missing tokenizer.") }
        if supportsVisionModel {
            return
        }
        if skipsComponentValidation {
            return
        }

        if components.textEncoder == nil { errors.append("Manifest components missing text_encoder.") }
        if supportsImagePipeline && components.transformer == nil { errors.append("Manifest components missing transformer.") }
        if supportsImagePipeline && !usesUnifiedImageTransformer && components.vae == nil { errors.append("Manifest components missing vae.") }
        if supportsImagePipeline && !usesUnifiedImageTransformer && components.scheduler == nil { errors.append("Manifest components missing scheduler.") }
    }

    private static func inferFamily(from modelId: String) -> MereRunModelManifest.Family? {
        if modelId.hasPrefix("image-klein-") { return .klein }
        if modelId.hasPrefix("image-zimage-") { return .zimage }
        if modelId.hasPrefix("image-hidream-") { return .hidream }
        if modelId.hasPrefix("vision-segment-") { return .sam }
        if modelId.hasPrefix("vision-ground-") { return .falcon }
        if modelId.hasPrefix("speech-tts-") { return .tts }
        if modelId.hasPrefix("speech-asr-") { return .asr }
        if modelId.hasPrefix("text-embed-") { return .embed }
        if modelId.hasPrefix("text-anonymize-") { return .privacy }
        if modelId.hasPrefix("text-code-") { return .code }
        if modelId.hasPrefix("vision-ocr-") { return .ocr }
        if modelId.hasPrefix("music-") { return .music }
        if modelId.hasPrefix("video-") { return .video }
        if modelId.hasPrefix("text-chat-psi-") { return .psi }
        if modelId == ModelResolver.ModelID.gemma4.rawValue
            || modelId == ModelResolver.ModelID.gemma4Nano.rawValue
            || modelId == ModelResolver.ModelID.gemma4Max.rawValue {
            return .gemma
        }
        if modelId == ModelResolver.ModelID.q35.rawValue || modelId == ModelResolver.ModelID.q35Nano.rawValue {
            return .qwen
        }
        return nil
    }

    private static func inferTier(from modelId: String) -> MereRunModelManifest.Tier? {
        if modelId.hasSuffix("-nano") { return .nano }
        if modelId.hasSuffix("-max") { return .max }
        if modelId.hasSuffix("-base") { return .base }
        return nil
    }

    private static func resolveComponentDirectory(
        _ ref: MereRunModelManifest.ComponentRef,
        modelRoot: URL,
        modelResolver: ModelResolver,
        fileManager: FileManager,
        label: String,
        report: (String) -> Void
    ) -> URL? {
        func existsDirectory(_ url: URL) -> Bool {
            var isDir: ObjCBool = false
            return fileManager.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }

        func resolveOne(_ ref: MereRunModelManifest.ComponentRef) -> URL? {
            switch ref {
            case .local(let path):
                return modelRoot.appendingPathComponent(path, isDirectory: true)
            case .absolute(let path):
                return URL(fileURLWithPath: path).standardizedFileURL
            case .model(let modelID, let path):
                guard let id = ModelResolver.ModelID(rawValue: modelID) else {
                    report("Component \(label) references unknown model id: \(modelID)")
                    return nil
                }
                do {
                    let resolved = try modelResolver.resolve(id)
                    return resolved.rootURL.appendingPathComponent(path, isDirectory: true)
                } catch {
                    report("Component \(label) references missing model: \(modelID)")
                    return nil
                }
            case .remote:
                report("Component \(label) references remote content, which is not yet resolvable in this validator.")
                return nil
            case .anyOf:
                return nil
            }
        }

        switch ref {
        case .anyOf(let candidates):
            var collected: [String] = []
            for candidate in candidates {
                if let url = resolveComponentDirectory(
                    candidate,
                    modelRoot: modelRoot,
                    modelResolver: modelResolver,
                    fileManager: fileManager,
                    label: label,
                    report: { collected.append($0) }
                ) {
                    return url
                }
            }
            report("Component \(label) could not be resolved from anyOf candidates.")
            for message in collected {
                report("  - \(message)")
            }
            return nil
        default:
            guard let url = resolveOne(ref) else { return nil }
            guard existsDirectory(url) else {
                report("Missing component directory for \(label): \(url.path)")
                return nil
            }
            return url
        }
    }

    private static func validateComponentDirectory(
        componentDir: URL,
        label: String,
        requiredConfigFilename: String?,
        requireWeights: Bool,
        errors: inout [String],
        fileManager: FileManager
    ) {
        if let requiredConfigFilename {
            let configURL = componentDir.appendingPathComponent(requiredConfigFilename)
            if !fileManager.fileExists(atPath: configURL.path) {
                errors.append("Missing \(label)/\(requiredConfigFilename)")
            }
        }

        if requireWeights, !hasAnyWeightFiles(in: componentDir, fileManager: fileManager) {
            errors.append("No *.safetensors weights found in \(label)/")
        }
    }

    private static func validateTokenizerDirectory(
        _ tokenizerDir: URL,
        warnings: inout [String],
        fileManager: FileManager
    ) {
        let tokenizerJSON = tokenizerDir.appendingPathComponent("tokenizer.json")
        let tokenizerConfig = tokenizerDir.appendingPathComponent("tokenizer_config.json")
        if !fileManager.fileExists(atPath: tokenizerJSON.path) && !fileManager.fileExists(atPath: tokenizerConfig.path) {
            warnings.append("Tokenizer directory missing tokenizer.json/tokenizer_config.json.")
        }
    }

    private static func hasAnyWeightFiles(in directory: URL, fileManager: FileManager) -> Bool {
        guard let children = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        for url in children {
            if url.pathExtension == "safetensors" {
                return true
            }
            if url.lastPathComponent.hasSuffix(".safetensors.index.json") {
                return true
            }
        }
        return false
    }

    private static func containsIncompleteFiles(in rootURL: URL, fileManager: FileManager) -> Bool {
        guard let e = fileManager.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return false
        }

        for case let url as URL in e {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            let name = url.lastPathComponent
            if name.hasSuffix(".incomplete") || name.hasSuffix(".partial") {
                return true
            }
        }
        return false
    }
}
