import Foundation
import MLX
import MLXNN

public struct ACEStepModelResources: Sendable, Hashable {
    public var decoderResources: ACEStepResources
    public var vaeResources: OobleckVAEResources
    public var lmResources: ACEStep5HzLMResources?
    public var textEncoderResources: ACEStep5HzLMResources?

    public init(
        decoderResources: ACEStepResources,
        vaeResources: OobleckVAEResources,
        lmResources: ACEStep5HzLMResources? = nil,
        textEncoderResources: ACEStep5HzLMResources? = nil
    ) {
        self.decoderResources = decoderResources
        self.vaeResources = vaeResources
        self.lmResources = lmResources
        self.textEncoderResources = textEncoderResources
    }
}

public actor ACEStepModelContainer {
    public enum ContainerError: LocalizedError {
        case missingDecoderFiles([URL])
        case missingVAEFiles([URL])
        case missingLMFiles([URL])
        case missingTextEncoderFiles([URL])

        public var errorDescription: String? {
            switch self {
            case .missingDecoderFiles(let urls):
                let list = urls.map(\.path).sorted().joined(separator: "\n")
                return "ACE-Step decoder files missing:\n\(list)"
            case .missingVAEFiles(let urls):
                let list = urls.map(\.path).sorted().joined(separator: "\n")
                return "ACE-Step VAE files missing:\n\(list)"
            case .missingLMFiles(let urls):
                let list = urls.map(\.path).sorted().joined(separator: "\n")
                return "ACE-Step 5Hz LM files missing:\n\(list)"
            case .missingTextEncoderFiles(let urls):
                let list = urls.map(\.path).sorted().joined(separator: "\n")
                return "ACE-Step text encoder files missing:\n\(list)"
            }
        }
    }

    private let configuredResources: ACEStepModelResources
    private let dtype: DType?
    private let verify: Module.VerifyUpdate

    private var cachedResources: ACEStepModelResources?
    private var cachedPipeline: ACEStepPipeline?

    public init(
        decoderResources: ACEStepResources,
        vaeResources: OobleckVAEResources,
        lmResources: ACEStep5HzLMResources? = nil,
        textEncoderResources: ACEStep5HzLMResources? = nil,
        dtype: DType? = .float32,
        verify: Module.VerifyUpdate = .noUnusedKeys
    ) {
        self.configuredResources = ACEStepModelResources(
            decoderResources: decoderResources,
            vaeResources: vaeResources,
            lmResources: lmResources,
            textEncoderResources: textEncoderResources
        )
        self.dtype = dtype
        self.verify = verify
    }

    public init(
        decoderRootURL: URL,
        vaeRootURL: URL,
        lmRootURL: URL? = nil,
        textEncoderRootURL: URL? = nil,
        dtype: DType? = .float32,
        verify: Module.VerifyUpdate = .noUnusedKeys
    ) {
        self.configuredResources = ACEStepModelResources(
            decoderResources: ACEStepResources(rootURL: decoderRootURL),
            vaeResources: OobleckVAEResources(rootURL: vaeRootURL),
            lmResources: lmRootURL.map(ACEStep5HzLMResources.init(rootURL:)),
            textEncoderResources: textEncoderRootURL.map(ACEStep5HzLMResources.init(rootURL:))
        )
        self.dtype = dtype
        self.verify = verify
    }

    public init(
        checkpointsRootURL: URL,
        turboSubdirectory: String = "acestep-v15-turbo",
        vaeSubdirectory: String = "vae",
        lmSubdirectory: String? = nil,
        textEncoderSubdirectory: String? = nil,
        dtype: DType? = .float32,
        verify: Module.VerifyUpdate = .noUnusedKeys
    ) {
        self.configuredResources = ACEStepModelResources(
            decoderResources: ACEStepResources(rootURL: checkpointsRootURL.appending(path: turboSubdirectory)),
            vaeResources: OobleckVAEResources(rootURL: checkpointsRootURL.appending(path: vaeSubdirectory)),
            lmResources: lmSubdirectory.map { ACEStep5HzLMResources(rootURL: checkpointsRootURL.appending(path: $0)) },
            textEncoderResources: textEncoderSubdirectory.map {
                ACEStep5HzLMResources(rootURL: checkpointsRootURL.appending(path: $0))
            }
        )
        self.dtype = dtype
        self.verify = verify
    }

    public func resources() throws -> ACEStepModelResources {
        if let cachedResources {
            return cachedResources
        }

        let decoderMissing = configuredResources.decoderResources.validate(fileManager: .default)
        if !decoderMissing.isEmpty {
            throw ContainerError.missingDecoderFiles(decoderMissing)
        }

        let vaeMissing = configuredResources.vaeResources.validate(fileManager: .default)
        if !vaeMissing.isEmpty {
            throw ContainerError.missingVAEFiles(vaeMissing)
        }

        if let lmResources = configuredResources.lmResources {
            let lmMissing = lmResources.validate(fileManager: .default)
            if !lmMissing.isEmpty {
                throw ContainerError.missingLMFiles(lmMissing)
            }
        }

        if let textResources = configuredResources.textEncoderResources {
            let textMissing = textResources.validate(fileManager: .default)
            if !textMissing.isEmpty {
                throw ContainerError.missingTextEncoderFiles(textMissing)
            }
        }

        cachedResources = configuredResources
        return configuredResources
    }

    public func pipeline() throws -> ACEStepPipeline {
        if let cachedPipeline {
            return cachedPipeline
        }

        let resources = try resources()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            lmResources: resources.lmResources,
            textEncoderResources: resources.textEncoderResources,
            dtype: dtype,
            verify: verify,
            fileManager: .default
        )
        cachedPipeline = pipeline
        return pipeline
    }

    public func invalidateCache() {
        cachedResources = nil
        cachedPipeline = nil
    }
}
