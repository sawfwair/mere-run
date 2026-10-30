import Foundation

public enum IdolError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidWeights(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail), .invalidWeights(let detail), .invalidInput(let detail):
            return "IDOL: \(detail)"
        }
    }
}

/// Publisher configuration, decoded without executing repository code.
public struct IdolConfiguration: Decodable, Sendable {
    public struct Component<Arguments: Decodable & Sendable>: Decodable, Sendable {
        public let name: String
        public let args: Arguments
    }
    public struct Denoiser: Decodable, Sendable {
        public let q_token_length: Int
        public let in_channels: Int
        public let cam_channels: Int
        public let model_channels: Int
        public let cond_channels: Int
        public let cond2_channels: Int
        public let num_blocks: Int
        public let num_refiner_blocks: Int
        public let num_heads: Int
        public let num_head_channels: Int?
        public let mlp_ratio: Float
    }
    public struct Decoder: Decodable, Sendable {
        public let in_channels: Int?
        public let model_channels: Int
        public let cond_channels: Int
        public let num_blocks: Int
        public let num_heads: Int
        public let mlp_ratio: Float
        public let representation_config: Representation?
    }
    public struct Representation: Decodable, Sendable {
        public struct Rates: Decodable, Sendable {
            public let _xyz: Float
            public let _features_dc: Float
            public let _opacity: Float
            public let _scaling: Float
            public let _rotation: Float
        }
        public let num_gaussians: Int
        public let lr: Rates
        public let perturbe_size: Float
        public let filterKernelSize: Float
        public let scaling_bias: Float
        public let opacity_bias: Float
        enum CodingKeys: String, CodingKey {
            case num_gaussians, lr, perturbe_size, scaling_bias, opacity_bias
            case filterKernelSize = "3d_filter_kernel_size"
        }
    }
    public struct Encoders: Decodable, Sendable {
        public let dinov3: String
        public let flux2_vae: String
    }
    public let denoiser: Component<Denoiser>
    public let decoder: Component<Decoder>
    public let decoder_gs: Component<Decoder>
    public let image_size: Int
    public let max_voxel_level: Int
    public let image_encoders: Encoders

    public func validate() throws {
        let den = denoiser.args
        guard denoiser.name == "LatentSeqMMFlowModel", decoder.name == "OctreeProbabilityFixedlenDecoder",
              decoder_gs.name == "GaussianFixedlenDecoder", den.q_token_length > 0, den.q_token_length <= 8192,
              den.in_channels > 0, den.in_channels <= 1024, den.cam_channels == 5,
              den.cond_channels > 0, den.cond_channels <= 4096, den.cond2_channels > 0, den.cond2_channels <= 512,
              den.num_blocks > 0, den.num_blocks <= 64, (2...8).contains(den.num_refiner_blocks),
              image_size > 0, image_size <= 1024, image_size % 16 == 0,
              (1...16).contains(max_voxel_level) else {
            throw IdolError.invalidConfiguration("Unsupported model architecture or dimensions.")
        }
        for (width, heads, ratio) in [(den.model_channels, den.num_heads, den.mlp_ratio),
            (decoder.args.model_channels, decoder.args.num_heads, decoder.args.mlp_ratio),
            (decoder_gs.args.model_channels, decoder_gs.args.num_heads, decoder_gs.args.mlp_ratio)] {
            guard (8...2048).contains(width), heads > 0, width % heads == 0, (width / heads) % 2 == 0,
                  ratio.isFinite, (1...8).contains(ratio) else {
                throw IdolError.invalidConfiguration("Invalid transformer dimensions.")
            }
        }
        guard (den.num_head_channels ?? 64) == den.model_channels / den.num_heads,
              (1...64).contains(decoder.args.num_blocks), (1...64).contains(decoder_gs.args.num_blocks),
              decoder.args.cond_channels == den.in_channels, decoder_gs.args.cond_channels == den.in_channels,
              decoder_gs.args.in_channels == 3, let rep = decoder_gs.args.representation_config,
              rep.num_gaussians == 32, rep.scaling_bias.isFinite, rep.scaling_bias > 0,
              rep.opacity_bias.isFinite, (0..<1).contains(rep.opacity_bias), rep.opacity_bias > 0,
              rep.filterKernelSize.isFinite, rep.filterKernelSize >= 0,
              rep.perturbe_size.isFinite, rep.perturbe_size > 0,
              [rep.lr._xyz, rep.lr._features_dc, rep.lr._scaling, rep.lr._rotation, rep.lr._opacity]
                .allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw IdolError.invalidConfiguration("Incompatible decoders or Gaussian representation.")
        }
    }
}

public struct IdolSamplingSettings: Codable, Sendable, Equatable {
    public let steps: Int
    public let guidance: Float
    public let timeShift: Float
    public let gaussianCount: Int
    public let seed: UInt64

    public init(steps: Int = 50, guidance: Float = 3, timeShift: Float = 3,
                gaussianCount: Int = 262144, seed: UInt64 = 42) throws {
        guard (1...1000).contains(steps), guidance.isFinite, guidance >= 0, guidance <= 100,
              timeShift.isFinite, timeShift > 0, timeShift <= 100,
              (32768...1048576).contains(gaussianCount), gaussianCount % 32 == 0 else {
            throw IdolError.invalidConfiguration("Use 1–1000 steps, finite guidance in [0,100], time shift in (0,100], and 32768–1048576 Gaussians in multiples of 32.")
        }
        self.steps = steps
        self.guidance = guidance
        self.timeShift = timeShift
        self.gaussianCount = gaussianCount
        self.seed = seed
    }

    public var schedule: [Float] {
        (0...steps).map { index in
            let t = Double(steps - index) / Double(steps)
            return Float(Double(timeShift) * t / (1 + (Double(timeShift) - 1) * t))
        }
    }
}
