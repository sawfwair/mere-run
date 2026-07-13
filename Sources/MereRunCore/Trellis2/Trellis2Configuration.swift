import Foundation

public struct Trellis2FlowConfiguration: Codable, Equatable, Sendable {
    public let resolution: Int
    public let inputChannels: Int
    public let outputChannels: Int
    public let modelChannels: Int
    public let conditionChannels: Int
    public let blockCount: Int
    public let headCount: Int
    public let mlpRatio: Float

    public init(
        resolution: Int,
        inputChannels: Int,
        outputChannels: Int,
        modelChannels: Int,
        conditionChannels: Int,
        blockCount: Int,
        headCount: Int,
        mlpRatio: Float
    ) {
        precondition(modelChannels.isMultiple(of: headCount))
        self.resolution = resolution
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.modelChannels = modelChannels
        self.conditionChannels = conditionChannels
        self.blockCount = blockCount
        self.headCount = headCount
        self.mlpRatio = mlpRatio
    }

    public var headDimension: Int { modelChannels / headCount }
    public var mlpChannels: Int { Int(Float(modelChannels) * mlpRatio) }

    public static let sparseStructure512 = Trellis2FlowConfiguration(
        resolution: 16,
        inputChannels: 8,
        outputChannels: 8,
        modelChannels: 1_536,
        conditionChannels: 1_024,
        blockCount: 30,
        headCount: 12,
        mlpRatio: 5.3334
    )

    public static let shape512 = Trellis2FlowConfiguration(
        resolution: 32,
        inputChannels: 32,
        outputChannels: 32,
        modelChannels: 1_536,
        conditionChannels: 1_024,
        blockCount: 30,
        headCount: 12,
        mlpRatio: 5.3334
    )

    /// The texture flow concatenates the normalized 32-channel shape latent
    /// with 32 channels of texture noise.
    public static let texture512 = Trellis2FlowConfiguration(
        resolution: 32,
        inputChannels: 64,
        outputChannels: 32,
        modelChannels: 1_536,
        conditionChannels: 1_024,
        blockCount: 30,
        headCount: 12,
        mlpRatio: 5.3334
    )
}

public struct Trellis2SamplerConfiguration: Codable, Equatable, Sendable {
    public let steps: Int
    public let sigmaMinimum: Float
    public let guidanceStrength: Float
    public let guidanceRescale: Float
    public let guidanceInterval: ClosedRange<Float>
    public let timestepRescale: Float

    public init(
        steps: Int,
        sigmaMinimum: Float = 1e-5,
        guidanceStrength: Float,
        guidanceRescale: Float,
        guidanceInterval: ClosedRange<Float>,
        timestepRescale: Float
    ) {
        precondition(steps > 0)
        self.steps = steps
        self.sigmaMinimum = sigmaMinimum
        self.guidanceStrength = guidanceStrength
        self.guidanceRescale = guidanceRescale
        self.guidanceInterval = guidanceInterval
        self.timestepRescale = timestepRescale
    }

    public static let sparseStructure = Trellis2SamplerConfiguration(
        steps: 12,
        guidanceStrength: 7.5,
        guidanceRescale: 0.7,
        guidanceInterval: 0.6...1.0,
        timestepRescale: 5.0
    )

    public static let shape = Trellis2SamplerConfiguration(
        steps: 12,
        guidanceStrength: 7.5,
        guidanceRescale: 0.5,
        guidanceInterval: 0.6...1.0,
        timestepRescale: 3.0
    )

    public static let texture = Trellis2SamplerConfiguration(
        steps: 12,
        guidanceStrength: 1.0,
        guidanceRescale: 0,
        guidanceInterval: 0.6...0.9,
        timestepRescale: 3.0
    )

    public var timesteps: [Float] {
        (0...steps).map { index in
            let linear = 1 - Float(index) / Float(steps)
            return timestepRescale * linear / (1 + (timestepRescale - 1) * linear)
        }
    }
}

public enum Trellis2Normalization {
    public static let shapeMean: [Float] = [
        0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
        -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
        0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
        -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665,
    ]

    public static let shapeStandardDeviation: [Float] = [
        5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
        5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
        4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
        5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
    ]

    public static let textureMean: [Float] = [
        3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
        0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
        -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
        1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862,
    ]

    public static let textureStandardDeviation: [Float] = [
        2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
        2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
        2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
        2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190,
    ]
}

public struct Trellis2VoxelCoordinate: Codable, Hashable, Sendable {
    public let x: Int32
    public let y: Int32
    public let z: Int32

    public init(x: Int32, y: Int32, z: Int32) {
        self.x = x
        self.y = y
        self.z = z
    }

    public subscript(axis: Int) -> Int32 {
        switch axis {
        case 0: x
        case 1: y
        default: z
        }
    }

    public func offset(x deltaX: Int32, y deltaY: Int32, z deltaZ: Int32) -> Trellis2VoxelCoordinate {
        Trellis2VoxelCoordinate(x: x + deltaX, y: y + deltaY, z: z + deltaZ)
    }
}
