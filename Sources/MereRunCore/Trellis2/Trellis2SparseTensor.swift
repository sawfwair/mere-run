import Foundation
@preconcurrency import MLX

/// CPU-side coordinate topology paired with Metal-resident feature rows.
/// TRELLIS.2 uses submanifold convolutions, so every convolution at a given
/// level shares this exact neighbor map.
final class Trellis2SparseTopology {
    let coordinates: [Trellis2VoxelCoordinate]
    private let coordinateIndices: [Trellis2VoxelCoordinate: Int]
    private var cachedNeighbors: [Trellis2SparseNeighborPairs]?

    init(coordinates: [Trellis2VoxelCoordinate]) {
        self.coordinates = coordinates
        self.coordinateIndices = Dictionary(
            uniqueKeysWithValues: coordinates.enumerated().map { ($1, $0) }
        )
    }

    var count: Int { coordinates.count }

    /// Source/destination pairs for twenty-seven kernels in checkpoint order:
    /// x, then y, then z. Missing neighbors are omitted so sparse convolution
    /// does not spend a dense matmul on zero rows.
    func convolutionNeighbors() -> [Trellis2SparseNeighborPairs] {
        if let cachedNeighbors { return cachedNeighbors }

        var result = [Trellis2SparseNeighborPairs]()
        result.reserveCapacity(27)
        for kernelX in 0..<3 {
            for kernelY in 0..<3 {
                for kernelZ in 0..<3 {
                    let deltaX = Int32(kernelX - 1)
                    let deltaY = Int32(kernelY - 1)
                    let deltaZ = Int32(kernelZ - 1)
                    var sources = [Int32]()
                    var destinations = [Int32]()
                    sources.reserveCapacity(coordinates.count)
                    destinations.reserveCapacity(coordinates.count)
                    for (destination, coordinate) in coordinates.enumerated() {
                        guard let source = coordinateIndices[
                            coordinate.offset(x: deltaX, y: deltaY, z: deltaZ)
                        ] else { continue }
                        sources.append(Int32(source))
                        destinations.append(Int32(destination))
                    }
                    result.append(Trellis2SparseNeighborPairs(
                        sources: sources,
                        destinations: destinations
                    ))
                }
            }
        }
        cachedNeighbors = result
        return result
    }
}

struct Trellis2SparseNeighborPairs {
    let sources: [Int32]
    let destinations: [Int32]
}

struct Trellis2SparseTensor {
    let features: MLXArray
    let topology: Trellis2SparseTopology

    init(features: MLXArray, coordinates: [Trellis2VoxelCoordinate]) throws {
        guard features.ndim == 2, features.dim(0) == coordinates.count else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[\(coordinates.count), channels] sparse features",
                actual: features.shape
            )
        }
        self.features = features
        self.topology = Trellis2SparseTopology(coordinates: coordinates)
    }

    init(features: MLXArray, topology: Trellis2SparseTopology) throws {
        guard features.ndim == 2, features.dim(0) == topology.count else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[\(topology.count), channels] sparse features",
                actual: features.shape
            )
        }
        self.features = features
        self.topology = topology
    }

    var coordinates: [Trellis2VoxelCoordinate] { topology.coordinates }
    var count: Int { topology.count }
    var channels: Int { features.dim(1) }

    func replacing(features: MLXArray) throws -> Trellis2SparseTensor {
        try Trellis2SparseTensor(features: features, topology: topology)
    }
}

struct Trellis2Subdivision {
    let parentCoordinates: [Trellis2VoxelCoordinate]
    let activeChildren: [[Bool]]

    init(logits: MLXArray, topology: Trellis2SparseTopology) throws {
        guard logits.shape == [topology.count, 8] else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[\(topology.count), 8] subdivision logits",
                actual: logits.shape
            )
        }
        MLX.eval(logits)
        let values = logits.asType(.float32).asArray(Float.self)
        self.parentCoordinates = topology.coordinates
        self.activeChildren = (0..<topology.count).map { parent in
            (0..<8).map { child in values[parent * 8 + child] > 0 }
        }
    }
}

enum Trellis2SparseOperations {
    static func linear(
        _ input: MLXArray,
        weights: Trellis2WeightStore,
        prefix: String
    ) throws -> MLXArray {
        try weights.linear(input, prefix: prefix)
    }

    static func convolution(
        _ input: Trellis2SparseTensor,
        weights: Trellis2WeightStore,
        prefix: String
    ) throws -> Trellis2SparseTensor {
        let weight = try weights.tensor("\(prefix).weight")
        guard weight.ndim == 5,
              weight.dim(1) == 3,
              weight.dim(2) == 3,
              weight.dim(3) == 3,
              weight.dim(4) == input.channels else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[output, 3, 3, 3, \(input.channels)] sparse convolution weight",
                actual: weight.shape
            )
        }

        let typedFeatures = input.features.asType(weight.dtype)
        let neighbors = input.topology.convolutionNeighbors()
        var output = MLX.zeros([input.count, weight.dim(0)], dtype: weight.dtype)
        var kernelIndex = 0
        for kernelX in 0..<3 {
            for kernelY in 0..<3 {
                for kernelZ in 0..<3 {
                    let pairs = neighbors[kernelIndex]
                    guard !pairs.sources.isEmpty else {
                        kernelIndex += 1
                        continue
                    }
                    let gathered = MLX.take(
                        typedFeatures,
                        MLXArray(pairs.sources),
                        axis: 0
                    )
                    let kernel = weight[0..., kernelX, kernelY, kernelZ, 0...]
                    let updates = MLX.matmul(gathered, kernel.transposed())
                    output = output.at[MLXArray(pairs.destinations)].add(updates)
                    kernelIndex += 1
                }
            }
        }
        if let bias = weights.values["\(prefix).bias"] {
            output = output + bias
        }
        return try input.replacing(features: output)
    }

    static func channelToSpatial(
        _ input: Trellis2SparseTensor,
        subdivision: Trellis2Subdivision,
        maximumTokens: Int
    ) throws -> Trellis2SparseTensor {
        guard subdivision.parentCoordinates == input.coordinates else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "subdivision coordinates matching sparse features",
                actual: [subdivision.parentCoordinates.count, input.count]
            )
        }
        guard input.channels.isMultiple(of: 8) else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "sparse channel count divisible by 8",
                actual: input.features.shape
            )
        }

        var coordinates = [Trellis2VoxelCoordinate]()
        var featureRows = [Int32]()
        coordinates.reserveCapacity(min(input.count * 8, maximumTokens))
        featureRows.reserveCapacity(min(input.count * 8, maximumTokens))
        for parentIndex in 0..<input.count {
            let parent = input.coordinates[parentIndex]
            for child in 0..<8 where subdivision.activeChildren[parentIndex][child] {
                coordinates.append(Trellis2VoxelCoordinate(
                    x: parent.x * 2 + Int32(child & 1),
                    y: parent.y * 2 + Int32((child >> 1) & 1),
                    z: parent.z * 2 + Int32((child >> 2) & 1)
                ))
                featureRows.append(Int32(parentIndex * 8 + child))
            }
        }
        guard !coordinates.isEmpty else { throw Trellis2ModelError.emptySparseStructure }
        guard coordinates.count <= maximumTokens else {
            throw Trellis2ModelError.excessiveSparseStructure(
                actual: coordinates.count,
                maximum: maximumTokens
            )
        }

        let reshaped = input.features.reshaped(input.count * 8, input.channels / 8)
        let selected = MLX.take(reshaped, MLXArray(featureRows), axis: 0)
        return try Trellis2SparseTensor(features: selected, coordinates: coordinates)
    }
}
