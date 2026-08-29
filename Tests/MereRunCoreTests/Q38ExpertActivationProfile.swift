import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Test-only teacher observations. The generator actor serializes observations;
/// the test switches modality only between awaited requests.
final class Q38ExpertActivationProfile: @unchecked Sendable {
    enum Modality: String, CaseIterable {
        case image, text
    }

    struct Moments {
        var squaredSum: MLXArray
        var count: MLXArray

        init(experts: Int, input: Int) {
            squaredSum = MLXArray.zeros([experts, input], dtype: .float32)
            count = MLXArray.zeros([experts], dtype: .float32)
        }

        mutating func observe(_ rows: MLXArray, indices: MLXArray) {
            let values = rows.asType(.float32)
            squaredSum = squaredSum.at[indices].add(values * values)
            count = count.at[indices].add(MLXArray.ones(indices.shape, dtype: .float32))
            // Break the lazy observation graph at each layer/chunk, rather than
            // retaining an entire teacher run's intermediate activations.
            MLX.eval(squaredSum, count)
        }
    }

    var modality = Modality.text
    private var modules: [String: [Modality: Moments]] = [:]

    func install(on model: Q35Model) {
        precondition(!Q35FusedSwitchGLUPolicy.enabled, "Teacher profiling requires unfused gate/up projections")
        var replacements: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened() {
            guard let expert = module as? Q35SwitchLinear else { continue }
            precondition(path.contains(".mlp.switch_mlp."))
            replacements.append((path, ObservedExpert(source: expert, path: path, profile: self)))
        }
        // Replacing an array-backed module tree requires all array positions;
        // a sparse single-layer update introduces unsupported empty entries.
        model.update(modules: ModuleChildren.unflattened(replacements))
    }

    func observe(path: String, experts: Int, rows: MLXArray, indices: MLXArray) {
        var moments = modules[path]?[modality] ?? Moments(experts: experts, input: rows.dim(-1))
        moments.observe(rows, indices: indices)
        modules[path, default: [:]][modality] = moments
    }

    func save(to url: URL, metadata: [String: String]) throws {
        precondition(!FileManager.default.fileExists(atPath: url.path), "Do not replace calibration evidence")
        let arrays = modules.reduce(into: [String: MLXArray]()) { result, entry in
            for (modality, moments) in entry.value {
                result["\(entry.key).\(modality.rawValue).squared_sum"] = moments.squaredSum
                result["\(entry.key).\(modality.rawValue).count"] = moments.count
            }
        }
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
        let observed = modules.values.reduce(0) { total, modalities in
            total + modalities.values.reduce(0) { $0 + ($1.count .> 0).sum().item(Int.self) }
        }
        FileHandle.standardError.write(Data(
            "[q38-activation-profile] modules=\(modules.count) observed_expert_modalities=\(observed)\n".utf8
        ))
    }

    static func importance(_ arrays: [String: MLXArray], path: String) throws -> MLXArray {
        let means = try Modality.allCases.map { modality in
            let prefix = "\(path).\(modality.rawValue)"
            let sum = try XCTUnwrap(arrays[prefix + ".squared_sum"])
            let count = try XCTUnwrap(arrays[prefix + ".count"])
            return sum / MLX.maximum(count, 1).expandedDimensions(axis: -1)
        }
        // Each modality contributes a mean, so hundreds of image tokens do not
        // automatically outweigh the text calibration requests. Overall scaling
        // of an expert's objective does not affect its affine optimum.
        return means[0] + means[1]
    }

    final class ObservedExpert: Q35SwitchLinear {
        private let path: String
        private let profile: Q38ExpertActivationProfile

        init(source: Q35SwitchLinear, path: String, profile: Q38ExpertActivationProfile) {
            self.path = path
            self.profile = profile
            super.init(weight: source.weight, scales: source.scales, biases: source.biases,
                       bias: source.bias, groupSize: source.groupSize, bits: source.bits)
        }

        override func callAsFunction(_ input: MLXArray, indices: MLXArray) -> MLXArray {
            if input.ndim == 3 {
                let topK = indices.dim(-1)
                let rows = MLX.repeated(input.reshaped([-1, input.dim(-1)]), count: topK, axis: 0)
                profile.observe(path: path, experts: weight.dim(0), rows: rows, indices: indices.flattened())
            }
            // Four-dimensional down-projection inputs delegate to applyFlat;
            // observing them here too would double-count decode activations.
            return super.callAsFunction(input, indices: indices)
        }

        override func applyFlat(_ input: MLXArray, indices: MLXArray, sortedIndices: Bool) -> MLXArray {
            profile.observe(path: path, experts: weight.dim(0),
                            rows: input.reshaped([-1, input.dim(-1)]), indices: indices.flattened())
            return super.applyFlat(input, indices: indices, sortedIndices: sortedIndices)
        }
    }
}
