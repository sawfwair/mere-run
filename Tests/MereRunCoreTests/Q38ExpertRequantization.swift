import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Research-only, in-memory conversion. The published Q4 source stays untouched.
enum Q38ExpertRequantization: String, Sendable {
    case q4
    case q3Group64 = "q3-g64"
    case q3ActivationRefit = "q3-g64-activation-refit"
    case q2Group32 = "q2-g32"

    var bits: Int { self == .q4 ? 4 : self == .q2Group32 ? 2 : 3 }
    var groupSize: Int { self == .q2Group32 ? 32 : 64 }

    func arrays(from expert: Q35SwitchLinear) throws -> (MLXArray, MLXArray, MLXArray) {
        let scales = try XCTUnwrap(expert.scales)
        let biases = try XCTUnwrap(expert.biases)
        XCTAssertEqual(expert.weight.ndim, 3)
        let dense = MLX.dequantized(
            expert.weight, scales: scales, biases: biases,
            groupSize: expert.groupSize, bits: expert.bits, dtype: .bfloat16
        )
        let result = MLX.quantized(dense, groupSize: groupSize, bits: bits)
        let resultBiases = try XCTUnwrap(result.2)
        MLX.eval(result.0, result.1, resultBiases)
        return (result.0, result.1, resultBiases)
    }

    func transform(_ model: Q35Model, profileURL: URL? = nil) throws {
        guard self != .q4 else { return }
        XCTAssertTrue(model.config.textConfig.isQwen4Exp)
        var count = 0
        var sourceBytes = 0
        var candidateBytes = 0
        // Mutate each module in place so the enumeration does not retain old
        // tensors. Evaluation and cache clearing bound transient conversion RAM.
        for (path, module) in model.leafModules().flattened().sorted(by: { $0.0 < $1.0 }) {
            guard let expert = module as? Q35SwitchLinear else { continue }
            XCTAssertTrue(path.contains(".mlp.switch_mlp."), path)
            XCTAssertEqual(expert.bits, 4, "This experiment requires a Q4 source")
            XCTAssertEqual(expert.groupSize, 64)
            sourceBytes += expert.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
            var converted = try arrays(from: expert)
            if self == .q3ActivationRefit {
                let originalScale = converted.1
                let originalBias = converted.2
                let profile = try SafetensorsStreamingLoader.loadArrays(
                    url: XCTUnwrap(profileURL), where: { $0.hasPrefix(path + ".") }
                )
                converted = try Q38ActivationWeightedRefit.arrays(
                    source: expert, candidate: converted,
                    importance: Q38ExpertActivationProfile.importance(profile, path: path)
                )
                let changed = ((converted.1 .!= originalScale) .|| (converted.2 .!= originalBias)).sum().item(Int.self)
                FileHandle.standardError.write(Data(
                    "[q38-affine-refit] module=\(path) changed_groups=\(changed) total_groups=\(converted.1.size)\n".utf8
                ))
            }
            let (weight, scales, biases) = converted
            try expert.update(parameters: ModuleParameters.unflattened([
                "weight": weight, "scales": scales, "biases": biases,
            ]), verify: .none)
            candidateBytes += weight.nbytes + scales.nbytes + biases.nbytes
            count += 1
            Memory.clearCache()
            let line = "[q38-requantize] recipe=\(rawValue) module=\(path) count=\(count) "
                + "active_bytes=\(Memory.activeMemory)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        XCTAssertEqual(count, model.config.textConfig.numHiddenLayers * 3)
        let line = "[q38-requantize-complete] recipe=\(rawValue) modules=\(count) "
            + "source_bytes=\(sourceBytes) candidate_bytes=\(candidateBytes)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
