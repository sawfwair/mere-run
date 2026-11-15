import Foundation
import MLX
import MLXFast
#if canImport(Metal)
import Metal
#endif

/// Weight-reusing Q8 GEMV for small, sorted expert batches. Packed parameters
/// are passed on every call, so replacing weights never leaves a stale capture.
final class Gemma4MetalExperts {
    static func supportsAutomaticReduction(deviceName: String?, architecture: String?) -> Bool {
        deviceName == "Apple M5 Max" && architecture == "applegpu_g17s"
    }

    static var automaticDistributedReduction: Bool {
        #if os(macOS) && canImport(Metal)
        guard isSupported, let device = MTLCreateSystemDefaultDevice() else { return false }
        return supportsAutomaticReduction(deviceName: device.name, architecture: GPU.deviceInfo().architecture)
        #else
        false
        #endif
    }

    static var isSupported: Bool {
        #if canImport(Metal)
        Device.defaultDevice().deviceType == .gpu
        #else
        false
        #endif
    }

    private let kernel: MLXFast.MLXFastKernel
    private let reductionKernel: MLXFast.MLXFastKernel
    private let activationKernel: MLXFast.MLXFastKernel
    private let compilationIdentities = (0..<16).map { _ in NSObject() }
    var byteLoads = true
    var optimizedLayout = true
    var fusedActivation = false
    var distributedReduction = Gemma4MetalExperts.automaticDistributedReduction
    var compilationIdentity: NSObject {
        compilationIdentities[(byteLoads ? 1 : 0) + (optimizedLayout ? 2 : 0) + (fusedActivation ? 4 : 0) + (distributedReduction ? 8 : 0)]
    }

    init() throws {
        let url = Bundle.module.url(forResource: "Gemma4Q8Experts", withExtension: "metal", subdirectory: "Kernels")!
        kernel = MLXFast.metalKernel(name: "gemma4_q8_expert_reuse",
            inputNames: ["input", "weights", "scales", "biases", "indices"], outputNames: ["output"],
            source: try String(contentsOf: url, encoding: .utf8))
        let reductionURL = Bundle.module.url(forResource: "Gemma4Q8Distributed", withExtension: "metal", subdirectory: "Kernels")!
        reductionKernel = MLXFast.metalKernel(name: "gemma4_q8_distributed",
            inputNames: ["input", "weights", "scales", "biases", "indices"], outputNames: ["output"],
            source: try String(contentsOf: reductionURL, encoding: .utf8))
        let activationURL = Bundle.module.url(forResource: "Gemma4Q8GLU", withExtension: "metal", subdirectory: "Kernels")!
        activationKernel = MLXFast.metalKernel(name: "gemma4_q8_glu",
            inputNames: ["input", "weights", "scales", "biases", "indices"], outputNames: ["output"],
            source: try String(contentsOf: activationURL, encoding: .utf8))
    }

    func callAsFunction(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, indices: MLXArray) -> MLXArray {
        let routes = x.dim(0), width = x.dim(2), outputs = weight.dim(1)
        let columns = optimizedLayout && outputs.isMultiple(of: 16) ? 8 : 4
        let tiles = outputs / (2 * columns)
        let grid = optimizedLayout ? (routes * 64, tiles, 1) : (tiles * 64, routes, 1)
        let projection = distributedReduction && columns == 8 ? reductionKernel : kernel
        return projection([x, weight, scales, biases, indices],
            template: [("R", routes), ("N", outputs), ("K", width), ("Rows", 2), ("O", columns), ("SG", 2),
                       ("ByteLoads", byteLoads), ("RouteFirst", optimizedLayout),
                       ("VPT", width.isMultiple(of: 256) ? 8 : 4)],
            grid: grid, threadGroup: (64, 1, 1),
            outputShapes: [[routes, 1, outputs]], outputDTypes: [x.dtype])[0]
    }

    func activation(_ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, indices: MLXArray) -> MLXArray {
        let routes = x.dim(0), width = x.dim(2), outputs = weight.dim(1)
        return activationKernel([x, weight, scales, biases, indices],
            template: [("R", routes), ("N", outputs), ("K", width), ("Rows", 2), ("O", 4), ("SG", 2),
                       ("ByteLoads", byteLoads), ("RouteFirst", true), ("VPT", width.isMultiple(of: 256) ? 8 : 4)],
            grid: (routes * 64, outputs / 16, 1), threadGroup: (64, 1, 1),
            outputShapes: [[routes, 1, outputs / 2]], outputDTypes: [x.dtype])[0]
    }
}
