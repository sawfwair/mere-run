import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Opt-in trained-weight checks. See export_4danyone_trace.py for the oracle.
final class FourDAnyoneRealModelTests: MereRunCoreTestCase {
    func testTrainedPoseEncoderMatchesFP32Reference() throws {
        try poseReferenceCheck(dtype: .float32, normalizedRMSE: 1e-4)
    }

    func testTrainedBF16PoseEncoderAgainstFP32Control() throws {
        try poseReferenceCheck(dtype: .bfloat16, normalizedRMSE: 0.03)
    }

    private func poseReferenceCheck(dtype: DType, normalizedRMSE: Float) throws {
        let root = try validationRoot()
        let trace = try reference("transformer", root: root)
        let model = try FourDAnyoneModelLoader.loadPoseEncoder(
            from: root.appendingPathComponent("assets/model.safetensors"), dtype: dtype
        )
        var metrics: [Metric] = []
        metrics.append(compare(
            try model(trace["pose.video"]!), trace["pose.features"]!, name: "pose", normalizedRMSE: normalizedRMSE
        ))
        let nullVideo = MLX.full(trace["pose.video"]![0..<1].shape, values: MLXArray(Float(-1)))
        metrics.append(compare(
            try model(nullVideo), trace["pose.null"]!, name: "nullPose", normalizedRMSE: normalizedRMSE
        ))
        try write(metrics, name: "pose-\(dtype)", root: root)
    }

    func testTrainedVAEEncodeAndDecodeMatchFP32Reference() throws {
        let root = try validationRoot()
        let trace = try reference("vae", root: root)
        let model = Wan2VAEModel()
        let url = root.appendingPathComponent("reference/vae.safetensors")
        let metadata = try SafetensorsStreamingLoader.metadata(url: url)
        // MLX includes the derived normalization constants in parameters();
        // upstream intentionally keeps those two arrays outside its checkpoint.
        let native = Dictionary(uniqueKeysWithValues: model.parameters().flattened()
            .filter { !["latentMean", "latentStandardDeviation"].contains($0.0) }
            .map { ($0, $1.shape) })
        XCTAssertEqual(metadata.mapValues(\.shape), native)
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: url, to: model, dtype: .float32, verify: .noUnusedKeys, batchSize: 8
        )
        eval(model.parameters().flattened().map(\.1))
        var metrics: [Metric] = []
        for frames in [1, 5] {
            let input = trace["video"]![0..., 0..., 0..<frames, 0..., 0...].transposed(0, 2, 3, 4, 1)
            let encoded = model.encodeVideo(input)
            metrics.append(compare(
                encoded.transposed(0, 4, 1, 2, 3), trace["encode\(frames)"]!,
                name: "encode\(frames)", normalizedRMSE: 1e-4
            ))
            // Decode the same reference latents to isolate decoder differences.
            let decoded = model.decode(trace["encode\(frames)"]!.transposed(0, 2, 3, 4, 1))
            metrics.append(compare(
                decoded.transposed(0, 4, 1, 2, 3), trace["decode\(frames)"]!,
                name: "decode\(frames)", normalizedRMSE: 1e-4
            ))
            try MLX.save(array: decoded, url: root.appendingPathComponent("native-vae-decoded-\(frames).npy"))
        }
        try write(metrics, name: "vae-fp32", root: root)
    }

    func testTrainedTransformerMatchesFP32Reference() throws {
        try transformerReferenceCheck(dtype: .float32, normalizedRMSE: 1e-4)
    }

    func testTrainedBF16TransformerAgainstFP32Control() throws {
        // This measures precision drift; it does not claim CUDA/BF16 parity.
        try transformerReferenceCheck(dtype: .bfloat16, normalizedRMSE: 0.03)
    }

    func testTrainedFP32ComputeWithBF16StorageMatchesFP32Reference() throws {
        try transformerReferenceCheck(dtype: .bfloat16, normalizedRMSE: 1e-4, computePrecision: .float32)
    }

    private func transformerReferenceCheck(
        dtype: DType, normalizedRMSE: Float, computePrecision: FourDAnyoneComputePrecision = .model
    ) throws {
        let root = try validationRoot()
        let trace = try reference("transformer", root: root)
        Memory.peakMemory = 0
        let started = Date()
        let model = try FourDAnyoneModelLoader.loadTransformer(
            from: root.appendingPathComponent("assets/model.safetensors"), dtype: dtype,
            computePrecision: computePrecision
        )
        let precisionName = computePrecision == .model ? "\(dtype)" : "\(computePrecision.rawValue)-compute-\(dtype)-storage"
        let loadSeconds = Date().timeIntervalSince(started)
        let context = try FourDAnyoneModelLoader.loadPromptContext(
            from: root.appendingPathComponent("assets/prompt_context.safetensors")
        )
        var metrics: [Metric] = []
        var forwardSeconds: [Double] = []
        for (run, packed) in [false, true, true, true].enumerated() {
            let name = packed ? "packed" : "direct"
            let runName = run > 1 ? "\(name).repeat\(run - 1)" : name
            let input = FourDAnyoneTransformerInput(
                latents: trace["latents"]!, sources: trace["sources"]![0..<(packed ? 5 : 1)],
                poseFeatures: trace["pose.features"]!,
                nullPoseFeatures: MLX.repeated(trace["pose.null"]!, count: packed ? 2 : 1, axis: 0),
                promptContext: context, timestep: 625
            )
            if dtype == .bfloat16, computePrecision == .model, run < 2,
               ProcessInfo.processInfo.environment["MERERUN_4DANYONE_DIAGNOSTIC_STAGES"] == "1" {
                try saveStages(model, input: input, name: name, root: root)
            }
            let runStarted = Date()
            let output = try model(input)
            eval(output)
            forwardSeconds.append(Date().timeIntervalSince(runStarted))
            let referenceKey = dtype == .bfloat16 && computePrecision == .model ? ".bf16_input_control" : ".prediction"
            metrics.append(compare(
                output, trace[name + referenceKey]!, name: runName + ".prediction", normalizedRMSE: normalizedRMSE
            ))
            try MLX.save(array: output, url: root.appendingPathComponent("native-\(runName)-\(precisionName).npy"))
        }
        XCTAssertTrue(model.parameters().flattened().allSatisfy { $0.1.dtype == dtype })
        try write(
            metrics, name: "transformer-\(precisionName)", root: root,
            loadSeconds: loadSeconds, forwardSeconds: forwardSeconds,
            computePrecision: computePrecision.rawValue,
            storedParameterBytes: model.parameters().flattened().reduce(0) { $0 + $1.1.size * $1.1.dtype.size }
        )
    }

    private func saveStages(
        _ model: FourDAnyoneTransformerModel, input: FourDAnyoneTransformerInput, name: String, root: URL
    ) throws {
        let grid = try input.validate(configuration: model.configuration)
        let views = input.latents.dim(0) + input.nullPoseFeatures.dim(0)
        let width = model.configuration.hiddenSize
        let headWidth = width / model.configuration.headCount
        let context = model.textEmbedding(input.promptContext)
        let time = model.embeddedTime(
            timestep: input.timestep, targets: input.latents.dim(0), packed: input.nullPoseFeatures.dim(0)
        )
        let modulation = model.timeProjection(time).reshaped(views, 6, width)
        let spatial = FourDAnyoneRoPE.prepare(grid: grid, headDimension: headWidth)
        let multiview = FourDAnyoneRoPE.prepare(
            grid: Wan2GridSize(frames: views, height: grid.height, width: grid.width), headDimension: headWidth
        )
        var hidden = model.assemble(input, grid: grid)
        var stages = ["assembly": hidden, "time": time, "context": context]
        for (index, block) in model.blocks.enumerated() {
            hidden = block(
                hidden, context: context, time: modulation, grid: grid,
                spatialRoPE: spatial, multiviewRoPE: multiview, maximumQueryTokens: 512
            )
            eval(hidden)
            if [0, 14, 29].contains(index) { stages["block\(index)"] = hidden.asType(.float32) }
        }
        try MLX.save(arrays: stages, url: root.appendingPathComponent("native-\(name)-bf16-stages.safetensors"))
    }

    private func validationRoot() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_4DANYONE_VALIDATION_ROOT"] else {
            throw XCTSkip("Set MERERUN_4DANYONE_VALIDATION_ROOT to the assets and exported reference directory.")
        }
        return URL(fileURLWithPath: path)
    }

    private func reference(_ name: String, root: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: root.appendingPathComponent("reference/\(name)-reference.safetensors"))
    }

    private struct Metric: Codable {
        let name: String
        let shape: [Int]
        let maximumAbsoluteError: Float
        let rootMeanSquareError: Float
        let referenceRootMeanSquare: Float
        let normalizedRootMeanSquareError: Float
        let cosineSimilarity: Float
    }

    private struct Receipt: Codable {
        let device: String
        let metrics: [Metric]
        let loadSeconds: Double
        let forwardSeconds: [Double]
        let peakAllocatedBytes: Int
        let computePrecision: String?
        let storedParameterBytes: Int?
    }

    private func compare(
        _ actual: MLXArray, _ expected: MLXArray, name: String, normalizedRMSE: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) -> Metric {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let error = a - b
        let rmse = MLX.sqrt(MLX.mean(error * error)).item(Float.self)
        let referenceRMS = MLX.sqrt(MLX.mean(b * b)).item(Float.self)
        let relative = rmse / max(referenceRMS, 1e-12)
        let cosine = (MLX.sum(a * b) / MLX.sqrt(MLX.sum(a * a) * MLX.sum(b * b))).item(Float.self)
        XCTAssertTrue(relative.isFinite, name, file: file, line: line)
        XCTAssertLessThanOrEqual(relative, normalizedRMSE, name, file: file, line: line)
        return Metric(
            name: name, shape: actual.shape, maximumAbsoluteError: MLX.max(MLX.abs(error)).item(Float.self),
            rootMeanSquareError: rmse, referenceRootMeanSquare: referenceRMS,
            normalizedRootMeanSquareError: relative, cosineSimilarity: cosine
        )
    }

    private func write(
        _ metrics: [Metric], name: String, root: URL, loadSeconds: Double = 0, forwardSeconds: [Double] = [],
        computePrecision: String? = nil, storedParameterBytes: Int? = nil
    ) throws {
        let receipt = Receipt(
            device: ProcessInfo.processInfo.environment["MERERUN_TEST_MLX_DEVICE"] ?? "cpu",
            metrics: metrics, loadSeconds: loadSeconds, forwardSeconds: forwardSeconds,
            peakAllocatedBytes: Memory.peakMemory, computePrecision: computePrecision,
            storedParameterBytes: storedParameterBytes
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(to: root.appendingPathComponent("native-\(name).json"))
    }
}
