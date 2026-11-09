#if os(macOS)
import Foundation
import MLX
import XCTest
@testable import MereRunCore

extension MuseGlimmerCheckpointQualificationTests {
    func testTiledWarmupStages() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_GLIMMER_QUALIFICATION_ROOT"] else { throw XCTSkip("Set checkpoint root") }
        let lease = try await checkpointAdmission()
        defer { lease.release() }
        let root = URL(fileURLWithPath: path)
        let config = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        let target = try loadTarget(root, config: config)
        let token = MLXArray([Int32(0)]).reshaped(1, 1)
        let cache = target.makeCache()
        var hidden = target.inputEmbeddings(for: token)
        MLX.eval(hidden)
        FileHandle.standardError.write(Data("[tiled-warmup-stage] embeddings\n".utf8))
        for (index, layer) in target.model.languageModel.layers.enumerated() {
            hidden = layer(hidden, cache: cache[index])
            MLX.eval(hidden)
            evaluateGemma4CacheStorage(cache)
            FileHandle.standardError.write(Data("[tiled-warmup-stage] layer=\(index)\n".utf8))
        }
        let logits = target.outputLogits(from: target.model.languageModel.norm(hidden))
        MLX.eval(logits)
        XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
        FileHandle.standardError.write(Data("[tiled-warmup-stage] head\n".utf8))
        let full = target.forward(token, cache: target.makeCache())
        MLX.eval(full.logits)
        XCTAssertTrue(MLX.all(MLX.isFinite(full.logits)).item(Bool.self))
        try record(Record(kind: "tiled-warmup-stages", modelRoot: path, bits: 8, id: "serial-and-full-graph"), env: env)
    }
}
#endif
