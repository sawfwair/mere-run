import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class TrainingDataCacheTests: MereRunCoreTestCase {

    func testRoundTripWithoutReferenceLatents() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let cache = TrainingDataCache(cacheDir: temp.appendingPathComponent("cache", isDirectory: true))
        try cache.initialize(wipe: true)

        let latents = MLXArray(Array(repeating: Float(1.0), count: 8), [1, 2, 2, 2]).asType(.bfloat16)
        let cond = MLXArray(Array(repeating: Float(2.0), count: 12), [1, 3, 4]).asType(.bfloat16)

        try cache.save(id: 7, latents: latents, cond: cond, width: 256, height: 192)

        let loaded = try cache.load(id: 7)
        XCTAssertEqual(loaded.width, 256)
        XCTAssertEqual(loaded.height, 192)
        XCTAssertNil(loaded.referenceLatents)
        XCTAssertEqual(loaded.latents.shape, latents.shape)
        XCTAssertEqual(loaded.cond.shape, cond.shape)
        XCTAssertEqual(loaded.latents[0, 0, 0, 0].item(Float.self), 1.0, accuracy: 1e-3)
        XCTAssertEqual(loaded.cond[0, 0, 0].item(Float.self), 2.0, accuracy: 1e-3)
    }

    func testRoundTripWithReferenceLatents() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let cache = TrainingDataCache(cacheDir: temp.appendingPathComponent("cache", isDirectory: true))
        try cache.initialize(wipe: true)

        let latents = MLXArray(Array(repeating: Float(3.0), count: 8), [1, 2, 2, 2]).asType(.bfloat16)
        let cond = MLXArray(Array(repeating: Float(4.0), count: 12), [1, 3, 4]).asType(.bfloat16)
        let reference = MLXArray(Array(repeating: Float(5.0), count: 8), [1, 2, 2, 2]).asType(.bfloat16)

        try cache.save(
            id: 11,
            latents: latents,
            cond: cond,
            referenceLatents: reference,
            width: 384,
            height: 384
        )

        let loaded = try cache.load(id: 11)
        XCTAssertEqual(loaded.width, 384)
        XCTAssertEqual(loaded.height, 384)
        XCTAssertNotNil(loaded.referenceLatents)
        XCTAssertEqual(loaded.latents.shape, latents.shape)
        XCTAssertEqual(loaded.cond.shape, cond.shape)
        XCTAssertEqual(loaded.referenceLatents?.shape, reference.shape)
        XCTAssertEqual(loaded.latents[0, 0, 0, 0].item(Float.self), 3.0, accuracy: 1e-3)
        XCTAssertEqual(loaded.cond[0, 0, 0].item(Float.self), 4.0, accuracy: 1e-3)
        guard let loadedReference = loaded.referenceLatents else {
            XCTFail("Expected reference latents to be present.")
            return
        }
        XCTAssertEqual(loadedReference[0, 0, 0, 0].item(Float.self), 5.0, accuracy: 1e-3)
    }
}
