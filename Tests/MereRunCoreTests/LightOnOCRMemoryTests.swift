import MLX
import XCTest
@testable import MereRunCore

final class LightOnOCRMemoryTests: MereRunCoreTestCase {
    func testReclaimsUnusedBuffersWithoutChangingLiveTensorsOrAllocatorLimits() {
        Memory.clearCache()
        let originalCacheLimit = Memory.cacheLimit
        let originalMemoryLimit = Memory.memoryLimit
        defer { Memory.clearCache() }

        let live = MLXArray([Float(1), 2, 3, 4]) * 7
        eval(live)
        do {
            let scratch = MLXArray.ones([1_048_576], dtype: .float32) * 3
            eval(scratch)
            XCTAssertEqual(scratch[0].item(Float.self), 3)
        }
        let unused = Memory.cacheMemory
        XCTAssertGreaterThanOrEqual(unused, 4 * 1_024 * 1_024)

        LightOnOCRGenerator.reclaimUnusedDecodeBuffers(cacheThresholdBytes: 1)

        XCTAssertLessThan(Memory.cacheMemory, unused)
        XCTAssertEqual(live.asArray(Float.self), [7, 14, 21, 28])
        XCTAssertEqual(Memory.cacheLimit, originalCacheLimit)
        XCTAssertEqual(Memory.memoryLimit, originalMemoryLimit)
    }
}
