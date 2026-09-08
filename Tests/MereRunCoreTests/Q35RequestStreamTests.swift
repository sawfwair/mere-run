import MLX
import XCTest
@testable import MereRunCore

final class Q35RequestStreamTests: MereRunCoreTestCase {
    func testSequentialRequestsReuseStreamsAfterExecutorHops() async {
        let generator = Q35Generator()
        var firstStream: MLX.Stream?
        for index in 0..<200 {
            let stream = await generator.withRequestStream {
                let stream = StreamOrDevice.default.stream
                let result = MLXArray([Float(index)]) * 2
                await Task.yield()
                XCTAssertEqual(StreamOrDevice.default.stream, stream)
                XCTAssertEqual(result.asArray(Float.self), [Float(index * 2)])
                return stream
            }
            if let firstStream {
                XCTAssertEqual(stream, firstStream, "Sequential requests must not allocate more backend streams")
            } else {
                firstStream = stream
            }
        }
    }

    func testOverlappingRequestsHaveDistinctStreamsAndRestoreParentScope() async {
        let generator = Q35Generator()
        await generator.withRequestStream {
            let parent = StreamOrDevice.default.stream
            await generator.withRequestStream {
                XCTAssertNotEqual(StreamOrDevice.default.stream, parent)
                await Task.yield()
                XCTAssertNotEqual(StreamOrDevice.default.stream, parent)
            }
            XCTAssertEqual(StreamOrDevice.default.stream, parent)
        }
    }

    func testThrownRequestReturnsItsStreamForReuse() async {
        let generator = Q35Generator()
        let first = await generator.withRequestStream { StreamOrDevice.default.stream }
        do {
            try await generator.withRequestStream {
                XCTAssertEqual(StreamOrDevice.default.stream, first)
                throw CancellationError()
            }
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            // Cancellation must release the lease just like successful completion.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        let next = await generator.withRequestStream { StreamOrDevice.default.stream }
        XCTAssertEqual(next, first)
    }
}
