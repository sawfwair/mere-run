import Foundation
import XCTest
import MLX
import MLXRandom
import MereRunMLXTestSupport
@testable import MereRunNemotronBoltModel

final class NemotronBoltRuntimeTests: MLXTestCase {
    func testPinnedHeterogeneousConfig() throws {
        let root = Bundle.module.resourceURL!.appendingPathComponent("Fixtures")
        let config = try NemotronBoltConfig.load(at: root)
        XCTAssertEqual(config.language.blocks.count, 88)
        XCTAssertEqual(config.language.blocks[1].expertCount, 240)
        XCTAssertEqual(config.language.blocks[1].moeIntermediateSize, 2_048)
        XCTAssertEqual(config.language.blocks[87].expertCount, 368)
        XCTAssertEqual(config.language.blocks[87].moeIntermediateSize, 2_560)
        XCTAssertEqual(config.language.blocks.filter { $0.blockType == "attention" }.count, 8)
    }

    func testIQ2ScaleSignsAndGrid() {
        var bytes = [UInt8](repeating: 0, count: 66)
        bytes[1] = 0x3c // FP16 one
        let positive = BoltPackedMatrix.decodeBlock(bytes, format: .iq2XXS)
        XCTAssertEqual(positive, [Float](repeating: 1, count: 256))
        bytes[2] = 1 // grid one: 43 followed by seven 8s
        bytes[6] = 1 // negative element zero and parity sign at seven
        bytes[9] = 0xf0 // maximum subgroup multiplier
        let decoded = BoltPackedMatrix.decodeBlock(bytes, format: .iq2XXS)
        XCTAssertEqual(decoded[0], -166.625)
        XCTAssertEqual(decoded[1], 31)
        XCTAssertEqual(decoded[7], -31)
        XCTAssertEqual(decoded[32], 1)
    }

    func testQ8SignedValues() {
        let values: [UInt8] = [0, 0x38] + (0..<32).map { UInt8(truncatingIfNeeded: $0 - 16) }
        XCTAssertEqual(BoltPackedMatrix.decodeBlock(values, format: .q8_0), (0..<32).map { Float($0 - 16) * 0.5 })
    }

    func testPackedMetalMatchesScalarAndRoutedInputs() throws {
        for format in [BoltPackedFormat.iq2XXS, .q8_0] {
            let k = format.blockSize * 2
            let o = 4
            let expertCount = 3
            var bytes = [UInt8](repeating: 0, count: expertCount * o * 2 * format.blockBytes)
            for block in 0..<(expertCount * o * 2) {
                let base = block * format.blockBytes
                bytes[base] = 0
                bytes[base + 1] = 0x38
                for offset in 2..<format.blockBytes { bytes[base + offset] = UInt8(truncatingIfNeeded: block * 31 + offset * 13) }
            }
            let matrix = try BoltPackedMatrix(weight: MLXArray(bytes, [expertCount, o, 2, format.blockBytes]), format: format, inputSize: k, outputSize: o)
            let x = (0..<(2 * k)).map { Float(($0 % 19) - 9) / 32 }
            let input = MLXArray(x, [1, 2, k])
            let ids: [Int32] = [2, 0, 1, 2]
            let indices = MLXArray(ids, [1, 2, 2])
            let output = matrix(input, indices: indices).asArray(Float.self)
            var expected: [Float] = []
            for route in 0..<4 {
                for row in 0..<o {
                    var sum: Float = 0
                    for block in 0..<2 {
                        let base = (Int(ids[route]) * o * 2 + row * 2 + block) * format.blockBytes
                        let decoded = BoltPackedMatrix.decodeBlock(Array(bytes[base..<(base + format.blockBytes)]), format: format)
                        for j in 0..<format.blockSize { sum += x[(route / 2) * k + block * format.blockSize + j] * decoded[j] }
                    }
                    expected.append(sum)
                }
            }
            for (actual, reference) in zip(output, expected) { XCTAssertEqual(actual, reference, accuracy: 0.003) }
            // The down projection receives one independent vector per route.
            let expanded = repeated(input.expandedDimensions(axis: -2), count: 2, axis: -2)
            let expandedOutput = matrix(expanded, indices: indices).asArray(Float.self)
            for (actual, reference) in zip(expandedOutput, expected) { XCTAssertEqual(actual, reference, accuracy: 0.003) }
        }
    }

    func testRouterBiasSelectsButDoesNotWeight() throws {
        let route = BoltRouter.route(MLXArray([Float(0)], [1, 1, 1]),
            weight: MLXArray.zeros([3, 1]), bias: MLXArray([Float(0), 2, 1]), topK: 2, scale: 5)
        XCTAssertEqual(Set(route.0.asArray(Int32.self)), Set([1, 2]))
        XCTAssertEqual(route.1.asArray(Float.self), [2.5, 2.5])
    }
}

extension NemotronBoltRuntimeTests {
    func testReleasedPackedRowsAgainstPinnedGGMLReference() throws {
        struct Fixture: Decodable {
            let format: String
            let columns: Int
            let rows: Int
            let bytes: [UInt8]
            let expected: [Float]
        }
        let url = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/released-rows.json")
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
        for fixture in fixtures {
            let format: BoltPackedFormat = fixture.format == "iq2" ? .iq2XXS : .q8_0
            let matrix = try BoltPackedMatrix(weight: MLXArray(fixture.bytes,
                [fixture.rows, fixture.columns / format.blockSize, format.blockBytes]),
                format: format, inputSize: fixture.columns, outputSize: fixture.rows)
            let x = (0..<fixture.columns).map { Float(($0 * 7) % 31 - 15) / 64 }
            let values = matrix(MLXArray(x, [1, 1, fixture.columns])).asArray(Float.self)
            for (actual, expected) in zip(values, fixture.expected) {
                XCTAssertEqual(actual, expected, accuracy: 1e-5)
            }
        }
    }

    func testCheckpointRetainsScaleBytesAndRejectsMalformedRanges() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("model.safetensors")
        func write(_ header: String, payload: [UInt8]) throws {
            let json = Data(header.utf8)
            var size = UInt64(json.count).littleEndian
            var data = withUnsafeBytes(of: &size) { Data($0) }
            data.append(json)
            data.append(contentsOf: payload)
            try data.write(to: url)
        }
        try write(#"{"scale":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}"#, payload: [0x38, 0x40])
        let arrays = try BoltCheckpoint.loadShard(at: url)
        XCTAssertEqual(arrays["scale"]?.dtype, .uint8)
        XCTAssertEqual(arrays["scale"]?.asArray(UInt8.self), [0x38, 0x40])
        try write(#"{"scale":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,3]}}"#, payload: [0, 1, 2])
        XCTAssertThrowsError(try BoltCheckpoint.loadShard(at: url))
        try write(#"{"scale":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[1,3]}}"#, payload: [0, 1, 2])
        XCTAssertThrowsError(try BoltCheckpoint.loadShard(at: url))
        try write(#"{"scale":{"dtype":"UNKNOWN","shape":[2],"data_offsets":[0,2]}}"#, payload: [0, 1])
        XCTAssertThrowsError(try BoltCheckpoint.loadShard(at: url))
    }
}

extension NemotronBoltRuntimeTests {
    func testMetalSSMScanMatchesPortableRecurrenceAndChunking() throws {
        #if os(macOS)
        Device.withDefaultDevice(.gpu) {
            MLXRandom.seed(3_505)
            let x = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 5, 2, 4])
            let b = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 5, 1, 32])
            let c = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 5, 1, 32])
            let dt = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 5, 2])
            let aLog = MLXArray([0.0, 0.2] as [Float])
            let d = MLXArray([0.75, 1.25] as [Float])
            let dtBias = MLXArray([-0.1, 0.1] as [Float])

            let portable = boltSSMUpdate(
                x: x,
                b: b,
                c: c,
                dt: dt,
                aLog: aLog,
                d: d,
                dtBias: dtBias,
                state: nil,
                minimumTimeStep: 0.001,
                maximumTimeStep: .greatestFiniteMagnitude,
                useMetalKernel: false
            )
            let first = boltSSMUpdate(
                x: x[0..., ..<3, 0..., 0...],
                b: b[0..., ..<3, 0..., 0...],
                c: c[0..., ..<3, 0..., 0...],
                dt: dt[0..., ..<3, 0...],
                aLog: aLog,
                d: d,
                dtBias: dtBias,
                state: nil,
                minimumTimeStep: 0.001,
                maximumTimeStep: .greatestFiniteMagnitude
            )
            let second = boltSSMUpdate(
                x: x[0..., 3..., 0..., 0...],
                b: b[0..., 3..., 0..., 0...],
                c: c[0..., 3..., 0..., 0...],
                dt: dt[0..., 3..., 0...],
                aLog: aLog,
                d: d,
                dtBias: dtBias,
                state: first.1,
                minimumTimeStep: 0.001,
                maximumTimeStep: .greatestFiniteMagnitude
            )
            let chunked = MLX.concatenated([first.0, second.0], axis: 1)
            MLX.eval(portable.0, portable.1, chunked, second.1)

            XCTAssertLessThan(
                MLX.max(MLX.abs(portable.0.asType(.float32) - chunked.asType(.float32)))
                    .item(Float.self),
                0.002
            )
            XCTAssertLessThan(
                MLX.max(MLX.abs(portable.1 - second.1)).item(Float.self),
                0.002
            )
        }
        #endif
    }

}

extension NemotronBoltRuntimeTests {
    func testReleasedNVFP4RowsAgainstScalarReference() throws {
        struct Fixture: Decodable {
            let parts: [String: [UInt8]]
            let expected: [Float]
            let global_scale: Float
        }
        let url = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/nvfp4-rows.json")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let weight = try XCTUnwrap(fixture.parts["weight"])
        let scales = try XCTUnwrap(fixture.parts["weight_scale"])
        var store = BoltWeightStore(arrays: [
            "test.weight": MLXArray(weight, [4, 2_048]),
            "test.weight_scale": MLXArray(scales, [4, 256]),
            "test.weight_scale_2": MLXArray(fixture.global_scale),
        ])
        let linear = try BoltLinear(store: &store, prefix: "test", input: 4_096, output: 4, format: nil)
        let x = (0..<4_096).map { Float(($0 * 7) % 31 - 15) / 64 }
        let actual = linear(MLXArray(x, [1, 1, 4_096])).asArray(Float.self)
        for (value, expected) in zip(actual, fixture.expected) {
            XCTAssertEqual(value, expected, accuracy: 1e-5)
        }
    }
}
