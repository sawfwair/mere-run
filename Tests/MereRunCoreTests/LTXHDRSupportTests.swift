import Foundation
import MLX
import MediaIO
@testable import MereRunCore
import XCTest

final class LTXHDRSupportTests: XCTestCase {
    func testWorkingSpaceTransfersRoundTripSceneLinearValues() {
        let values: [Float] = [0, 0.001, 0.01, 0.18, 1, 4, 16]
        for transfer in LTXHDRTransfer.allCases {
            let encoded = LTXHDRColorPipeline.compress(values, transfer: transfer)
            let decoded = LTXHDRColorPipeline.decompress(encoded, transfer: transfer)
            for (actual, expected) in zip(decoded, values) {
                XCTAssertEqual(
                    actual,
                    expected,
                    accuracy: max(0.000_02, abs(expected) * 0.000_02),
                    "\(transfer.rawValue) failed at \(expected)"
                )
            }
        }
    }

    func testHDRDecodePreservesUnboundedLinearEXRAndBuildsHLGSignal() {
        let workingCodes: [Float] = [
            0.5, 0.75, 1.0,
            0.25, 0.5, 0.75,
        ]
        let decoded = (MLXArray(workingCodes)
            .reshaped(1, 1, 2, 3)
            .transposed(0, 3, 1, 2)
            .reshaped(1, 3, 1, 1, 2) * MLXArray(Float(2))) - MLXArray(Float(1))

        let output = LTXHDRColorPipeline.decode(
            decoded,
            transfer: .acesCCT,
            exrColorSpace: .acescg
        )
        MLX.eval(output.working, output.exr, output.hlg)

        XCTAssertEqual(output.working.shape, [1, 1, 2, 3])
        XCTAssertEqual(output.exr.shape, [1, 1, 2, 3])
        XCTAssertEqual(output.hlg.shape, [1, 1, 2, 3])
        XCTAssertGreaterThan(output.exr.max().item(Float.self), 100)
        let hlg = output.hlg.asArray(Float.self)
        XCTAssertTrue(hlg.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testLinearRec709ConditioningUsesACEScctWorkingSpace() throws {
        let image = try MediaFloatImage(
            width: 1,
            height: 1,
            rgb: [1, 0.18, 0]
        )
        let conditioning = LTXHDRColorPipeline.makeConditioningImage(
            image,
            colorSpace: .srgbLinear,
            dtype: .float32
        )
        MLX.eval(conditioning)

        XCTAssertEqual(conditioning.shape, [1, 3, 1, 1, 1])
        let values = conditioning.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0 >= -1 && $0 <= 1 })
        XCTAssertNotEqual(values[0], values[1])
    }

    func testHDRICLoRAMetadataAndOfficialStageTwoDefaults() throws {
        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("hdr.safetensors")
        let header = """
        {
          "__metadata__": {
            "hdr_transform": "logc3",
            "reference_downscale_factor": "2"
          },
          "transformer_blocks.0.attn1.to_q.lora_A.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4]
          }
        }
        """
        var data = Data()
        let headerData = Data(header.utf8)
        var headerSize = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerData)
        data.append(Data(repeating: 0, count: 4))
        try data.write(to: url)

        let configuration = try XCTUnwrap(
            ltxHDRLoRAConfiguration(LTXLoRAConfiguration(url: url))
        )
        XCTAssertEqual(configuration.hdrTransform, .logC3)
        XCTAssertEqual(configuration.referenceDownscaleFactor, 2)

        let options = LTXHDRICLoRAOptions(highQuality: true)
        XCTAssertTrue(options.highQuality)
        XCTAssertEqual(options.stage2Phases.count, 1)
        XCTAssertEqual(options.stage2Phases[0].sigmas, [0.909375, 0.725, 0])
        XCTAssertEqual(options.stage2Phases[0].tiling.frameTiles, 2)
        XCTAssertEqual(options.stage2Phases[0].tiling.frameOverlap, 8)
        XCTAssertEqual(options.stage2Phases[0].tiling.heightOverlap, 6)
        XCTAssertTrue(options.stage2Phases[0].usesICLoRAConditioning)
    }

    func testStackedICLoRAReferenceScalesResolveNeutralMetadataAndRejectConflicts() throws {
        let directory = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: directory) }
        let neutral = directory.appendingPathComponent("neutral.safetensors")
        let scaled = directory.appendingPathComponent("scaled.safetensors")
        let conflicting = directory.appendingPathComponent("conflicting.safetensors")
        try MLX.save(
            arrays: ["placeholder": MLXArray([Float(0)])],
            metadata: ["reference_downscale_factor": "1"],
            url: neutral
        )
        try MLX.save(
            arrays: ["placeholder": MLXArray([Float(0)])],
            metadata: [
                "reference_downscale_factor": "2",
                "reference_temporal_scale_factor": "3",
            ],
            url: scaled
        )
        try MLX.save(
            arrays: ["placeholder": MLXArray([Float(0)])],
            metadata: ["reference_downscale_factor": "4"],
            url: conflicting
        )

        let resolved = try ltxLoRAReferenceScaleConfiguration([
            LTXLoRAConfiguration(url: neutral),
            LTXLoRAConfiguration(url: scaled),
        ])
        XCTAssertEqual(resolved.downscaleFactor, 2)
        XCTAssertEqual(resolved.temporalScaleFactor, 3)
        XCTAssertThrowsError(
            try ltxLoRAReferenceScaleConfiguration([
                LTXLoRAConfiguration(url: scaled),
                LTXLoRAConfiguration(url: conflicting),
            ])
        ) { error in
            guard case LTXLoRAReferenceScaleMetadataError.conflictingValues(
                key: "reference_downscale_factor",
                first: 2,
                second: 4,
                url: conflicting
            ) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }
}
