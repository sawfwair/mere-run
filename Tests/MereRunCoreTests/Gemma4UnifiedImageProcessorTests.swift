import Foundation
import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class Gemma4UnifiedImageProcessorTests: MereRunCoreTestCase {
    func testExpandsBareImagePlaceholdersToUnifiedSoftTokens() throws {
        let expanded = try Gemma4UnifiedImageProcessor.expandedPromptTokens(
            [10, 258_880, 20, 258_880, 30],
            softTokenCounts: [2, 3],
            imageTokenId: 258_880,
            boiTokenId: 255_999,
            eoiTokenId: 258_882
        )

        XCTAssertEqual(expanded, [
            10,
            255_999,
            258_880,
            258_880,
            258_882,
            20,
            255_999,
            258_880,
            258_880,
            258_880,
            258_882,
            30,
        ])
    }

    func testAlreadyExpandedPromptIsLeftUnchanged() throws {
        let tokens = [255_999, 258_880, 258_880, 258_882, 42]
        let expanded = try Gemma4UnifiedImageProcessor.expandedPromptTokens(
            tokens,
            softTokenCounts: [2],
            imageTokenId: 258_880,
            boiTokenId: 255_999,
            eoiTokenId: 258_882
        )

        XCTAssertEqual(expanded, tokens)
    }

    func testMultimodalTokenTypesMarkImageTokensOnly() {
        let mmTokenTypeIds = Gemma4UnifiedImageProcessor.mmTokenTypeIds(
            tokens: [255_999, 258_880, 258_880, 258_882, 42],
            imageTokenId: 258_880
        )

        MLX.eval(mmTokenTypeIds)
        XCTAssertEqual(mmTokenTypeIds.shape, [1, 5])
        XCTAssertEqual(mmTokenTypeIds.asArray(Int32.self), [0, 1, 1, 0, 0])
    }

    func testStorageAndCountOnlyPathsAgreeForLocalImage() throws {
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: imageURL) }
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 4,
                height: 2,
                rgba8: Array(repeating: 128, count: 4 * 2 * 4)
            ),
            to: imageURL
        )
        let config = try JSONDecoder().decode(
            Gemma4UnifiedVisionConfig.self,
            from: Data(#"{"patch_size":1,"pooling_kernel_size":2,"model_patch_size":2,"mm_embed_dim":8,"mm_posemb_size":4,"num_soft_tokens":4,"rms_norm_eps":0.000001,"output_proj_dims":8}"#.utf8)
        )

        let references = [imageURL.path]
        let counts = try Gemma4UnifiedImageProcessor.softTokenCounts(
            imageReferences: references,
            visionConfig: config
        )
        let storage = try Gemma4UnifiedImageProcessor.makeStorage(
            imageReferences: references,
            visionConfig: config
        )

        XCTAssertEqual(counts, storage.softTokenCounts)
        XCTAssertEqual(storage.pixelShape, [1, 4, 12])
        XCTAssertEqual(storage.imagePositionShape, [1, 4, 2])
    }

    func testTrainingBatchRejectsImageContentDrift() throws {
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: imageURL) }
        try MediaImageIO.writePNG(
            try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 32, count: 16)),
            to: imageURL
        )
        let config = try JSONDecoder().decode(
            Gemma4UnifiedVisionConfig.self,
            from: Data(#"{"patch_size":1,"pooling_kernel_size":2,"model_patch_size":2,"mm_embed_dim":8,"mm_posemb_size":4,"num_soft_tokens":2,"rms_norm_eps":0.000001,"output_proj_dims":8}"#.utf8)
        )
        let softTokenCount = try XCTUnwrap(Gemma4UnifiedImageProcessor.softTokenCounts(
            imageReferences: [imageURL.path],
            visionConfig: config
        ).first)
        let tokens = [6] + Array(repeating: 5, count: softTokenCount) + [7, 8]
        let example = TextSFTTokenizedExample(
            inputTokenIds: tokens,
            labelTokenIds: Array(tokens.dropFirst()) + [9],
            lossMask: Array(repeating: 0, count: tokens.count - 1) + [1],
            multimodalInputs: TextSFTMultimodalInputs(
                imageReferences: [imageURL.path],
                imageSHA256: [try TextSFTDataset.fileDigest(imageURL)],
                softTokenCounts: [softTokenCount],
                mmTokenTypeIds: tokens.map { $0 == 5 ? 1 : 0 },
                mmTokenTypeShape: [1, tokens.count]
            )
        )

        XCTAssertNoThrow(try Gemma4VLMLoRATrainingPipeline.makeTrainingBatch(
            [example],
            visionConfig: config,
            imageTokenId: 5
        ))

        try MediaImageIO.writePNG(
            try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 224, count: 16)),
            to: imageURL
        )
        XCTAssertThrowsError(try Gemma4VLMLoRATrainingPipeline.makeTrainingBatch(
            [example],
            visionConfig: config,
            imageTokenId: 5
        )) { error in
            XCTAssertTrue(String(describing: error).contains("imageContentChanged"))
        }
    }
}
