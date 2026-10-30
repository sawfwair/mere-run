import Foundation
import MediaIO
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class UnlimitedOCRTests: MLXTestCase {
    func testManagedModelHasPinnedOriginalFilesAndNativeManifest() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: UnlimitedOCRResources.modelID))
        XCTAssertEqual(spec.category, .visionOCR)
        XCTAssertEqual(spec.validationKind, .unlimitedOCR)
        XCTAssertEqual(spec.upstreamRepoId, "baidu/Unlimited-OCR-2")
        XCTAssertEqual(spec.upstreamRevision, UnlimitedOCRResources.revision)
        XCTAssertEqual(spec.hubFallback?.patterns, UnlimitedOCRResources.files)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertFalse(UnlimitedOCRResources.files.contains { $0.hasSuffix(".py") })
        let manifest = MereRunModelManifest.template(for: .unlimitedOCR2)
        XCTAssertEqual(manifest.engine, .unlimitedOCR)
        XCTAssertEqual(manifest.supports, [.visionOCR])
        XCTAssertEqual(manifest.family, .ocr)
        let emptyRoot = URL(fileURLWithPath: "/nonexistent-unlimited-ocr-fixture")
        XCTAssertEqual(UnlimitedOCRResources.missingFiles(root: emptyRoot).count, UnlimitedOCRResources.files.count)
    }

    func testSmallImagesSkipTilesAndDocumentPagesChoosePredictableGrids() {
        let small = UnlimitedOCRImage.tileGrid(width: 400, height: 600)
        XCTAssertEqual(small.columns, 1)
        XCTAssertEqual(small.rows, 1)
        let portrait = UnlimitedOCRImage.tileGrid(width: 640, height: 1280)
        XCTAssertEqual(portrait.columns, 1)
        XCTAssertEqual(portrait.rows, 2)
        let landscape = UnlimitedOCRImage.tileGrid(width: 1280, height: 640)
        XCTAssertEqual(landscape.columns, 2)
        XCTAssertEqual(landscape.rows, 1)
    }

    func testLetterboxPadsGrayAndPreservesChannelNormalization() throws {
        let image = try MediaImage(width: 2, height: 1, rgba8: [255, 0, 0, 255, 255, 0, 0, 255])
        let pixels = try UnlimitedOCRImage.letterbox(image).asType(.float32)
        XCTAssertEqual(pixels.shape, [1, 1024, 1024, 3])
        // 127/255 normalized to [-1, 1], then rounded to BF16.
        XCTAssertEqual(pixels[0, 0, 0, 0].item(Float.self), -129 / 32768, accuracy: 1e-7)
        XCTAssertEqual(pixels[0, 512, 512, 0].item(Float.self), 1)
        XCTAssertEqual(pixels[0, 512, 512, 1].item(Float.self), -1)
        XCTAssertEqual(pixels[0, 512, 512, 2].item(Float.self), -1)
    }

    func testLetterboxCentersOddPaddingWithPillowTiesToEvenRounding() throws {
        let image = try MediaImage(width: 1, height: 3, rgba8: [255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255])
        let pixels = try UnlimitedOCRImage.letterbox(image).asType(.float32)
        // Contain produces width 341; pad centers the 683-pixel remainder at
        // round(341.5) == 342 rather than flooring it to 341.
        XCTAssertEqual(pixels[0, 512, 341, 0].item(Float.self), -129 / 32768, accuracy: 1e-7)
        XCTAssertEqual(pixels[0, 512, 342, 0].item(Float.self), 1)
        XCTAssertEqual(pixels[0, 512, 682, 0].item(Float.self), 1)
        XCTAssertEqual(pixels[0, 512, 683, 0].item(Float.self), -129 / 32768, accuracy: 1e-7)
    }
}
