@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import XCTest

/// Earth's Map view in the registry: the input column draws an Earth run's result as a map,
/// the result panel keeps the tensor's header beside it, and a run of any other command or a
/// result that no longer describes the input leaves the column to the input.
@MainActor
final class StudioEarthMapRendererTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("earth-renderer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    func testTheMapViewDrawsAnEarthRunInTheInputColumn() throws {
        let input = root.appendingPathComponent("delta.safetensors")
        let output = try TensorFixtures.write(
            to: root.appendingPathComponent("delta-logits.safetensors"),
            tensors: [.float32("logits", shape: [1, 2, 4, 4])]
        )
        let item = row(templateID: .geoFlood, input: input, output: output)
        let header = try XCTUnwrap(StudioSafetensorsHeader.load(from: output))

        XCTAssertEqual(
            StudioResultRenderers.canvasRendering(for: .map, document: .tensor(.safetensors(header)), item: item, inputURL: input),
            .earthMap(StudioEarthMapSource(kind: .flood, output: output, input: input, revision: item.updatedAt))
        )
        XCTAssertNil(
            StudioResultRenderers.canvasRendering(for: .map, document: nil, item: nil, inputURL: input),
            "a result that doesn't describe the input on screen leaves the column to the checklist"
        )
        XCTAssertNil(
            StudioResultRenderers.canvasRendering(for: .tensor, document: .tensor(.safetensors(header)), item: item, inputURL: input),
            "the tensor view keeps the input"
        )
        let condition = row(templateID: .sfxConditionText, input: nil, output: output)
        XCTAssertNil(StudioResultRenderers.canvasRendering(for: .map, document: nil, item: condition, inputURL: nil))

        XCTAssertEqual(
            StudioResultRenderers.rendering(for: .map, document: .tensor(.safetensors(header)), item: item),
            .tensor(.safetensors(header)),
            "the panel beside the map keeps the header"
        )
    }

    func testRastersBecomeImagesOfTheirSize() throws {
        let raster = StudioEarthRaster(width: 3, height: 2, rgba: [UInt8](repeating: 255, count: 3 * 2 * 4))
        let image = try XCTUnwrap(StudioEarthMapView.image(raster))
        XCTAssertEqual([image.width, image.height], [510, 340], "enlarged by whole pixels")
        XCTAssertNil(StudioEarthMapView.image(StudioEarthRaster(width: 0, height: 0, rgba: [])))
        XCTAssertEqual(StudioEarthMapView.percent(0.1237, digits: 1), "12.4%")
        XCTAssertEqual(StudioEarthMapView.percent(0.65, digits: 0), "65%")
    }

    private func row(templateID: CommandTemplateID, input: URL?, output: URL) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(), mode: .readImage, prompt: "", inputURL: input, outputURL: output,
            createdAt: Date(timeIntervalSince1970: 1_800_000_000), updatedAt: Date(timeIntervalSince1970: 1_800_000_060),
            status: .completed, exitCode: 0, commandPreview: "mere.run geo", outputText: nil,
            templateID: templateID, commandDraft: nil, artifactURLs: [output]
        )
    }
}
