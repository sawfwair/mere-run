import AppKit
@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import simd
import XCTest

/// The splat viewer: an IDOL run's card draws the splat PLY with Metal in place of the Quick
/// Look tile, chosen by the PLY's header, while a mesh keeps Quick Look and its counts. The
/// renderer draws upright, sorted, and blended offscreen, which is also how a bitmap capture of
/// the view gets its pixels.
@MainActor
final class StudioGaussianSplatRendererTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("splat-renderer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    func testAnIDOLCardDrawsTheSplatInPlaceOfTheQuickLookTile() throws {
        let folder = root.appendingPathComponent("idol", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let ply = folder.appendingPathComponent("object.ply")
        let camera = folder.appendingPathComponent("camera.json")
        let input = folder.appendingPathComponent("input.png")
        let manifest = folder.appendingPathComponent("manifest.json")
        try SplatFixtures.ply(SplatFixtures.sphere(count: 64)).write(to: ply)
        try Data(#"{"fov": 23.6, "pitch": 0.2, "radius": 4, "yaw": 2.2}"#.utf8).write(to: camera)
        try Data([0]).write(to: input)
        try Data(#"{"schemaVersion": 1, "gaussianCount": 64}"#.utf8).write(to: manifest)
        let files = [ply, input, camera, manifest]
        let item = row(templateID: .visionImageTo3DIdol, output: manifest, artifacts: files)

        let card = try XCTUnwrap(StudioResultRenderers.cardRendering(for: item, files: files))
        let result = StudioGaussianSplatResult(splatURL: ply, cameraURL: camera, preparedInputURL: input)
        XCTAssertEqual(card, StudioCardRendering(rendering: .gaussianSplat(result, summary: StudioMeshSummary(gaussianCount: 64)),
                                                 placement: .replacesOutputs))
        XCTAssertEqual(StudioResultRenderers.renderedFiles(of: card.rendering, item: item), [ply])
    }

    /// A mesh or point cloud PLY keeps the Quick Look tile and the counts under it.
    func testAMeshCardKeepsQuickLook() throws {
        let cloud = root.appendingPathComponent("chair.ply")
        try SplatFixtures.ply([SplatFixtures.Row(position: .zero)], properties: ["x", "y", "z", "red", "green", "blue"]).write(to: cloud)
        let manifest = root.appendingPathComponent("chair-manifest.json")
        try Data(#"{"vertexCount": 8, "triangleCount": 12}"#.utf8).write(to: manifest)
        let item = row(templateID: .imageReconstruct3D, output: cloud, artifacts: [cloud, manifest])
        XCTAssertEqual(StudioResultRenderers.cardRendering(for: item, files: [cloud, manifest]),
                       StudioCardRendering(rendering: .meshSummary(StudioMeshSummary(vertexCount: 8, triangleCount: 12)), placement: .belowOutputs))
    }

    /// The sort puts the farthest Gaussian first for any eye.
    func testTheSortDrawsBackToFront() throws {
        let rows = (0..<200).map { index in
            SplatFixtures.Row(position: SIMD3(Float(index % 7) * 0.1, Float(index % 5) * 0.1, Float(index) * 0.01 - 1))
        }
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(rows))
        let renderer = try XCTUnwrap(StudioGaussianSplatRenderer(splat: splat))
        for yaw in stride(from: Float(0), to: 2 * .pi, by: 0.9) {
            let orbit = StudioGaussianSplatOrbit(yaw: yaw, pitch: 0.4, distance: 4, target: .zero)
            let order = try XCTUnwrap(renderer.order(for: orbit))
            let indices = UnsafeBufferPointer(start: order.buffer.contents().assumingMemoryBound(to: UInt32.self), count: renderer.count)
            XCTAssertEqual(Set(indices).count, renderer.count, "every Gaussian exactly once")
            let depths = indices.map { simd_dot(splat.positions[Int($0)] - orbit.fileEye, orbit.fileForward) }
            XCTAssertTrue(zip(depths, depths.dropFirst()).allSatisfy { $0 >= $1 - 1e-4 }, "farthest first at yaw \(yaw)")
        }
    }

    /// Offscreen, the sphere fixture fills the middle of the frame and its bright cap (file −y)
    /// is on top: the viewer stands a 3DGS file upright.
    func testTheRendererDrawsTheSplatUpright() throws {
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(SplatFixtures.sphere(count: 4_096)))
        let renderer = try XCTUnwrap(StudioGaussianSplatRenderer(splat: splat))
        let orbit = StudioGaussianSplatOrbit.framing(splat, camera: nil)
        let image = try XCTUnwrap(renderer.image(orbit: orbit, width: 128, height: 128))
        let bitmap = NSBitmapImageRep(cgImage: image)
        func pixel(_ x: Int, _ y: Int) -> NSColor { bitmap.colorAt(x: x, y: y) ?? .clear }
        XCTAssertEqual(pixel(2, 2).alphaComponent, 0, accuracy: 0.01, "the corner is clear for the card's background")
        XCTAssertGreaterThan(pixel(64, 64).alphaComponent, 0.95, "the sphere covers the center")
        func lightness(row y: Int) -> CGFloat {
            (44..<84).reduce(0) { total, x in
                let color = pixel(x, y)
                return total + color.redComponent + color.greenComponent + color.blueComponent
            } / 120
        }
        XCTAssertGreaterThan(lightness(row: 38), lightness(row: 90) + 0.15, "the bright cap is on top")
    }

    /// A bitmap capture (`cacheDisplay`, what the snapshot harness and printing use) never sees a
    /// CAMetalLayer's contents; the view draws an offscreen frame for it instead.
    func testABitmapCaptureOfTheViewShowsTheSplat() throws {
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(SplatFixtures.sphere(count: 2_048)))
        let renderer = try XCTUnwrap(StudioGaussianSplatRenderer(splat: splat))
        let view = StudioGaussianSplatMetalView(scene: StudioGaussianSplatScene(renderer: renderer, splat: splat, camera: nil))
        view.frame = CGRect(x: 0, y: 0, width: 120, height: 90)
        view.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        XCTAssertGreaterThan(bitmap.colorAt(x: bitmap.pixelsWide / 2, y: bitmap.pixelsHigh / 2)?.alphaComponent ?? 0, 0.95)
    }

    func testTheSceneLabelCountsTheGaussiansAndAnyStride() throws {
        let rows = (0..<10).map { SplatFixtures.Row(position: SIMD3(Float($0), 0, 0)) }
        let whole = try StudioGaussianSplat.decode(SplatFixtures.ply(rows))
        let strided = try StudioGaussianSplat.decode(SplatFixtures.ply(rows), maximumCount: 5)
        let renderer = try XCTUnwrap(StudioGaussianSplatRenderer(splat: whole))
        XCTAssertEqual(StudioGaussianSplatScene(renderer: renderer, splat: whole, camera: nil).countText, "10 Gaussians")
        XCTAssertEqual(StudioGaussianSplatScene(renderer: renderer, splat: strided, camera: nil).countText, "5 of 10 Gaussians")
    }

    private func row(templateID: CommandTemplateID, output: URL, artifacts: [URL]) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(), mode: templateID.studioTask.mode ?? .createImage, prompt: "", inputURL: nil, outputURL: output,
            createdAt: Date(), updatedAt: Date(), status: .completed, exitCode: 0, commandPreview: "mere.run",
            outputText: nil, templateID: templateID, commandDraft: nil, artifactURLs: artifacts
        )
    }
}
