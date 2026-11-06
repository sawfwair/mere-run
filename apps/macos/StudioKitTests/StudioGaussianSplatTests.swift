@testable import StudioKit
import StudioTestSupport
import simd
import XCTest

/// The splat viewer reads INRIA 3DGS PLY files, IDOL's `object.ply` among them: the header names
/// the properties and their order, rows are binary little-endian, and the reader applies the
/// activations a renderer needs (SH DC to RGB, sigmoid opacity, exp scales, unit rotations).
final class StudioGaussianSplatTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("splat-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    private let rows = [
        SplatFixtures.Row(position: SIMD3(0.1, -0.2, 0.3), dc: SIMD3(1, -1, 0), opacityLogit: 0,
                          logScale: SIMD3(0, log(2), log(0.5)), rotation: SIMD4(2, 0, 0, 0)),
        SplatFixtures.Row(position: SIMD3(-0.5, 0.25, 0), dc: SIMD3(10, -10, 0.5), opacityLogit: 2,
                          logScale: SIMD3(repeating: -3), rotation: SIMD4(0, 0, 0, 3)),
    ]

    func testTheHeaderNamesIDOLsPropertiesInOrder() throws {
        let header = try StudioPLYHeader.parse(SplatFixtures.ply(rows))
        XCTAssertEqual(header.format, .binaryLittleEndian)
        XCTAssertEqual(header.vertex?.count, 2)
        XCTAssertEqual(header.vertex?.properties.map(\.name), SplatFixtures.idolProperties)
        XCTAssertEqual(header.vertex?.rowByteCount, 17 * 4)
        XCTAssertTrue(header.isGaussianSplat)
        XCTAssertEqual(header.missingSplatProperties, [])
    }

    func testTheReaderAppliesTheSplatActivations() throws {
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(rows))
        XCTAssertEqual(splat.count, 2)
        XCTAssertEqual(splat.fileCount, 2)
        XCTAssertEqual(splat.stride, 1)
        XCTAssertEqual(splat.positions[0], SIMD3(0.1, -0.2, 0.3))
        // color = clamp(0.5 + 0.28209479 · f_dc, 0, 1)
        XCTAssertEqual(splat.colors[0].x, 0.5 + 0.282_094_79, accuracy: 1e-6)
        XCTAssertEqual(splat.colors[0].y, 0.5 - 0.282_094_79, accuracy: 1e-6)
        XCTAssertEqual(splat.colors[0].z, 0.5, accuracy: 1e-6)
        XCTAssertEqual(simd_distance(splat.colors[1], SIMD3(1, 0, 0.5 + 0.5 * 0.282_094_79)), 0, accuracy: 1e-6, "clamped to 0...1")
        // opacity = sigmoid(logit)
        XCTAssertEqual(splat.opacities[0], 0.5, accuracy: 1e-6)
        XCTAssertEqual(splat.opacities[1], 1 / (1 + exp(-2)), accuracy: 1e-6)
        // scale = exp(log scale)
        XCTAssertEqual(splat.scales[0].x, 1, accuracy: 1e-6)
        XCTAssertEqual(splat.scales[0].y, 2, accuracy: 1e-6)
        XCTAssertEqual(splat.scales[0].z, 0.5, accuracy: 1e-6)
        // rot_0 is the real part; rotations come back unit length.
        XCTAssertEqual(splat.rotations[0].real, 1, accuracy: 1e-6)
        XCTAssertEqual(splat.rotations[1].imag.z, 1, accuracy: 1e-6)
        XCTAssertEqual(splat.rotations[1].real, 0, accuracy: 1e-6)
        XCTAssertEqual(splat.boundsMin, SIMD3(-0.5, -0.2, 0))
        XCTAssertEqual(splat.boundsMax, SIMD3(0.1, 0.25, 0.3))
    }

    /// Other 3DGS exporters drop the normals, add `f_rest_*`, or reorder; the header decides.
    func testThePropertyOrderComesFromTheHeader() throws {
        let shuffled = ["rot_3", "opacity", "f_rest_0", "z", "scale_2", "f_dc_2", "y", "rot_0", "scale_0",
                        "f_dc_0", "x", "rot_1", "f_rest_1", "scale_1", "f_dc_1", "rot_2"]
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(rows, properties: shuffled))
        let reference = try StudioGaussianSplat.decode(SplatFixtures.ply(rows))
        XCTAssertEqual(splat.positions, reference.positions)
        XCTAssertEqual(splat.colors, reference.colors)
        XCTAssertEqual(splat.opacities, reference.opacities)
        XCTAssertEqual(splat.scales, reference.scales)
        XCTAssertEqual(splat.rotations.map(\.vector), reference.rotations.map(\.vector))
    }

    func testAnElementBeforeTheVerticesIsSkipped() throws {
        var data = Data("ply\nformat binary_little_endian 1.0\nelement camera 1\nproperty double fx\nproperty uchar id\n".utf8)
        data += Data("element vertex 2\n".utf8)
        data += Data((SplatFixtures.idolProperties.map { "property float \($0)\n" }.joined() + "end_header\n").utf8)
        data += Data(repeating: 0xAB, count: 9)
        let plain = SplatFixtures.ply(rows)
        data += plain[(try StudioPLYHeader.parse(plain).dataOffset)...]
        let splat = try StudioGaussianSplat.decode(data)
        XCTAssertEqual(splat.positions, try StudioGaussianSplat.decode(plain).positions)
    }

    func testAFileOverTheCapIsReadAtAStride() throws {
        let many = (0..<10).map { SplatFixtures.Row(position: SIMD3(Float($0), 0, 0)) }
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(many), maximumCount: 4)
        XCTAssertEqual(splat.fileCount, 10)
        XCTAssertEqual(splat.stride, 3)
        XCTAssertEqual(splat.positions.map(\.x), [0, 3, 6, 9])
    }

    func testNonFiniteRowsAreDropped() throws {
        var broken = rows
        broken[0].position.x = .nan
        let splat = try StudioGaussianSplat.decode(SplatFixtures.ply(broken))
        XCTAssertEqual(splat.count, 1)
        XCTAssertEqual(splat.fileCount, 2)
        XCTAssertEqual(splat.positions[0], SIMD3(-0.5, 0.25, 0))
    }

    func testFormatsAndShapesTheViewerCannotDrawAreRejectedCleanly() {
        XCTAssertThrowsError(try StudioGaussianSplat.decode(SplatFixtures.ply(rows, format: "ascii"))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .unsupportedFormat("ASCII"))
            XCTAssertEqual((error as? LocalizedError)?.errorDescription,
                           "Studio reads binary little-endian splat PLY files, not ASCII.")
        }
        XCTAssertThrowsError(try StudioGaussianSplat.decode(SplatFixtures.ply(rows, format: "binary_big_endian"))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .unsupportedFormat("big-endian binary"))
        }
        XCTAssertThrowsError(try StudioGaussianSplat.decode(SplatFixtures.ply(rows, properties: ["x", "y", "z", "red", "green", "blue"]))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .missingProperties([
                "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
            ]))
        }
        XCTAssertThrowsError(try StudioGaussianSplat.decode(Data("solid cube\n".utf8))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .notPLY)
        }
        XCTAssertThrowsError(try StudioGaussianSplat.decode(Data("ply\nformat binary_little_endian 1.0\n".utf8))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .unterminatedHeader)
        }
        let full = SplatFixtures.ply(rows)
        XCTAssertThrowsError(try StudioGaussianSplat.decode(full.prefix(full.count - 5))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .truncated(expectedBytes: 136, actualBytes: 131))
        }
        XCTAssertThrowsError(try StudioGaussianSplat.decode(SplatFixtures.ply([]))) { error in
            XCTAssertEqual(error as? StudioGaussianSplatError, .empty)
        }
        let mesh = Data("ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n".utf8)
        XCTAssertFalse(try StudioPLYHeader.parse(mesh).isGaussianSplat)
    }

    /// A splat is found by its header, not its name: a point cloud PLY keeps Quick Look, and
    /// IDOL's camera and prepared input come from the same folder.
    func testASplatResultIsFoundByItsHeader() throws {
        let splat = root.appendingPathComponent("object.ply")
        let cloud = root.appendingPathComponent("cloud.ply")
        let camera = root.appendingPathComponent("camera.json")
        let input = root.appendingPathComponent("input.png")
        try SplatFixtures.ply(rows).write(to: splat)
        try SplatFixtures.ply(rows, properties: ["x", "y", "z", "nx", "ny", "nz"]).write(to: cloud)
        XCTAssertTrue(StudioPLYHeader.isGaussianSplat(at: splat))
        XCTAssertFalse(StudioPLYHeader.isGaussianSplat(at: cloud))
        XCTAssertFalse(StudioPLYHeader.isGaussianSplat(at: camera), "not a PLY")
        XCTAssertNil(StudioGaussianSplatResult.find(in: [cloud, camera]))
        XCTAssertEqual(StudioGaussianSplatResult.find(in: [cloud, splat, camera, input]),
                       StudioGaussianSplatResult(splatURL: splat, cameraURL: camera, preparedInputURL: input))
        let renamed = root.appendingPathComponent("scan.ply")
        try FileManager.default.moveItem(at: splat, to: renamed)
        XCTAssertEqual(StudioGaussianSplatResult.find(in: [renamed])?.splatURL, renamed)
    }

    func testTheLoaderReadsFromDiskOffTheCallersThread() async throws {
        let url = root.appendingPathComponent("object.ply")
        try SplatFixtures.ply(SplatFixtures.sphere(count: 512)).write(to: url)
        let splat = try await StudioGaussianSplat.load(from: url)
        XCTAssertEqual(splat.count, 512)
        let sphere = splat.framingSphere
        XCTAssertEqual(simd_length(sphere.center), 0, accuracy: 0.05)
        XCTAssertEqual(sphere.radius, 0.4, accuracy: 0.01)
    }

    /// IDOL's `camera.json` (`IdolCamera`), as `native-submarine` wrote it in qualification.
    func testTheInputCameraSetsTheStartingView() throws {
        let url = root.appendingPathComponent("camera.json")
        try Data(#"{"fov": 23.643785, "pitch": 0.21155815, "radius": 3.9961631, "yaw": 2.2415354}"#.utf8).write(to: url)
        let camera = try XCTUnwrap(StudioGaussianSplatCamera.load(from: url))
        XCTAssertEqual(camera.radius, 3.9961631)
        let orbit = StudioGaussianSplatOrbit.framing(sphere: (SIMD3(0, 0.1, 0), 0.5), camera: camera)
        XCTAssertEqual(orbit.yaw, .pi - 2.2415354, accuracy: 1e-6)
        XCTAssertEqual(orbit.pitch, 0.21155815, accuracy: 1e-6)
        XCTAssertEqual(orbit.target, SIMD3(0, -0.1, 0), "file y is down; the world stands it upright")
        // IDOL's direction toward the camera, (sin yaw cos pitch, cos yaw cos pitch, sin pitch) in
        // its z-up frame, is (x, z, −y) in the upright world.
        let idol = SIMD3(sin(camera.yaw) * cos(camera.pitch), cos(camera.yaw) * cos(camera.pitch), sin(camera.pitch))
        let toward = simd_normalize(orbit.eye - orbit.target)
        XCTAssertEqual(simd_distance(toward, SIMD3(idol.x, idol.z, -idol.y)), 0, accuracy: 1e-5)
        XCTAssertNil(StudioGaussianSplatCamera.load(from: root.appendingPathComponent("missing.json")))
    }

    func testFramingFitsTheSphereInTheNarrowerSide() {
        let square = StudioGaussianSplatOrbit.framing(sphere: (.zero, 1), camera: nil)
        let tall = StudioGaussianSplatOrbit.framing(sphere: (.zero, 1), camera: nil, aspect: 0.5)
        XCTAssertEqual(square.distance, 1.15 / sin(StudioGaussianSplatOrbit.defaultFieldOfView / 2), accuracy: 1e-4)
        XCTAssertGreaterThan(tall.distance, square.distance)
        XCTAssertEqual(square.yaw, StudioGaussianSplatOrbit.defaultAngles.yaw)
    }

    func testGesturesOrbitZoomAndPanWithinLimits() {
        var orbit = StudioGaussianSplatOrbit.framing(sphere: (.zero, 1), camera: nil)
        let framed = orbit
        orbit.orbit(by: CGSize(width: 100, height: 0), height: 100)
        XCTAssertEqual(orbit.yaw, framed.yaw - .pi, accuracy: 1e-5, "a full-height drag turns half a turn")
        orbit.orbit(by: CGSize(width: 0, height: 10_000), height: 100)
        XCTAssertEqual(orbit.pitch, StudioGaussianSplatOrbit.pitchLimit, "never over the pole")
        orbit.zoom(by: 0.001, framedDistance: framed.distance)
        XCTAssertEqual(orbit.distance, framed.distance * 0.05, accuracy: 1e-5)
        orbit.zoom(by: 1_000, framedDistance: framed.distance)
        XCTAssertEqual(orbit.distance, framed.distance * 8, accuracy: 1e-4)
        var panned = framed
        panned.pan(by: CGSize(width: 10, height: 0), height: 100)
        XCTAssertNotEqual(panned.target, framed.target)
        XCTAssertEqual(simd_dot(panned.target - framed.target, framed.forward), 0, accuracy: 1e-6, "panning stays in the view plane")
        let halfway = framed.interpolated(to: orbit, progress: 0.5)
        XCTAssertEqual(halfway.distance, (framed.distance + orbit.distance) / 2, accuracy: 1e-4)
        XCTAssertEqual(framed.interpolated(to: orbit, progress: 1).distance, orbit.distance, accuracy: 1e-4)
    }

    /// The view matrix looks from the eye at the target with +y up on screen.
    func testTheViewMatrixPutsTheTargetAheadAndUpUp() {
        let orbit = StudioGaussianSplatOrbit(yaw: 0.7, pitch: 0.3, distance: 3, target: SIMD3(1, 2, 3))
        let target = orbit.viewMatrix * SIMD4(orbit.target, 1)
        XCTAssertEqual(target.x, 0, accuracy: 1e-5)
        XCTAssertEqual(target.y, 0, accuracy: 1e-5)
        XCTAssertEqual(target.z, 3, accuracy: 1e-5)
        let above = orbit.viewMatrix * SIMD4(orbit.target + SIMD3(0, 1, 0), 1)
        XCTAssertGreaterThan(above.y, 0)
        let fileTarget = orbit.fileViewMatrix * SIMD4(orbit.target * StudioGaussianSplatOrbit.fileToWorld, 1)
        XCTAssertEqual(fileTarget.z, 3, accuracy: 1e-5)
        XCTAssertEqual(simd_distance(orbit.fileEye * StudioGaussianSplatOrbit.fileToWorld, orbit.eye), 0, accuracy: 1e-6)
    }
}
