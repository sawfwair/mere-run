@testable import StudioKit
import StudioTestSupport
import XCTest

/// Earth results as maps: the safetensors reader decodes every dtype a bundle's writer is
/// likely to use, Flood and Fire logits become a class mask with each class's share over the
/// bundle's natural color, and TESSERA and OlmoEarth embeddings become false color. Every file
/// is a small synthetic one written here in the layout the `geo` commands use.
final class StudioEarthMapTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("earth-map-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    // MARK: Reader

    func testTheReaderDecodesValuesSlicesAndMetadata() throws {
        let url = try TensorFixtures.write(
            to: root.appendingPathComponent("logits.safetensors"),
            tensors: [
                .float32("first", shape: [2, 3], values: [1, 2, 3, 4, 5, 6]),
                .float32("second", shape: [2], values: [-1.5, 0.25]),
            ],
            metadata: ["format": "mere.run/terramind-flood-logits-v1"]
        )
        let file = try StudioSafetensorsFile(url: url)
        XCTAssertEqual(file.entries.map(\.name), ["first", "second"], "in written order")
        XCTAssertEqual(file.metadata["format"], "mere.run/terramind-flood-logits-v1")
        XCTAssertEqual(try file.values("first"), StudioTensorValues(shape: [2, 3], values: [1, 2, 3, 4, 5, 6]))
        XCTAssertEqual(try file.values("first", elements: 3..<6), [4, 5, 6])
        XCTAssertEqual(try file.values("second").values, [-1.5, 0.25])
        XCTAssertThrowsError(try file.values("third")) { error in
            XCTAssertEqual(error as? StudioSafetensorsError, .missingTensor("third"))
            XCTAssertEqual((error as? StudioSafetensorsError)?.reason, "The file has no third tensor.")
        }
        XCTAssertThrowsError(try file.values("first", elements: 4..<7)) { error in
            XCTAssertEqual(error as? StudioSafetensorsError, .outOfRange("first"))
        }
    }

    func testTheReaderDecodesHalfBFloatDoubleAndIntegerDTypes() throws {
        let doubles = [2.5, -0.125].map(\.bitPattern)
        let file = try StudioSafetensorsFile(data: TensorFixtures.safetensors(tensors: [
            // 1, -2, 0.5, the smallest subnormal, +infinity.
            .raw("half", dtype: "F16", shape: [5], littleEndian: [UInt16(0x3C00), 0xC000, 0x3800, 0x0001, 0x7C00]),
            // 1, -3.
            .raw("brain", dtype: "BF16", shape: [2], littleEndian: [UInt16(0x3F80), 0xC040]),
            .raw("double", dtype: "F64", shape: [2], littleEndian: doubles),
            .raw("reflectance", dtype: "U16", shape: [2], littleEndian: [UInt16(1_200), 65_535]),
            .raw("signed", dtype: "I16", shape: [2], littleEndian: [Int16(-300), 300]),
            .raw("bytes", dtype: "U8", shape: [2], littleEndian: [UInt8(7), 255]),
            .raw("long", dtype: "I64", shape: [1], littleEndian: [Int64(-42)]),
        ]))
        XCTAssertEqual(try file.values("half").values, [1, -2, 0.5, 0x1p-24, .infinity])
        XCTAssertEqual(try file.values("brain").values, [1, -3])
        XCTAssertEqual(try file.values("double").values, [2.5, -0.125])
        XCTAssertEqual(try file.values("reflectance").values, [1_200, 65_535])
        XCTAssertEqual(try file.values("signed").values, [-300, 300])
        XCTAssertEqual(try file.values("bytes").values, [7, 255])
        XCTAssertEqual(try file.values("long").values, [-42])
        XCTAssertEqual(StudioSafetensorsFile.halfToFloat(0x8000).sign, .minus, "negative zero keeps its sign")
    }

    func testTheReaderRefusesWhatItCannotDecode() throws {
        XCTAssertThrowsError(try StudioSafetensorsFile(data: Data("not a tensor file".utf8), fileName: "notes.txt")) { error in
            XCTAssertEqual(error as? StudioSafetensorsError, .notSafetensors("notes.txt"))
        }
        let complex = try StudioSafetensorsFile(data: TensorFixtures.safetensors(tensors: [
            .raw("phase", dtype: "C64", shape: [1], littleEndian: [UInt64(0)]),
        ]))
        XCTAssertThrowsError(try complex.values("phase")) { error in
            XCTAssertEqual((error as? StudioSafetensorsError)?.reason, "phase is C64, which Studio can't decode.")
        }
        // Six bytes declared as two float32 values: the byte range disagrees with the shape.
        let short = try StudioSafetensorsFile(data: TensorFixtures.safetensors(tensors: [
            .raw("logits", dtype: "F32", shape: [2], littleEndian: [UInt16(0), 0, 0]),
        ]))
        XCTAssertThrowsError(try short.values("logits")) { error in
            XCTAssertEqual(error as? StudioSafetensorsError, .malformedTensor("logits"))
        }
        XCTAssertThrowsError(try StudioSafetensorsFile(url: root.appendingPathComponent("gone.safetensors"))) { error in
            XCTAssertEqual(error as? StudioSafetensorsError, .unreadable("gone.safetensors"))
        }
    }

    // MARK: Transforms

    /// Argmax over the class axis of `[classes, height, width]`, ties to the lower class.
    func testArgmaxPicksEachPixelsClassAndCountsShares() {
        let background: [Float] = [2, 0, 1, 5]
        let flood: [Float] = [1, 3, 1, 9]
        let map = StudioEarthClassMap.argmax(logits: background + flood, classCount: 2, height: 2, width: 2)
        XCTAssertEqual(map.classes, [0, 1, 0, 1], "the tie at pixel 2 stays background")
        XCTAssertEqual(map.fractions, [0.5, 0.5])

        let raster = map.raster(colors: [nil, .floodWater])
        XCTAssertEqual(raster.pixel(x: 0, y: 0), [0, 0, 0, 0], "background is see-through")
        XCTAssertEqual(raster.pixel(x: 1, y: 0), [0x1E, 0x88, 0xE5, 255])
        XCTAssertEqual(raster.pixel(x: 1, y: 1), [0x1E, 0x88, 0xE5, 255])
    }

    /// Each band stretches on its own between its 2nd and 98th percentiles, so a normalized
    /// band and the raw one it came from draw the same picture.
    func testNaturalColorStretchesEachBandAndIgnoresNormalization() {
        let red: [Float] = [0, 1, 2, 3]
        let green: [Float] = [10, 10, 10, 10]
        let blue: [Float] = [3, 2, 1, .nan]
        let raw = StudioEarthNaturalColor.raster(red: red, green: green, blue: blue, width: 2, height: 2)
        XCTAssertEqual(raw.pixel(x: 0, y: 0), [0, 128, 255, 255], "a flat band is mid-gray")
        XCTAssertEqual(raw.pixel(x: 0, y: 1), [170, 128, 0, 255])
        XCTAssertEqual(raw.pixel(x: 1, y: 1), [0, 0, 0, 0], "a missing value is see-through")
        let normalized = StudioEarthNaturalColor.raster(
            red: red.map { ($0 - 1.5) / 0.7 }, green: green.map { $0 / 3 }, blue: blue.map { $0 * 4 + 100 }, width: 2, height: 2
        )
        XCTAssertEqual(normalized, raw)
    }

    /// Embeddings whose variance lies along two known directions: the first component finds
    /// the larger one, the second the other, and the pixel furthest along each is the brightest
    /// in its channel.
    func testPrincipalComponentsFindTheDirectionsOfMostVariance() {
        let dimension = 8
        var pixels: [Float] = []
        for index in 0..<64 {
            let major = Float(index % 8) - 3.5
            let minor = Float(index / 8) - 3.5
            var vector = [Float](repeating: 0.5, count: dimension)
            vector[2] += 4 * major
            vector[5] -= 1 * minor
            pixels += vector
        }
        let projection = StudioEarthProjection.fit(pixels: pixels, dimension: dimension)
        XCTAssertEqual(projection.componentCount, 3)
        let first = (0..<dimension).map { projection.basis[$0 * 3] }
        let second = (0..<dimension).map { projection.basis[$0 * 3 + 1] }
        XCTAssertEqual(first[2], 1, accuracy: 1e-3, "the largest loading is made positive")
        XCTAssertEqual(abs(second[5]), 1, accuracy: 1e-3)
        XCTAssertEqual(second[5], 1, accuracy: 1e-3, "and so is the second component's")

        let raster = projection.raster(pixels: pixels, width: 8, height: 8)
        XCTAssertEqual(raster.pixel(x: 7, y: 0)[0], 255, "furthest along the first component is the reddest")
        XCTAssertEqual(raster.pixel(x: 0, y: 0)[0], 0)
        XCTAssertEqual(raster.pixel(x: 0, y: 0)[1], 255, "the second component's loading is negative in the data, so low minor is bright green")
        XCTAssertEqual(raster.pixel(x: 3, y: 3)[2], 128, "no variance left for the third: mid-gray")
        XCTAssertEqual(StudioEarthProjection.fit(pixels: pixels, dimension: dimension), projection, "the same file gets the same colors")
    }

    func testTESSERAPixelsWrapIntoTheSmallestNearSquare() {
        XCTAssertTrue(StudioEarthMap.tesseraGrid(pixelCount: 4_096) == (64, 64))
        XCTAssertTrue(StudioEarthMap.tesseraGrid(pixelCount: 10) == (4, 3))
        XCTAssertTrue(StudioEarthMap.tesseraGrid(pixelCount: 1) == (1, 1))
    }

    // MARK: Loading

    /// Flood: two tiles of `[batch, 2, height, width]` logits over a bundle whose S2L2A is
    /// `[batch, 12, 4, height, width]` in TerraMind order. The base comes from B04, B03, B02 of
    /// the last observation.
    func testFloodLogitsBecomeAMaskOverNaturalColor() throws {
        let (height, width) = (4, 4)
        let plane = height * width
        // Tile 0: the top half floods. Tile 1: nothing does.
        let tile0 = [Float](repeating: 1, count: plane) + (0..<plane).map { $0 < plane / 2 ? 2 : 0 }
        let tile1 = [Float](repeating: 1, count: plane) + [Float](repeating: -1, count: plane)
        let output = try TensorFixtures.write(
            to: root.appendingPathComponent("flood-logits.safetensors"),
            tensors: [.float32("logits", shape: [2, 2, height, width], values: tile0 + tile1)]
        )
        let input = try TensorFixtures.write(
            to: root.appendingPathComponent("flood-input.safetensors"),
            tensors: [
                .float32("S2L2A", shape: [2, 12, 4, height, width], values: Self.terraMindBands(batch: 2, height: height, width: width)),
                .float32("S1RTC", shape: [2, 2, 4, height, width]),
                .float32("DEM", shape: [2, 1, 4, height, width]),
            ]
        )

        let map = try StudioEarthMap.load(StudioEarthMapSource(kind: .flood, output: output, input: input))
        XCTAssertEqual(map.content, .classes([
            StudioEarthMapClass(name: "Background", color: nil), StudioEarthMapClass(name: "Flood", color: .floodWater),
        ]))
        XCTAssertEqual(map.tiles.count, 2)
        XCTAssertEqual(map.tiles[0].fractions, [0.5, 0.5])
        XCTAssertEqual(map.tiles[1].fractions, [1, 0])
        XCTAssertEqual(map.tiles[0].overlay.pixel(x: 0, y: 0), [0x1E, 0x88, 0xE5, 255])
        XCTAssertEqual(map.tiles[0].overlay.pixel(x: 0, y: 3)[3], 0)
        XCTAssertEqual(map.baseCaption, "Natural color from B04, B03, and B02, observation 4 of 4.")
        XCTAssertEqual(map.notes, [])

        let base = try XCTUnwrap(map.tiles[0].base)
        XCTAssertEqual(base.width, width)
        // Red rises left to right, green top to bottom, blue is flat — in the last observation
        // only; the first three run the other way.
        XCTAssertEqual(base.pixel(x: 0, y: 0), [0, 0, 128, 255])
        XCTAssertEqual(base.pixel(x: 3, y: 0), [255, 0, 128, 255])
        XCTAssertEqual(base.pixel(x: 0, y: 3), [0, 255, 128, 255])
    }

    func testFireNamesItsClassAndAMismatchedBundleLeavesTheMaskAlone() throws {
        let output = try TensorFixtures.write(
            to: root.appendingPathComponent("fire-logits.safetensors"),
            tensors: [.float32("logits", shape: [1, 2, 4, 4], values: [Float](repeating: 0, count: 16) + [Float](repeating: 1, count: 16))]
        )
        let input = try TensorFixtures.write(
            to: root.appendingPathComponent("fire-input.safetensors"),
            tensors: [.float32("S2L2A", shape: [1, 12, 4, 8, 8])]
        )
        let map = try StudioEarthMap.load(StudioEarthMapSource(kind: .fire, output: output, input: input))
        XCTAssertEqual(map.content, .classes([
            StudioEarthMapClass(name: "Background", color: nil), StudioEarthMapClass(name: "Fire", color: .fire),
        ]))
        XCTAssertEqual(map.tiles[0].fractions, [0, 1])
        XCTAssertNil(map.tiles[0].base)
        XCTAssertNil(map.baseCaption)
        XCTAssertEqual(map.notes, ["Couldn't draw the natural-color layer. S2L2A is [1, 12, 4, 8, 8], not [1, 12, observations, 4, 4]."])

        let gone = try StudioEarthMap.load(StudioEarthMapSource(kind: .fire, output: output, input: root.appendingPathComponent("moved.safetensors")))
        XCTAssertEqual(gone.notes, ["Couldn't draw the natural-color layer. The input bundle isn't on disk."])
        XCTAssertEqual(gone.tiles.count, 1, "the mask still draws")
    }

    func testAResultTheMapCannotReadSaysWhy() throws {
        let embeddings = try TensorFixtures.write(
            to: root.appendingPathComponent("wrong.safetensors"),
            tensors: [.float32("embeddings", shape: [4, 8])]
        )
        XCTAssertThrowsError(try StudioEarthMap.load(StudioEarthMapSource(kind: .flood, output: embeddings, input: nil))) { error in
            XCTAssertEqual((error as? StudioEarthMapError)?.errorDescription, "Couldn't draw the map. The file has no logits tensor.")
        }
        let flat = try TensorFixtures.write(
            to: root.appendingPathComponent("flat.safetensors"),
            tensors: [.float32("logits", shape: [2, 16])]
        )
        XCTAssertThrowsError(try StudioEarthMap.load(StudioEarthMapSource(kind: .flood, output: flat, input: nil))) { error in
            XCTAssertEqual(
                (error as? StudioEarthMapError)?.errorDescription,
                "Couldn't draw the map. logits is [2, 16], not [batch, classes, height, width]."
            )
        }
        XCTAssertThrowsError(try StudioEarthMap.load(StudioEarthMapSource(kind: .olmoEarth, output: embeddings, input: nil))) { error in
            XCTAssertEqual(
                (error as? StudioEarthMapError)?.errorDescription,
                "Couldn't draw the map. The file has no S2L2A_EMBEDDINGS, S1RTC_EMBEDDINGS, or LANDSAT_EMBEDDINGS tensor."
            )
        }
    }

    /// TESSERA's `[pixels, dimensions]` has no grid: the pixels wrap in input order, the
    /// leftover cells stay see-through, and the map says so.
    func testTESSERAEmbeddingsBecomeFalseColorInInputOrder() throws {
        let values = (0..<(10 * 16)).map { Float(($0 * 7) % 13) / 13 }
        let output = try TensorFixtures.write(
            to: root.appendingPathComponent("tessera.safetensors"),
            tensors: [.float32("embeddings", shape: [10, 16], values: values)],
            metadata: ["format": "mere.run/tessera-v2-embeddings-v1", "dimensions": "16"]
        )
        let map = try StudioEarthMap.load(StudioEarthMapSource(kind: .tessera, output: output, input: nil))
        XCTAssertEqual(map.content, .embedding(tensor: "embeddings", dimension: 16))
        XCTAssertEqual(map.tiles.count, 1)
        let overlay = map.tiles[0].overlay
        XCTAssertEqual([overlay.width, overlay.height], [4, 3])
        XCTAssertEqual(overlay.pixel(x: 1, y: 2)[3], 255, "pixel 9")
        XCTAssertEqual(overlay.pixel(x: 2, y: 2), [0, 0, 0, 0], "past the last pixel")
        XCTAssertNil(map.tiles[0].base)
        XCTAssertEqual(map.notes, ["TESSERA encodes each pixel on its own and stores no grid, so the 10 pixels run in input order, 4 to a row."])
    }

    /// OlmoEarth's `[batch, rows, columns, dimensions]` grid over raw S2L2A
    /// `[batch, height, width, time, 12]`: the first modality is drawn, the others named, and
    /// the base comes from B04, B03, B02 (2, 1, 0 in OlmoEarth order) of the last observation.
    func testOlmoEarthEmbeddingsBecomeFalseColorOverNaturalColor() throws {
        let (rows, columns, dimension) = (2, 2, 6)
        let embeddings = (0..<(2 * rows * columns * dimension)).map { Float(($0 * 5) % 11) }
        let output = try TensorFixtures.write(
            to: root.appendingPathComponent("olmoearth.safetensors"),
            tensors: [
                .float32("S2L2A_EMBEDDINGS", shape: [2, rows, columns, dimension], values: embeddings),
                .float32("S1RTC_EMBEDDINGS", shape: [2, rows, columns, dimension]),
            ]
        )
        let (height, width, observations) = (8, 8, 2)
        var raw: [Float] = []
        for _ in 0..<2 {
            for pixel in 0..<(height * width) {
                for observation in 0..<observations {
                    for band in 0..<12 {
                        let last = observation == observations - 1
                        switch band {
                        case 2: raw.append(last ? Float(pixel % width) * 100 : 0)
                        case 1: raw.append(last ? 1_000 : Float(pixel))
                        case 0: raw.append(last ? Float(pixel / width) * 100 : 0)
                        default: raw.append(9_999)
                        }
                    }
                }
            }
        }
        let input = try TensorFixtures.write(
            to: root.appendingPathComponent("observations.safetensors"),
            tensors: [
                .int32("TIMESTAMPS", shape: [2, observations, 3], values: [1, 0, 2024, 1, 6, 2024, 1, 0, 2024, 1, 6, 2024]),
                .float32("S2L2A", shape: [2, height, width, observations, 12], values: raw),
            ]
        )
        let map = try StudioEarthMap.load(StudioEarthMapSource(kind: .olmoEarth, output: output, input: input))
        XCTAssertEqual(map.content, .embedding(tensor: "S2L2A_EMBEDDINGS", dimension: dimension))
        XCTAssertEqual(map.tiles.count, 2)
        XCTAssertEqual([map.tiles[0].overlay.width, map.tiles[0].overlay.height], [columns, rows])
        XCTAssertEqual(map.notes, ["Showing S2L2A_EMBEDDINGS. The file also holds S1RTC_EMBEDDINGS."])
        XCTAssertEqual(map.baseCaption, "Natural color from S2L2A B04, B03, and B02, observation 2 of 2.")
        let base = try XCTUnwrap(map.tiles[1].base)
        XCTAssertEqual([base.width, base.height], [width, height])
        XCTAssertEqual(base.pixel(x: 0, y: 0), [0, 128, 0, 255])
        XCTAssertEqual(base.pixel(x: 7, y: 7), [255, 128, 255, 255])
    }

    // MARK: Wiring

    func testEarthTasksOpenOnTheMap() {
        for task in [StudioTask.earthFlood, .earthFire, .earthTessera, .earthOlmoEarth] {
            XCTAssertEqual(task.analyzeArchetype?.views, [.map, .tensor, .json], "\(task)")
        }
    }

    func testAMapSourceComesFromAnEarthRunsSafetensors() throws {
        let input = root.appendingPathComponent("tiles.safetensors")
        let output = root.appendingPathComponent("tiles-logits.safetensors")
        let updated = Date(timeIntervalSince1970: 1_800_000_000)
        func row(_ templateID: CommandTemplateID, artifacts: [URL]) -> StudioLibraryItem {
            StudioLibraryItem(
                id: UUID(), mode: .readImage, prompt: "", inputURL: input, outputURL: artifacts.first,
                createdAt: updated, updatedAt: updated, status: .completed, exitCode: 0, commandPreview: "mere.run geo",
                outputText: nil, templateID: templateID, commandDraft: nil, artifactURLs: artifacts
            )
        }
        XCTAssertEqual(
            StudioEarthMapSource(item: row(.geoFlood, artifacts: [output]), inputURL: nil),
            StudioEarthMapSource(kind: .flood, output: output, input: input, revision: updated)
        )
        XCTAssertEqual(StudioEarthMapSource(item: row(.geoOlmoEarth, artifacts: [output]), inputURL: nil)?.kind, .olmoEarth)
        XCTAssertNil(StudioEarthMapSource(item: row(.geoTessera, artifacts: []), inputURL: nil), "no result file yet")
        XCTAssertNil(StudioEarthMapSource(item: row(.sfxConditionText, artifacts: [output]), inputURL: nil), "not an Earth run")
    }

    // MARK: Fixtures

    /// S2L2A in TerraMind order, `[batch, 12, 4, height, width]`: in the last observation B04
    /// rises left to right, B03 top to bottom, and B02 is flat; earlier observations run the
    /// other way, and every other band is noise the map must not pick.
    private static func terraMindBands(batch: Int, height: Int, width: Int) -> [Float] {
        var values: [Float] = []
        for _ in 0..<batch {
            for band in 0..<12 {
                for observation in 0..<4 {
                    for y in 0..<height {
                        for x in 0..<width {
                            let last = observation == 3
                            switch band {
                            case 3: values.append(Float(last ? x : width - x) * 0.4 - 1)
                            case 2: values.append(Float(last ? y : height - y) * 0.3)
                            case 1: values.append(-0.2)
                            default: values.append(Float((x * 31 + y * 17 + band) % 7))
                            }
                        }
                    }
                }
            }
        }
        return values
    }
}
