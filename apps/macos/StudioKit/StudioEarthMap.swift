import Accelerate
import Foundation

// Earth results as pictures. The `geo` commands write safetensors (`GeoFloodCommand`,
// `GeoFireCommand`, `GeoTESSERACommand`, `GeoOlmoEarthCommand`, and docs/runtime/geo.md):
//
// - Flood and Fire: `logits`, F32 `[batch, 2, 256, 256]`. Class 1 is the event (the reference
//   validator scores `argmax(axis: 1) == 1`), class 0 everything else. The input bundle's
//   `S2L2A` is `[batch, 12, 4, 256, 256]`, normalized, in TerraMind's band order
//   (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12).
// - TESSERA: `embeddings`, F32 `[pixels, dimensions]`. The input is per-pixel time series
//   (`S2` is `[pixels, time, 10]`), so the file carries no grid.
// - OlmoEarth: `<MODALITY>_EMBEDDINGS`, F32 `[batch, height / patch, width / patch, dimensions]`,
//   time-pooled; `_TOKENS` beside them when asked. The input's `S2L2A` is raw
//   `[batch, height, width, time, 12]` in OlmoEarth's order (B02, B03, B04, B08, …).
//
// Everything here is a pure function of those buffers, so a map is testable without a view.

/// An opaque 8-bit color a class is drawn in.
package struct StudioEarthColor: Equatable, Sendable {
    package let red: UInt8
    package let green: UInt8
    package let blue: UInt8

    package init(red: UInt8, green: UInt8, blue: UInt8) {
        self.red = red
        self.green = green
        self.blue = blue
    }

    /// Flood water: a saturated blue that reads over both land and the dark-mode frame.
    package static let floodWater = StudioEarthColor(red: 0x1E, green: 0x88, blue: 0xE5)
    /// Fire: an orange-red that reads over vegetation and bare ground.
    package static let fire = StudioEarthColor(red: 0xF4, green: 0x51, blue: 0x1E)
    /// Any further class a model might add, in turn.
    package static let extra: [StudioEarthColor] = [
        StudioEarthColor(red: 0xFD, green: 0xD8, blue: 0x35),
        StudioEarthColor(red: 0x8E, green: 0x24, blue: 0xAA),
        StudioEarthColor(red: 0x00, green: 0x89, blue: 0x7B),
    ]
}

/// An RGBA picture, 8 bits per channel, top row first. A see-through pixel is all zeros, so
/// the bytes are the same premultiplied or not.
package struct StudioEarthRaster: Equatable, Sendable {
    package let width: Int
    package let height: Int
    package let rgba: [UInt8]

    package init(width: Int, height: Int, rgba: [UInt8]) {
        self.width = width
        self.height = height
        self.rgba = rgba
    }

    /// (red, green, blue, alpha) at a pixel.
    package func pixel(x: Int, y: Int) -> [UInt8] {
        let start = (y * width + x) * 4
        return Array(rgba[start..<(start + 4)])
    }
}

// MARK: - Segmentation

/// One tile's per-pixel class: the argmax over the class axis of `[classes, height, width]`
/// logits, and the share of the tile each class takes.
package struct StudioEarthClassMap: Equatable, Sendable {
    package let width: Int
    package let height: Int
    package let classCount: Int
    /// Row-major class indices.
    package let classes: [UInt8]
    /// Each class's share of the tile's pixels, summing to 1.
    package let fractions: [Double]

    /// The argmax of one tile's logits, laid out `[classes, height, width]`. Ties keep the
    /// lower class, as NumPy's `argmax` does.
    package static func argmax(logits: [Float], classCount: Int, height: Int, width: Int) -> StudioEarthClassMap {
        precondition((1...256).contains(classCount) && logits.count == classCount * height * width)
        let plane = height * width
        var classes = [UInt8](repeating: 0, count: plane)
        var counts = [Int](repeating: 0, count: classCount)
        for pixel in 0..<plane {
            var best = logits[pixel]
            var bestClass = 0
            for candidate in 1..<max(1, classCount) {
                let value = logits[candidate * plane + pixel]
                if value > best {
                    best = value
                    bestClass = candidate
                }
            }
            classes[pixel] = UInt8(bestClass)
            counts[bestClass] += 1
        }
        return StudioEarthClassMap(
            width: width, height: height, classCount: classCount, classes: classes,
            fractions: counts.map { plane == 0 ? 0 : Double($0) / Double(plane) }
        )
    }

    /// The mask: each pixel in its class's color, or see-through where the color is nil.
    package func raster(colors: [StudioEarthColor?]) -> StudioEarthRaster {
        var rgba = [UInt8](repeating: 0, count: classes.count * 4)
        for (pixel, index) in classes.enumerated() {
            guard Int(index) < colors.count, let color = colors[Int(index)] else { continue }
            rgba[pixel * 4] = color.red
            rgba[pixel * 4 + 1] = color.green
            rgba[pixel * 4 + 2] = color.blue
            rgba[pixel * 4 + 3] = 255
        }
        return StudioEarthRaster(width: width, height: height, rgba: rgba)
    }
}

// MARK: - Contrast

/// A linear stretch from the 2nd to the 98th percentile, the usual way to show reflectance or
/// normalized bands whose range the file doesn't declare. Clipping both tails keeps a few
/// clouds or a dead pixel from flattening the rest.
package enum StudioEarthStretch {
    /// The percentile bounds of the finite values, from at most `sampleLimit` evenly spaced
    /// samples; nil when none is finite.
    package static func bounds(
        _ values: [Float], low: Float = 0.02, high: Float = 0.98, sampleLimit: Int = 65_536
    ) -> ClosedRange<Float>? {
        let step = max(1, values.count / max(1, sampleLimit))
        var sample: [Float] = []
        sample.reserveCapacity(min(values.count, sampleLimit + 1))
        for index in stride(from: 0, to: values.count, by: step) where values[index].isFinite {
            sample.append(values[index])
        }
        guard !sample.isEmpty else { return nil }
        sample.sort()
        func percentile(_ fraction: Float) -> Float {
            sample[Int((Float(sample.count - 1) * fraction).rounded())]
        }
        return percentile(low)...percentile(high)
    }

    /// `value` placed in `bounds` as a byte; a flat range is mid-gray.
    package static func byte(_ value: Float, in bounds: ClosedRange<Float>) -> UInt8 {
        let span = bounds.upperBound - bounds.lowerBound
        guard span > 0 else { return 128 }
        let unit = min(1, max(0, (value - bounds.lowerBound) / span))
        return UInt8((unit * 255).rounded())
    }
}

/// Three bands as a picture, each stretched on its own. A normalized band's offset and scale
/// differ from its neighbors', so a shared stretch would tint the result; stretching each
/// band undoes any positive per-band affine normalization.
package enum StudioEarthNaturalColor {
    package static func raster(red: [Float], green: [Float], blue: [Float], width: Int, height: Int) -> StudioEarthRaster {
        precondition(red.count == width * height && green.count == red.count && blue.count == red.count)
        let bands = [red, green, blue]
        let bounds = bands.map { StudioEarthStretch.bounds($0) ?? 0...0 }
        var rgba = [UInt8](repeating: 0, count: red.count * 4)
        for pixel in 0..<red.count {
            guard red[pixel].isFinite, green[pixel].isFinite, blue[pixel].isFinite else { continue }
            for channel in 0..<3 {
                rgba[pixel * 4 + channel] = StudioEarthStretch.byte(bands[channel][pixel], in: bounds[channel])
            }
            rgba[pixel * 4 + 3] = 255
        }
        return StudioEarthRaster(width: width, height: height, rgba: rgba)
    }
}

// MARK: - Embeddings

/// Embeddings as false color: each pixel's vector projected on the first three principal
/// components, which become red, green, and blue.
///
/// PCA rather than the first three channels: OlmoEarth's channels carry no ordering, and
/// TESSERA's Matryoshka prefix ends in a layer norm, so three raw channels show little of the
/// variance. The fit stays cheap at any tile size because the covariance comes from at most
/// `sampleLimit` evenly spaced pixels (a 1,024-wide Teacher embedding is a 1,024² matrix), and
/// the top three eigenvectors come from orthogonal iteration on it rather than a full
/// decomposition. Fitting once over every tile keeps a color meaning the same thing from tile
/// to tile.
package struct StudioEarthProjection: Equatable, Sendable {
    package let dimension: Int
    package let mean: [Float]
    /// `dimension × componentCount`, row-major: column `j` is component `j`, unit length, its
    /// largest loading positive so the colors don't flip between runs.
    package let basis: [Float]
    package let componentCount: Int
    /// The 2nd to 98th percentile of each component over the fitted sample.
    package let bounds: [ClosedRange<Float>]

    package static func fit(
        pixels: [Float], dimension: Int, sampleLimit: Int = 4_096, iterations: Int = 96
    ) -> StudioEarthProjection {
        precondition(dimension > 0 && pixels.count.isMultiple(of: dimension))
        let count = pixels.count / dimension
        let components = min(3, dimension)
        let step = max(1, count / max(1, sampleLimit))
        let rows = Array(stride(from: 0, to: count, by: step).prefix(sampleLimit))
        guard !rows.isEmpty else {
            return StudioEarthProjection(
                dimension: dimension, mean: [Float](repeating: 0, count: dimension),
                basis: [Float](repeating: 0, count: dimension * components), componentCount: components,
                bounds: [ClosedRange<Float>](repeating: 0...0, count: components)
            )
        }

        // The centered sample, `rows × dimension`.
        var mean = [Float](repeating: 0, count: dimension)
        for row in rows {
            for column in 0..<dimension { mean[column] += pixels[row * dimension + column] }
        }
        for column in 0..<dimension { mean[column] /= Float(rows.count) }
        var centered = [Float](repeating: 0, count: rows.count * dimension)
        for (index, row) in rows.enumerated() {
            for column in 0..<dimension {
                centered[index * dimension + column] = pixels[row * dimension + column] - mean[column]
            }
        }

        // Covariance up to scale: centeredᵀ · centered.
        var transposed = [Float](repeating: 0, count: centered.count)
        vDSP_mtrans(centered, 1, &transposed, 1, vDSP_Length(dimension), vDSP_Length(rows.count))
        var covariance = [Float](repeating: 0, count: dimension * dimension)
        vDSP_mmul(transposed, 1, centered, 1, &covariance, 1,
                  vDSP_Length(dimension), vDSP_Length(dimension), vDSP_Length(rows.count))

        // Orthogonal iteration from a fixed start, so the same file always gets the same colors.
        var generator = UInt64(0x9E37_79B9_7F4A_7C15)
        var basis = (0..<(dimension * components)).map { _ -> Float in
            generator = generator &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int64(bitPattern: generator >> 11) % 2_000_001) / 1_000_000 - 1
        }
        orthonormalize(&basis, dimension: dimension, components: components)
        var product = [Float](repeating: 0, count: basis.count)
        for _ in 0..<iterations {
            vDSP_mmul(covariance, 1, basis, 1, &product, 1,
                      vDSP_Length(dimension), vDSP_Length(components), vDSP_Length(dimension))
            basis = product
            orthonormalize(&basis, dimension: dimension, components: components)
        }
        for component in 0..<components {
            var largest = 0
            for row in 0..<dimension where abs(basis[row * components + component]) > abs(basis[largest * components + component]) {
                largest = row
            }
            if basis[largest * components + component] < 0 {
                for row in 0..<dimension { basis[row * components + component] *= -1 }
            }
        }

        var projected = [Float](repeating: 0, count: rows.count * components)
        vDSP_mmul(centered, 1, basis, 1, &projected, 1,
                  vDSP_Length(rows.count), vDSP_Length(components), vDSP_Length(dimension))
        var bounds = (0..<components).map { component in
            StudioEarthStretch.bounds(stride(from: component, to: projected.count, by: components).map { projected[$0] }) ?? 0...0
        }
        // A component with next to no spread is rounding noise (the data has fewer directions
        // than three); stretching it would paint that noise at full contrast, so it stays flat.
        let leading = bounds.first.map { $0.upperBound - $0.lowerBound } ?? 0
        for component in bounds.indices where bounds[component].upperBound - bounds[component].lowerBound <= leading * 1e-3 {
            bounds[component] = 0...0
        }
        return StudioEarthProjection(
            dimension: dimension, mean: mean, basis: basis, componentCount: components, bounds: bounds
        )
    }

    /// Each pixel's coordinates on the components, `pixels × componentCount`.
    package func project(_ pixels: [Float]) -> [Float] {
        let count = pixels.count / dimension
        var projected = [Float](repeating: 0, count: count * componentCount)
        guard count > 0 else { return projected }
        vDSP_mmul(pixels, 1, basis, 1, &projected, 1,
                  vDSP_Length(count), vDSP_Length(componentCount), vDSP_Length(dimension))
        // (x − mean) · basis = x · basis − mean · basis, without copying the pixels.
        for component in 0..<componentCount {
            var offset: Float = 0
            for row in 0..<dimension { offset += mean[row] * basis[row * componentCount + component] }
            for pixel in 0..<count { projected[pixel * componentCount + component] -= offset }
        }
        return projected
    }

    /// The pixels as a `width × height` picture in row-major order. Positions past the last
    /// pixel stay see-through.
    package func raster(pixels: [Float], width: Int, height: Int) -> StudioEarthRaster {
        let projected = project(pixels)
        let count = min(pixels.count / dimension, width * height)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for pixel in 0..<count {
            for component in 0..<componentCount {
                rgba[pixel * 4 + component] = StudioEarthStretch.byte(
                    projected[pixel * componentCount + component], in: bounds[component]
                )
            }
            rgba[pixel * 4 + 3] = 255
        }
        return StudioEarthRaster(width: width, height: height, rgba: rgba)
    }

    /// Modified Gram–Schmidt over the columns of a row-major `dimension × components` matrix.
    /// A column with nothing left after the earlier ones is zeroed rather than divided by zero.
    private static func orthonormalize(_ matrix: inout [Float], dimension: Int, components: Int) {
        for column in 0..<components {
            for earlier in 0..<column {
                var dot: Float = 0
                for row in 0..<dimension { dot += matrix[row * components + column] * matrix[row * components + earlier] }
                for row in 0..<dimension { matrix[row * components + column] -= dot * matrix[row * components + earlier] }
            }
            var norm: Float = 0
            for row in 0..<dimension { norm += matrix[row * components + column] * matrix[row * components + column] }
            norm = norm.squareRoot()
            for row in 0..<dimension {
                matrix[row * components + column] = norm > 1e-20 ? matrix[row * components + column] / norm : 0
            }
        }
    }
}

// MARK: - The map

/// Which Earth command wrote the result.
package enum StudioEarthMapKind: String, Equatable, Hashable, Sendable {
    case flood
    case fire
    case tessera
    case olmoEarth

    package init?(templateID: CommandTemplateID?) {
        switch templateID {
        case .geoFlood: self = .flood
        case .geoFire: self = .fire
        case .geoTessera: self = .tessera
        case .geoOlmoEarth: self = .olmoEarth
        default: return nil
        }
    }

    package var title: String {
        switch self {
        case .flood: return "Flood"
        case .fire: return "Fire"
        case .tessera: return "TESSERA"
        case .olmoEarth: return "OlmoEarth"
        }
    }

    /// The classes a segmentation head's logits mean, by index; empty for the embedders.
    /// Class 0 is drawn see-through so the base layer shows under it.
    package func classes(count: Int) -> [StudioEarthMapClass] {
        let event: StudioEarthMapClass
        switch self {
        case .flood: event = StudioEarthMapClass(name: "Flood", color: .floodWater)
        case .fire: event = StudioEarthMapClass(name: "Fire", color: .fire)
        case .tessera, .olmoEarth: return []
        }
        let known = [StudioEarthMapClass(name: "Background", color: nil), event]
        return (0..<count).map { index in
            index < known.count
                ? known[index]
                : StudioEarthMapClass(name: "Class \(index)", color: StudioEarthColor.extra[(index - known.count) % StudioEarthColor.extra.count])
        }
    }
}

/// One class of a segmentation map: what it means and how it is drawn.
package struct StudioEarthMapClass: Equatable, Sendable {
    package let name: String
    /// Nil draws the class see-through: the background.
    package let color: StudioEarthColor?

    package init(name: String, color: StudioEarthColor?) {
        self.name = name
        self.color = color
    }
}

/// The files a map is drawn from: the result, the bundle the run read, and the row's revision,
/// so a rerun into the same path is read again.
package struct StudioEarthMapSource: Equatable, Hashable, Sendable {
    package let kind: StudioEarthMapKind
    package let output: URL
    package let input: URL?
    package let revision: Date

    package init(kind: StudioEarthMapKind, output: URL, input: URL?, revision: Date = .distantPast) {
        self.kind = kind
        self.output = output
        self.input = input
        self.revision = revision
    }

    /// The map for an Earth run, or nil for any other run or one without a safetensors result.
    package init?(item: StudioLibraryItem, inputURL: URL?) {
        guard let kind = StudioEarthMapKind(templateID: item.templateID),
              let output = StudioAnalyzeDocumentSource.url(for: item),
              output.pathExtension.lowercased() == "safetensors" else { return nil }
        self.init(kind: kind, output: output, input: item.inputURL ?? inputURL, revision: item.updatedAt)
    }
}

/// Why a map couldn't be drawn.
package enum StudioEarthMapError: Error, Equatable, LocalizedError {
    case unreadable(StudioSafetensorsError)
    case unexpectedShape(name: String, shape: [Int], expected: String)

    package var errorDescription: String? {
        switch self {
        case .unreadable(let error):
            return "Couldn't draw the map. \(error.reason)"
        case .unexpectedShape(let name, let shape, let expected):
            return "Couldn't draw the map. \(name) is \(StudioEarthMap.shapeText(shape)), not \(expected)."
        }
    }
}

/// An Earth result ready to draw: one overlay per tile (a class mask or false-color
/// embeddings), the natural-color layer under it when the input bundle has one, and what the
/// picture means.
package struct StudioEarthMap: Equatable, Sendable {
    package struct Tile: Equatable, Sendable {
        package let overlay: StudioEarthRaster
        package let base: StudioEarthRaster?
        /// Each class's share of the tile; empty for embeddings.
        package let fractions: [Double]
    }

    package enum Content: Equatable, Sendable {
        /// A segmentation head's classes, by index.
        case classes([StudioEarthMapClass])
        /// False color from the named tensor's vectors of `dimension` values.
        case embedding(tensor: String, dimension: Int)
    }

    package let kind: StudioEarthMapKind
    package let content: Content
    package let tiles: [Tile]
    /// What the base layer is, when there is one.
    package let baseCaption: String?
    /// What else to know: why there's no base layer, or how the pixels are laid out.
    package let notes: [String]

    /// The names OlmoEarth gives its time-pooled grids, in the order the map prefers them.
    package static let olmoEarthEmbeddingTensors = ["S2L2A_EMBEDDINGS", "S1RTC_EMBEDDINGS", "LANDSAT_EMBEDDINGS"]

    /// Reads the result and, for the base layer, the input bundle. Throws only when the result
    /// itself can't be drawn; a base layer that can't be read becomes a note.
    package static func load(_ source: StudioEarthMapSource) throws -> StudioEarthMap {
        let file: StudioSafetensorsFile
        do {
            file = try StudioSafetensorsFile(url: source.output)
        } catch let error as StudioSafetensorsError {
            throw StudioEarthMapError.unreadable(error)
        }
        do {
            switch source.kind {
            case .flood, .fire:
                return try segmentation(source, file: file)
            case .tessera:
                return try tessera(file)
            case .olmoEarth:
                return try olmoEarth(source, file: file)
            }
        } catch let error as StudioSafetensorsError {
            throw StudioEarthMapError.unreadable(error)
        }
    }

    // MARK: Flood and Fire

    private static func segmentation(_ source: StudioEarthMapSource, file: StudioSafetensorsFile) throws -> StudioEarthMap {
        let logits = try file.entry("logits")
        guard logits.shape.count == 4, logits.shape.allSatisfy({ $0 > 0 }), logits.shape[1] <= 256 else {
            throw StudioEarthMapError.unexpectedShape(
                name: "logits", shape: logits.shape, expected: "[batch, classes, height, width]"
            )
        }
        let (batch, classCount, height, width) = (logits.shape[0], logits.shape[1], logits.shape[2], logits.shape[3])
        let classes = source.kind.classes(count: classCount)
        let plane = classCount * height * width
        let base = terraMindNaturalColor(input: source.input, batch: batch, height: height, width: width)
        let tiles = try (0..<batch).map { tile in
            let map = StudioEarthClassMap.argmax(
                logits: try file.values("logits", elements: (tile * plane)..<((tile + 1) * plane)),
                classCount: classCount, height: height, width: width
            )
            return Tile(overlay: map.raster(colors: classes.map(\.color)), base: base.layers?[tile], fractions: map.fractions)
        }
        return StudioEarthMap(
            kind: source.kind, content: .classes(classes), tiles: tiles,
            baseCaption: base.caption, notes: base.note.map { [$0] } ?? []
        )
    }

    /// TerraMind's S2L2A order puts B04, B03, and B02 at 3, 2, and 1.
    static let terraMindRGBBands = [3, 2, 1]

    /// The natural-color layer under a Flood or Fire mask: B04, B03, and B02 of the bundle's
    /// last observation, for each tile, when `S2L2A` lines up with the logits.
    private static func terraMindNaturalColor(
        input: URL?, batch: Int, height: Int, width: Int
    ) -> (layers: [StudioEarthRaster]?, caption: String?, note: String?) {
        let expected = "[\(batch), 12, observations, \(height), \(width)]"
        return naturalColor(input: input) { file in
            let entry = try file.entry("S2L2A")
            guard entry.shape.count == 5, entry.shape[0] == batch, entry.shape[1] == 12, entry.shape[2] > 0,
                  entry.shape[3] == height, entry.shape[4] == width else {
                return .failure("S2L2A is \(shapeText(entry.shape)), not \(expected).")
            }
            let observations = entry.shape[2]
            let plane = height * width
            let layers = try (0..<batch).map { tile in
                let bands = try terraMindRGBBands.map { band in
                    let start = ((tile * 12 + band) * observations + observations - 1) * plane
                    return try file.values("S2L2A", elements: start..<(start + plane))
                }
                return StudioEarthNaturalColor.raster(red: bands[0], green: bands[1], blue: bands[2], width: width, height: height)
            }
            return .success((layers, "Natural color from B04, B03, and B02, observation \(observations) of \(observations)."))
        }
    }

    // MARK: TESSERA

    private static func tessera(_ file: StudioSafetensorsFile) throws -> StudioEarthMap {
        let entry = try file.entry("embeddings")
        guard entry.shape.count == 2, entry.shape.allSatisfy({ $0 > 0 }) else {
            throw StudioEarthMapError.unexpectedShape(name: "embeddings", shape: entry.shape, expected: "[pixels, dimensions]")
        }
        let (count, dimension) = (entry.shape[0], entry.shape[1])
        let pixels = try file.values("embeddings").values
        let projection = StudioEarthProjection.fit(pixels: pixels, dimension: dimension)
        let grid = tesseraGrid(pixelCount: count)
        let overlay = projection.raster(pixels: pixels, width: grid.columns, height: grid.rows)
        let pixelWord = count == 1 ? "1 pixel" : "\(count.formatted()) pixels"
        return StudioEarthMap(
            kind: .tessera, content: .embedding(tensor: "embeddings", dimension: dimension),
            tiles: [Tile(overlay: overlay, base: nil, fractions: [])], baseCaption: nil,
            notes: ["TESSERA encodes each pixel on its own and stores no grid, so the \(pixelWord) run in input order, \(grid.columns) to a row."]
        )
    }

    /// TESSERA's pixels laid out row by row: a square when the count is one (a flattened
    /// square tile comes back as it was), otherwise the narrowest near-square that holds them.
    package static func tesseraGrid(pixelCount: Int) -> (columns: Int, rows: Int) {
        let root = Int(Double(pixelCount).squareRoot().rounded(.up))
        let columns = max(1, root)
        return (columns, max(1, (pixelCount + columns - 1) / columns))
    }

    // MARK: OlmoEarth

    /// OlmoEarth's S2L2A order puts B04, B03, and B02 at 2, 1, and 0.
    static let olmoEarthRGBBands = [2, 1, 0]

    private static func olmoEarth(_ source: StudioEarthMapSource, file: StudioSafetensorsFile) throws -> StudioEarthMap {
        let present = olmoEarthEmbeddingTensors.filter(file.contains)
        guard let name = present.first else {
            throw StudioSafetensorsError.missingTensor(StudioEarthInputRequirement.list(olmoEarthEmbeddingTensors, conjunction: "or"))
        }
        let entry = try file.entry(name)
        guard entry.shape.count == 4, entry.shape.allSatisfy({ $0 > 0 }) else {
            throw StudioEarthMapError.unexpectedShape(name: name, shape: entry.shape, expected: "[batch, rows, columns, dimensions]")
        }
        let (batch, rows, columns, dimension) = (entry.shape[0], entry.shape[1], entry.shape[2], entry.shape[3])
        let pixels = try file.values(name).values
        let projection = StudioEarthProjection.fit(pixels: pixels, dimension: dimension)
        let tileLength = rows * columns * dimension
        let base = olmoEarthNaturalColor(input: source.input, batch: batch, rows: rows, columns: columns)
        let tiles = (0..<batch).map { tile in
            Tile(
                overlay: projection.raster(
                    pixels: Array(pixels[(tile * tileLength)..<((tile + 1) * tileLength)]), width: columns, height: rows
                ),
                base: base.layers?[tile], fractions: []
            )
        }
        var notes: [String] = []
        if present.count > 1 {
            notes.append("Showing \(name). The file also holds \(StudioEarthInputRequirement.list(Array(present.dropFirst()), conjunction: "and")).")
        }
        if let note = base.note { notes.append(note) }
        return StudioEarthMap(
            kind: .olmoEarth, content: .embedding(tensor: name, dimension: dimension), tiles: tiles,
            baseCaption: base.caption, notes: notes
        )
    }

    /// The raw S2L2A bands under an OlmoEarth grid, when the bundle's tiles cover the grid at a
    /// whole patch size.
    private static func olmoEarthNaturalColor(
        input: URL?, batch: Int, rows: Int, columns: Int
    ) -> (layers: [StudioEarthRaster]?, caption: String?, note: String?) {
        naturalColor(input: input) { file in
            let entry = try file.entry("S2L2A")
            guard entry.shape.count == 5, entry.shape[0] == batch, entry.shape[3] > 0, entry.shape[4] == 12,
                  entry.shape[1].isMultiple(of: rows), entry.shape[2].isMultiple(of: columns),
                  entry.shape[1] / rows == entry.shape[2] / columns, entry.shape[1] >= rows else {
                return .failure("S2L2A is \(shapeText(entry.shape)), which doesn't cover the \(rows) × \(columns) grid.")
            }
            let (height, width, observations) = (entry.shape[1], entry.shape[2], entry.shape[3])
            let tileLength = height * width * observations * 12
            let layers = try (0..<batch).map { tile in
                let block = try file.values("S2L2A", elements: (tile * tileLength)..<((tile + 1) * tileLength))
                let bands = olmoEarthRGBBands.map { band in
                    (0..<(height * width)).map { pixel in block[(pixel * observations + observations - 1) * 12 + band] }
                }
                return StudioEarthNaturalColor.raster(red: bands[0], green: bands[1], blue: bands[2], width: width, height: height)
            }
            return .success((layers, "Natural color from S2L2A B04, B03, and B02, observation \(observations) of \(observations)."))
        }
    }

    // MARK: Shared

    private enum BaseLayers {
        case success(([StudioEarthRaster], String))
        case failure(String)
    }

    /// Reads the input bundle for a base layer; any reason it can't becomes the note the map
    /// shows beside the overlay instead.
    private static func naturalColor(
        input: URL?, read: (StudioSafetensorsFile) throws -> BaseLayers
    ) -> (layers: [StudioEarthRaster]?, caption: String?, note: String?) {
        let prefix = "Couldn't draw the natural-color layer."
        guard let input, FileManager.default.fileExists(atPath: input.path) else {
            return (nil, nil, "\(prefix) The input bundle isn't on disk.")
        }
        do {
            switch try read(try StudioSafetensorsFile(url: input)) {
            case .success(let (layers, caption)):
                return (layers, caption, nil)
            case .failure(let reason):
                return (nil, nil, "\(prefix) \(reason)")
            }
        } catch let error as StudioSafetensorsError {
            return (nil, nil, "\(prefix) \(error.reason)")
        } catch {
            return (nil, nil, "\(prefix) \(error.localizedDescription)")
        }
    }

    /// "[1, 2, 256, 256]"
    package static func shapeText(_ shape: [Int]) -> String {
        "[\(shape.map(String.init).joined(separator: ", "))]"
    }
}
