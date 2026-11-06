import AppKit
import StudioKit
import SwiftUI

// Earth's Map view: what a `geo` run wrote, drawn as a picture in the Analyze input column.
// Flood and Fire show the class mask over the bundle's natural color with an opacity slider
// and a legend of each class's share; TESSERA and OlmoEarth show their embeddings as false
// color. The reading and the math are `StudioEarthMap` in StudioKit; this file only draws.

/// An Earth result as a map, read off the main thread when its files change.
struct StudioEarthMapView: View {
    let source: StudioEarthMapSource
    let maxHeight: CGFloat

    @State private var phase: Phase = .loading
    @State private var tileIndex = 0
    @State private var opacity: Double?

    private enum Phase {
        case loading
        case failed(String)
        case loaded(Drawn)
    }

    /// The map with its pictures made once, not on every body evaluation.
    private struct Drawn {
        let map: StudioEarthMap
        let overlays: [CGImage?]
        let bases: [CGImage?]
    }

    var body: some View {
        Group {
            switch phase {
            case .loading:
                placeholder {
                    ProgressView().controlSize(.small)
                    Text("Reading the map…")
                }
            case .failed(let message):
                placeholder {
                    Image(systemName: "map")
                        .font(.system(size: 26, weight: .medium))
                        .foregroundStyle(MereRunTheme.textMuted)
                    Text(message)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
            case .loaded(let drawn):
                loaded(drawn)
            }
        }
        .task(id: source) { await load() }
    }

    private func placeholder<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: MereRunTheme.Spacing.sm) {
            content()
        }
        .font(.system(size: 13))
        .foregroundStyle(MereRunTheme.textSecondary)
        .padding(MereRunTheme.Spacing.xl)
        .frame(maxWidth: .infinity)
        .frame(height: min(maxHeight, 260))
        .background(MereRunTheme.surface)
        .mereMediaFrame()
        .accessibilityElement(children: .combine)
    }

    private func load() async {
        phase = .loading
        tileIndex = 0
        let source = source
        let result = await Task.detached(priority: .userInitiated) {
            Result { try StudioEarthMap.load(source) }
        }.value
        guard !Task.isCancelled else { return }
        switch result {
        case .success(let map):
            phase = .loaded(Drawn(
                map: map,
                overlays: map.tiles.map { Self.image($0.overlay) },
                bases: map.tiles.map { $0.base.flatMap(Self.image) }
            ))
        case .failure(let error as StudioEarthMapError):
            phase = .failed(error.errorDescription ?? "Couldn't draw the map.")
        case .failure(let error):
            phase = .failed("Couldn't draw the map. \(error.localizedDescription)")
        }
    }

    // MARK: - The map

    @ViewBuilder
    private func loaded(_ drawn: Drawn) -> some View {
        let map = drawn.map
        let index = min(tileIndex, max(0, map.tiles.count - 1))
        VStack(alignment: .leading, spacing: 10) {
            if map.tiles.indices.contains(index), let overlay = drawn.overlays[index] {
                picture(overlay: overlay, base: drawn.bases[index], map: map, index: index)
                    .frame(maxWidth: .infinity)
                    .frame(maxHeight: max(160, maxHeight - Self.chromeHeight(map)))
                    .background(MereRunTheme.surface)
                    .mereMediaFrame()
            }
            if map.tiles.count > 1 || drawn.bases[index] != nil {
                controls(map: map, index: index, hasBase: drawn.bases[index] != nil)
            }
            legend(map: map, index: index)
            captions(map: map, hasBase: drawn.bases[index] != nil)
        }
    }

    /// Room under the picture for the controls, legend, and notes.
    private static func chromeHeight(_ map: StudioEarthMap) -> CGFloat {
        96 + CGFloat(map.notes.count) * 18
    }

    private func picture(overlay: CGImage, base: CGImage?, map: StudioEarthMap, index: Int) -> some View {
        ZStack {
            if let base {
                Image(decorative: base, scale: 1)
                    .resizable()
                    .interpolation(.high)
            } else {
                // Class 0 is see-through; with nothing under it, the frame's own fill shows.
                MereRunTheme.surfaceRaised
            }
            Image(decorative: overlay, scale: 1)
                .resizable()
                .interpolation(.none)
                .opacity(base == nil ? 1 : overlayOpacity(map))
        }
        .aspectRatio(CGFloat(overlay.width) / CGFloat(max(1, overlay.height)), contentMode: .fit)
        .accessibilityElement()
        .accessibilityAddTraits(.isImage)
        .accessibilityLabel(accessibilityDescription(map: map, index: index))
    }

    private func overlayOpacity(_ map: StudioEarthMap) -> Double {
        opacity ?? Self.defaultOpacity(map)
    }

    /// A mask starts translucent so the ground shows through; false color starts opaque.
    private static func defaultOpacity(_ map: StudioEarthMap) -> Double {
        if case .classes = map.content { return 0.65 }
        return 1
    }

    private func controls(map: StudioEarthMap, index: Int, hasBase: Bool) -> some View {
        HStack(spacing: 10) {
            if map.tiles.count > 1 {
                Button {
                    tileIndex = index - 1
                } label: {
                    Image(systemName: "chevron.left")
                }
                .buttonStyle(.mereIcon)
                .disabled(index == 0)
                .accessibilityLabel("Previous tile")
                Text("Tile \(index + 1) of \(map.tiles.count)")
                    .font(.system(size: 12, weight: .medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textSecondary)
                Button {
                    tileIndex = index + 1
                } label: {
                    Image(systemName: "chevron.right")
                }
                .buttonStyle(.mereIcon)
                .disabled(index + 1 >= map.tiles.count)
                .accessibilityLabel("Next tile")
            }
            Spacer(minLength: 8)
            if hasBase {
                Text("Opacity")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .accessibilityHidden(true)
                Slider(value: Binding(get: { overlayOpacity(map) }, set: { opacity = $0 }), in: 0...1)
                    .controlSize(.small)
                    .frame(width: 140)
                    .accessibilityLabel(Self.isMask(map) ? "Mask opacity" : "False-color opacity")
                    .accessibilityValue(Self.percent(overlayOpacity(map), digits: 0))
                Text(Self.percent(overlayOpacity(map), digits: 0))
                    .font(.system(size: 12, weight: .medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(width: 38, alignment: .trailing)
                    .accessibilityHidden(true)
            }
        }
        .frame(minHeight: 24)
    }

    @ViewBuilder
    private func legend(map: StudioEarthMap, index: Int) -> some View {
        switch map.content {
        case .classes(let classes):
            let fractions = map.tiles.indices.contains(index) ? map.tiles[index].fractions : []
            HStack(spacing: 16) {
                ForEach(Array(classes.enumerated()), id: \.offset) { offset, item in
                    HStack(spacing: 6) {
                        swatch(item.color)
                        Text(item.name)
                            .foregroundStyle(MereRunTheme.textPrimary)
                        Text(Self.percent(offset < fractions.count ? fractions[offset] : 0, digits: 1))
                            .monospacedDigit()
                            .foregroundStyle(MereRunTheme.textSecondary)
                    }
                    .accessibilityElement(children: .combine)
                }
                Spacer(minLength: 0)
            }
            .font(.system(size: 12, weight: .medium))
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Legend")
        case .embedding(let tensor, let dimension):
            HStack(spacing: 12) {
                ForEach(Array(["Red", "Green", "Blue"].enumerated()), id: \.offset) { offset, name in
                    HStack(spacing: 6) {
                        swatch(Self.componentColors[offset])
                        Text("PC \(offset + 1)")
                            .foregroundStyle(MereRunTheme.textPrimary)
                    }
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("\(name): principal component \(offset + 1)")
                }
                Spacer(minLength: 8)
                Text("\(dimension) dimensions · \(tensor)")
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .font(.system(size: 12, weight: .medium))
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Legend")
        }
    }

    private static let componentColors = [
        StudioEarthColor(red: 0xE5, green: 0x39, blue: 0x35),
        StudioEarthColor(red: 0x43, green: 0xA0, blue: 0x47),
        StudioEarthColor(red: 0x1E, green: 0x88, blue: 0xE5),
    ]

    private func swatch(_ color: StudioEarthColor?) -> some View {
        RoundedRectangle(cornerRadius: 3)
            .fill(color.map(Self.color) ?? Color.clear)
            .overlay {
                RoundedRectangle(cornerRadius: 3)
                    .strokeBorder(MereRunTheme.border, lineWidth: color == nil ? 1 : 0)
            }
            .frame(width: 12, height: 12)
            .accessibilityHidden(true)
    }

    @ViewBuilder
    private func captions(map: StudioEarthMap, hasBase: Bool) -> some View {
        let lines = (hasBase ? [map.baseCaption].compactMap { $0 } : []) + map.notes
        if !lines.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(lines, id: \.self) { line in
                    Text(line)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .font(.system(size: 11.5))
            .foregroundStyle(MereRunTheme.textMuted)
        }
    }

    // MARK: - Words and pictures

    private func accessibilityDescription(map: StudioEarthMap, index: Int) -> String {
        var sentence = "\(map.kind.title) map"
        if map.tiles.count > 1 { sentence += ", tile \(index + 1) of \(map.tiles.count)" }
        switch map.content {
        case .classes(let classes):
            let fractions = map.tiles[index].fractions
            let shares = classes.enumerated().map { offset, item in
                "\(item.name) \(Self.percent(offset < fractions.count ? fractions[offset] : 0, digits: 1))"
            }
            sentence += ": " + shares.joined(separator: ", ")
        case .embedding(_, let dimension):
            sentence += ": false color from \(dimension)-dimensional embeddings"
        }
        return sentence + "."
    }

    private static func isMask(_ map: StudioEarthMap) -> Bool {
        if case .classes = map.content { return true }
        return false
    }

    /// "12.4%" or "65%".
    static func percent(_ fraction: Double, digits: Int) -> String {
        String(format: "%.\(digits)f%%", fraction * 100)
    }

    private static func color(_ color: StudioEarthColor) -> Color {
        Color(red: Double(color.red) / 255, green: Double(color.green) / 255, blue: Double(color.blue) / 255)
    }

    /// The raster as an image, its see-through pixels left see-through. A small raster (a
    /// 16 × 16 embedding grid) is first enlarged by whole pixels to about 512 on its long side,
    /// so every cell stays a crisp square however the frame scales it; SwiftUI's
    /// `.interpolation(.none)` alone isn't honored by every renderer.
    static func image(_ raster: StudioEarthRaster) -> CGImage? {
        guard raster.width > 0, raster.height > 0,
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let factor = max(1, 512 / max(raster.width, raster.height))
        let (width, height) = (raster.width * factor, raster.height * factor)
        var rgba = raster.rgba
        if factor > 1 {
            rgba = [UInt8](repeating: 0, count: width * height * 4)
            for y in 0..<height {
                for x in 0..<width {
                    let source = ((y / factor) * raster.width + x / factor) * 4
                    let target = (y * width + x) * 4
                    rgba[target..<(target + 4)] = raster.rgba[source..<(source + 4)]
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
