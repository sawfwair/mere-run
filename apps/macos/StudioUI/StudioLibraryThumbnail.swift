import AVFoundation
import AppKit
import StudioKit
import SwiftUI

/// Thumbnails for Library rows and tiles: the picture itself for images, a poster frame for video,
/// a peak silhouette for audio, and the first line for a text result. Decoding is off the main
/// actor and cached by path, size, and modification date, so scrolling a long Library re-reads
/// nothing and a file replaced in place still refreshes.
@MainActor
final class StudioThumbnailCache {
    static let shared = StudioThumbnailCache()

    private let images = NSCache<NSString, NSImage>()
    private let waveforms = NSCache<NSString, NSArray>()

    init(countLimit: Int = 240) {
        images.countLimit = countLimit
        waveforms.countLimit = countLimit
    }

    /// The identity of one cached thumbnail. The modification date is part of it so a run that
    /// overwrites its output in place (a retake writing the same path) invalidates its own
    /// thumbnail; `nil` means the file is gone, which is its own distinct key.
    nonisolated static func key(url: URL, maxPixelSize: Int, modified: Date?) -> String {
        let stamp = modified.map { String(format: "%.0f", $0.timeIntervalSince1970) } ?? "missing"
        return "\(url.standardizedFileURL.path)|\(maxPixelSize)|\(stamp)"
    }

    nonisolated static func modificationDate(of url: URL, fileManager: FileManager = .default) -> Date? {
        (try? fileManager.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }

    func image(forKey key: String) -> NSImage? {
        images.object(forKey: key as NSString)
    }

    func store(_ image: NSImage, forKey key: String) {
        images.setObject(image, forKey: key as NSString)
    }

    func peaks(forKey key: String) -> [Float]? {
        waveforms.object(forKey: key as NSString) as? [Float]
    }

    func store(_ peaks: [Float], forKey key: String) {
        waveforms.setObject(peaks as NSArray, forKey: key as NSString)
    }
}

/// A still from a movie, for rows whose output is a clip.
enum StudioVideoPosterLoader {
    /// The frame the poster is taken from: a moment in, so a clip that fades up from black does
    /// not thumbnail as a black square.
    static let posterTime = CMTime(seconds: 0.6, preferredTimescale: 600)

    static func poster(for url: URL, maxPixelSize: CGFloat) -> StudioLoadedImage? {
        guard StudioOutputFileKind.classify(url) == .video else { return nil }
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: maxPixelSize, height: maxPixelSize)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 2, preferredTimescale: 600)
        guard let cgImage = try? generator.copyCGImage(at: posterTime, actualTime: nil) else {
            return nil
        }
        return StudioLoadedImage(image: NSImage(
            cgImage: cgImage,
            size: NSSize(width: cgImage.width, height: cgImage.height)
        ))
    }
}

/// The thumbnail for one Library row or grid tile.
struct StudioLibraryThumbnail: View {
    let item: StudioLibraryItem
    /// The tile's side in points; the decode asks for twice that in pixels.
    let side: CGFloat

    private var url: URL? { item.outputURL }

    private var kind: StudioOutputFileKind? {
        StudioLibraryPresenter.fileKind(of: item)
    }

    var body: some View {
        Group {
            switch kind {
            case .image:
                if let url {
                    StudioAsyncImagePreview(
                        url: url,
                        maxPixelSize: side * 2,
                        contentMode: .fill,
                        fallbackSystemImage: item.displaySystemImage
                    )
                } else {
                    glyph
                }
            case .video:
                if let url {
                    StudioPosterThumbnail(url: url, side: side, fallbackSystemImage: item.displaySystemImage)
                } else {
                    glyph
                }
            case .audio:
                if let url {
                    StudioWaveformThumbnail(url: url, side: side)
                } else {
                    glyph
                }
            default:
                if let line = firstLine {
                    Text(line)
                        .font(.system(size: max(7, side * 0.16), weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .lineLimit(side > 56 ? 3 : 2)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding(4)
                } else {
                    glyph
                }
            }
        }
        .accessibilityHidden(true)
    }

    private var glyph: some View {
        Image(systemName: item.displaySystemImage)
            .font(.system(size: max(11, side * 0.34), weight: .medium))
            .foregroundStyle(MereRunTheme.textMuted)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The opening of a text result — the transcript's first sentence, the caption, the JSON's
    /// first key — so a text row is legible without opening it.
    private var firstLine: String? {
        guard let text = item.outputText?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return nil
        }
        let line = text.split(separator: "\n", omittingEmptySubsequences: true).first.map(String.init) ?? text
        return String(line.prefix(120))
    }
}

/// A movie's poster frame, decoded once and cached.
private struct StudioPosterThumbnail: View {
    let url: URL
    let side: CGFloat
    let fallbackSystemImage: String

    @State private var image: NSImage?
    @State private var failed = false

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .clipped()
            } else if failed {
                Image(systemName: fallbackSystemImage)
                    .font(.system(size: max(11, side * 0.34), weight: .medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Color.clear
            }
        }
        .overlay(alignment: .bottomTrailing) {
            if image != nil {
                Image(systemName: "play.fill")
                    .font(.system(size: max(6, side * 0.16), weight: .bold))
                    .foregroundStyle(.white)
                    .padding(2)
                    .background(Circle().fill(Color.black.opacity(0.45)))
                    .padding(3)
            }
        }
        .task(id: url) {
            let maxPixelSize = side * 2
            let key = StudioThumbnailCache.key(
                url: url,
                maxPixelSize: Int(maxPixelSize.rounded()),
                modified: StudioThumbnailCache.modificationDate(of: url)
            )
            if let cached = StudioThumbnailCache.shared.image(forKey: key) {
                image = cached
                return
            }
            let loaded = await Task.detached(priority: .utility) {
                StudioVideoPosterLoader.poster(for: url, maxPixelSize: maxPixelSize)
            }.value
            guard !Task.isCancelled else { return }
            if let poster = loaded?.image {
                StudioThumbnailCache.shared.store(poster, forKey: key)
                image = poster
            } else {
                failed = true
            }
        }
    }
}

/// A miniature of the same peaks the player draws, so an audio row reads as audio.
private struct StudioWaveformThumbnail: View {
    let url: URL
    let side: CGFloat

    @State private var peaks: [Float] = []

    private var barCount: Int { max(10, Int(side / 3)) }

    var body: some View {
        Canvas { context, size in
            let bars = peaks.isEmpty ? [Float](repeating: 0.22, count: barCount) : peaks
            let slot = size.width / CGFloat(bars.count)
            let barWidth = max(1, slot * 0.55)
            let middle = size.height / 2
            for (index, peak) in bars.enumerated() {
                let height = max(1.5, CGFloat(peak) * size.height * 0.78)
                let rect = CGRect(
                    x: CGFloat(index) * slot + (slot - barWidth) / 2,
                    y: middle - height / 2,
                    width: barWidth,
                    height: height
                )
                context.fill(
                    Path(roundedRect: rect, cornerRadius: barWidth / 2),
                    with: .color(peaks.isEmpty ? MereRunTheme.textMuted.opacity(0.4) : MereRunTheme.accent.opacity(0.75))
                )
            }
        }
        .padding(.horizontal, 4)
        .task(id: url) {
            let count = barCount
            let key = StudioThumbnailCache.key(
                url: url,
                maxPixelSize: count,
                modified: StudioThumbnailCache.modificationDate(of: url)
            )
            if let cached = StudioThumbnailCache.shared.peaks(forKey: key) {
                peaks = cached
                return
            }
            let loaded = await Task.detached(priority: .utility) {
                StudioWaveformLoader.peaks(url: url, barCount: count)
            }.value
            guard !Task.isCancelled, let loaded else { return }
            StudioThumbnailCache.shared.store(loaded, forKey: key)
            peaks = loaded
        }
    }
}
