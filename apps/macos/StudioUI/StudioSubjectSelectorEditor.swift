import AppKit
import CoreGraphics
import StudioKit
import SwiftUI

// Video ▸ Subjects keeps each selector as three text fields in its plan (one box, positive
// points, negative points). This draws them on the picture they describe instead, and keeps the
// text fields as what the plan writer reads.

/// The picture a subject selector is drawn on.
enum StudioSubjectPicture: Equatable {
    /// The reference image, in its own pixels.
    case image(URL)
    /// One frame of the driving clip at `time`, as `video prepare-masks` sees it: center-cropped
    /// into the plan's `canvas` (see `StudioSubjectFrameGeometry`).
    case clipFrame(url: URL, time: TimeInterval, canvas: CGSize)
}

/// A selector's box and points, drawn on its picture and written back to the plan's text fields.
///
/// The prompts live here with stable identities so selection and dragging work; every change is
/// encoded into the three bound strings, and a string edited elsewhere (an older plan, the Command
/// view) reseeds the drawing when it no longer matches.
struct StudioSubjectSelectorEditor: View {
    let picture: StudioSubjectPicture
    @Binding var box: String
    @Binding var positivePoints: String
    @Binding var negativePoints: String
    /// What the picture is, for the caption under it ("Drawn on the reference image.").
    let caption: String

    @State private var prompts: [StudioRegionPrompt] = []
    @State private var image: NSImage?
    @State private var imageSize = CGSize.zero
    @State private var unavailable = false

    private var texts: [String] { [box, positivePoints, negativePoints] }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if unavailable {
                Text("This picture could not be opened, so there is nothing to draw on.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 8)
            } else {
                StudioRegionPromptEditor(
                    image: image,
                    imageSize: imageSize,
                    prompts: $prompts,
                    maximumBoxes: 1,
                    maxHeight: 220,
                    placeholder: "Loading picture…"
                )
            }
            Text(readout)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
        .onAppear { prompts = StudioSubjectSelectorText.prompts(box: box, positivePoints: positivePoints, negativePoints: negativePoints) }
        .onChange(of: prompts) { _, prompts in
            let encoded = Self.encode(prompts)
            if encoded != texts {
                box = encoded[0]
                positivePoints = encoded[1]
                negativePoints = encoded[2]
            }
        }
        .onChange(of: texts) { _, texts in
            if Self.encode(prompts) != texts {
                prompts = StudioSubjectSelectorText.prompts(box: texts[0], positivePoints: texts[1], negativePoints: texts[2])
            }
        }
        .task(id: picture) { await load() }
    }

    private var readout: String {
        let drawn = prompts.countDescription
        return drawn.isEmpty ? caption : "\(caption) \(drawn)."
    }

    private static func encode(_ prompts: [StudioRegionPrompt]) -> [String] {
        [
            StudioSubjectSelectorText.boxText(prompts),
            StudioSubjectSelectorText.positivePointsText(prompts),
            StudioSubjectSelectorText.negativePointsText(prompts),
        ]
    }

    private func load() async {
        let picture = picture
        let loaded = await Task.detached(priority: .userInitiated) {
            StudioSubjectPictureLoader.load(picture)
        }.value
        guard !Task.isCancelled else { return }
        if let loaded {
            image = loaded.image
            imageSize = loaded.pixelSize
            unavailable = false
        } else {
            image = nil
            unavailable = true
        }
    }
}

/// Decodes the picture a selector is drawn on, in the pixel space its coordinates use.
enum StudioSubjectPictureLoader {
    struct Loaded {
        let image: NSImage
        let pixelSize: CGSize
    }

    static func load(_ picture: StudioSubjectPicture) -> Loaded? {
        switch picture {
        case .image(let url):
            guard let size = StudioAnalyzeMediaInfo.pixelSize(of: url),
                  let loaded = StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: 1_200) else {
                return nil
            }
            return Loaded(image: loaded.image, pixelSize: size)
        case .clipFrame(let url, let time, let canvas):
            guard let frame = StudioVideoFrameLoader.frame(of: url, at: time, maxPixelSize: 1_600)?.image,
                  let composed = composed(frame, into: canvas) else { return nil }
            return Loaded(image: composed, pixelSize: canvas)
        }
    }

    /// The frame drawn into the plan canvas the way the CLI crops it, so the canvas's pixels are
    /// the coordinates the selector is written in.
    private static func composed(_ frame: NSImage, into canvas: CGSize) -> NSImage? {
        let width = Int(canvas.width.rounded())
        let height = Int(canvas.height.rounded())
        guard width > 0, height > 0,
              let source = frame.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let context = CGContext(
                  data: nil,
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bytesPerRow: 0,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else { return nil }
        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.interpolationQuality = .high
        // The cover rect is centered, so the same rect serves Core Graphics' bottom-left origin.
        context.draw(source, in: StudioSubjectFrameGeometry.coverRect(
            source: CGSize(width: source.width, height: source.height), canvas: canvas
        ))
        guard let image = context.makeImage() else { return nil }
        return NSImage(cgImage: image, size: NSSize(width: width, height: height))
    }
}
