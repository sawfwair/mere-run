import AppKit
import StudioKit
import SwiftUI

// The Vision overlays: face boxes with their five landmarks, and pose landmarks, drawn over the
// picture they were found in. The canvas shows one in the Analyze input column for the Points
// view; the face picker in the task inspector shows the same overlay with clickable faces, so
// `--face-index` is chosen by pointing at the face.

/// Which document the overlay draws.
enum StudioVisionOverlay: Equatable {
    case faces(StudioFaceOverlayResult)
    case pose(StudioPoseOverlayResult)
}

/// The input picture with a face or pose document drawn over it in the picture's own pixels.
/// The picture is shown upright; the documents place their boxes and landmarks in the stored
/// pixels the CLI decoded, so every point maps through the photo's orientation.
struct StudioVisionOverlayView: View {
    let imageURL: URL
    let overlay: StudioVisionOverlay
    /// When set, each face box is a button that picks its index, and the chosen one is drawn in
    /// the accent.
    var selectedFaceIndex: Binding<Int>?

    @State private var image: NSImage?
    @State private var orientation = StudioImageOrientation.up
    @State private var didLoad = false

    var body: some View {
        Group {
            if let image {
                // The frame hugs the picture, so the overlay's bounds are the displayed image
                // and one uniform scale maps stored pixels onto it.
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .overlay {
                        GeometryReader { geometry in
                            let rect = CGRect(origin: .zero, size: geometry.size)
                            ZStack(alignment: .topLeading) {
                                Canvas { context, size in
                                    let fitted = CGRect(origin: .zero, size: size)
                                    switch overlay {
                                    case .faces(let faces):
                                        drawFaces(faces, in: fitted, context: &context)
                                    case .pose(let pose):
                                        drawPose(pose, in: fitted, context: &context)
                                    }
                                }
                                if let selectedFaceIndex, case .faces(let faces) = overlay {
                                    faceButtons(faces, selection: selectedFaceIndex, in: rect)
                                }
                            }
                        }
                    }
            } else {
                Rectangle()
                    .fill(MereRunTheme.surfaceRaised)
                    .aspectRatio(4 / 3, contentMode: .fit)
                    .overlay {
                        if didLoad {
                            Image(systemName: "photo")
                                .font(.system(size: 32, weight: .semibold))
                                .foregroundStyle(MereRunTheme.textMuted)
                        } else {
                            ProgressView().controlSize(.small)
                        }
                    }
            }
        }
        .task(id: imageURL) {
            didLoad = false
            let url = imageURL
            let loaded = await Task.detached(priority: .userInitiated) {
                (StudioImagePreviewLoader.downsampledImage(from: url, maxPixelSize: 1_600)?.image,
                 StudioImageMetadata.read(url)?.orientation ?? .up)
            }.value
            guard !Task.isCancelled else { return }
            image = loaded.0
            orientation = loaded.1
            didLoad = true
        }
        .accessibilityElement(children: selectedFaceIndex == nil ? .ignore : .contain)
        .accessibilityLabel(accessibilityDescription)
    }

    private var accessibilityDescription: String {
        switch overlay {
        case .faces(let faces):
            return "\(imageURL.lastPathComponent) with \(faces.faces.count == 1 ? "1 face" : "\(faces.faces.count) faces") marked"
        case .pose(let pose):
            return "\(imageURL.lastPathComponent) with \(pose.summary)"
        }
    }

    // MARK: Geometry

    /// Where a stored-pixel point of a `storedSize` document lands in the fitted upright picture.
    private func viewPoint(_ point: CGPoint, storedSize: CGSize, in rect: CGRect) -> CGPoint {
        let shown = orientation.displaySize(ofStored: storedSize)
        let upright = orientation.displayPoint(fromStored: point, storedSize: storedSize)
        return CGPoint(
            x: rect.minX + upright.x * rect.width / max(1, shown.width),
            y: rect.minY + upright.y * rect.height / max(1, shown.height)
        )
    }

    private func faceFrame(_ face: StudioFaceOverlayResult.Record, result: StudioFaceOverlayResult, in rect: CGRect) -> CGRect {
        let storedSize = CGSize(width: max(1, result.width), height: max(1, result.height))
        let box = face.detection.boundingBox
        let start = viewPoint(CGPoint(x: box.x, y: box.y), storedSize: storedSize, in: rect)
        let end = viewPoint(CGPoint(x: box.x + box.width, y: box.y + box.height), storedSize: storedSize, in: rect)
        return StudioRegionGeometry.rect(from: start, to: end)
    }

    // MARK: Faces

    /// One transparent button per detected face, so a click picks it. The Canvas underneath
    /// draws the chosen one in the accent.
    private func faceButtons(_ result: StudioFaceOverlayResult, selection: Binding<Int>, in rect: CGRect) -> some View {
        ForEach(result.faces, id: \.index) { face in
            let frame = faceFrame(face, result: result, in: rect)
            Button {
                selection.wrappedValue = face.index
            } label: {
                Color.clear
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(width: max(frame.width, 8), height: max(frame.height, 8))
            .position(x: frame.midX, y: frame.midY)
            .help("Face \(face.index), \(Int(face.detection.score * 100))% confidence")
            .accessibilityLabel("Face \(face.index), \(Int(face.detection.score * 100)) percent")
            .accessibilityAddTraits(selection.wrappedValue == face.index ? .isSelected : [])
        }
    }

    private func drawFaces(_ result: StudioFaceOverlayResult, in rect: CGRect, context: inout GraphicsContext) {
        if let selectedFaceIndex {
            drawSelectableFaces(result, selected: selectedFaceIndex.wrappedValue, in: rect, context: &context)
            return
        }
        let storedSize = CGSize(width: max(1, result.width), height: max(1, result.height))
        for face in result.faces {
            let frame = faceFrame(face, result: result, in: rect)
            let path = Path(roundedRect: frame, cornerRadius: 4)
            context.stroke(path, with: .color(MereRunTheme.accent), lineWidth: 2)
            for point in face.detection.landmarks {
                let center = viewPoint(CGPoint(x: point.x, y: point.y), storedSize: storedSize, in: rect)
                context.fill(
                    Path(ellipseIn: CGRect(x: center.x - 3, y: center.y - 3, width: 6, height: 6)),
                    with: .color(MereRunTheme.yellow)
                )
            }
            drawTag(
                Text("Face \(face.index + 1) \(String(format: "%.2f", face.detection.score))"),
                above: frame, in: rect, fill: MereRunTheme.accent, foreground: MereRunTheme.onAccent, context: &context
            )
        }
    }

    /// The picker's rendering: the chosen face in the accent with a soft fill, the rest as quiet
    /// white outlines, each numbered the way `--face-index` counts them.
    private func drawSelectableFaces(
        _ result: StudioFaceOverlayResult,
        selected: Int,
        in rect: CGRect,
        context: inout GraphicsContext
    ) {
        for face in result.faces {
            let frame = faceFrame(face, result: result, in: rect)
            let isSelected = face.index == selected
            let path = Path(roundedRect: frame, cornerRadius: 3)
            if isSelected {
                context.fill(path, with: .color(MereRunTheme.accent.opacity(0.18)))
            }
            context.stroke(
                path,
                with: .color(isSelected ? MereRunTheme.accent : Color.white.opacity(0.85)),
                lineWidth: isSelected ? 2.5 : 1.5
            )
            drawTag(
                Text("\(face.index)"),
                above: frame, in: rect,
                fill: isSelected ? MereRunTheme.accent : Color.white.opacity(0.85),
                foreground: isSelected ? MereRunTheme.onAccent : Color.black.opacity(0.8),
                context: &context
            )
        }
    }

    /// A small label tab sitting on the top edge of `frame`, kept inside the picture.
    private func drawTag(
        _ text: Text,
        above frame: CGRect,
        in rect: CGRect,
        fill: Color,
        foreground: Color,
        context: inout GraphicsContext
    ) {
        let tag = text.font(.system(size: 10.5, weight: .semibold)).foregroundColor(foreground)
        let tagSize = context.resolve(tag).measure(in: CGSize(width: 160, height: 20))
        let tagRect = CGRect(
            x: frame.minX,
            y: max(rect.minY, frame.minY - tagSize.height - 4),
            width: tagSize.width + 8,
            height: tagSize.height + 3
        )
        context.fill(Path(roundedRect: tagRect, cornerRadius: 3), with: .color(fill))
        context.draw(tag, at: CGPoint(x: tagRect.midX, y: tagRect.midY), anchor: .center)
    }

    // MARK: Pose

    private func drawPose(_ result: StudioPoseOverlayResult, in rect: CGRect, context: inout GraphicsContext) {
        let storedSize = CGSize(width: max(1, result.imageWidth), height: max(1, result.imageHeight))
        for subject in result.subjects {
            let color = Self.color(forSubjectKind: subject.kind)
            for point in subject.points {
                let center = viewPoint(result.storedPoint(point), storedSize: storedSize, in: rect)
                context.fill(
                    Path(ellipseIn: CGRect(x: center.x - 2.5, y: center.y - 2.5, width: 5, height: 5)),
                    with: .color(color.opacity(max(0.25, point.confidence)))
                )
            }
        }
    }

    /// Body in the accent, hands in yellow, faces in green: the three kinds a pose run reports.
    static func color(forSubjectKind kind: String) -> Color {
        switch kind {
        case "body": return MereRunTheme.accent
        case "hand": return MereRunTheme.yellow
        default: return MereRunTheme.green
        }
    }
}

/// The picker the task inspector shows for `--face-index`: the picture with the newest Face
/// detection run's boxes on it, each a button. Loads the document itself so the inspector row
/// only has to know where it is.
struct StudioFacePickerView: View {
    let imageURL: URL
    let documentURL: URL
    @Binding var selection: Int

    @State private var faces: StudioFaceOverlayResult?

    var body: some View {
        Group {
            if let faces {
                StudioVisionOverlayView(imageURL: imageURL, overlay: .faces(faces), selectedFaceIndex: $selection)
            } else {
                ProgressView().controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: documentURL) { faces = StudioFaceOverlayResult.load(from: documentURL) }
    }
}

// MARK: - Result panel rows

/// One row per subject a pose run found: its kind, index, and how many landmarks it carries.
struct StudioPoseSubjectRows: View {
    let result: StudioPoseOverlayResult

    var body: some View {
        StudioResultRowList(count: result.subjects.count) {
            ForEach(Array(result.subjects.enumerated()), id: \.offset) { _, subject in
                HStack(spacing: 10) {
                    Circle()
                        .fill(StudioVisionOverlayView.color(forSubjectKind: subject.kind))
                        .frame(width: 10, height: 10)
                    Text("\(subject.kind.capitalized) \(subject.index + 1)")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("\(subject.points.count) landmarks")
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .fixedSize()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .accessibilityElement(children: .combine)
                StudioResultHairline()
            }
        }
    }
}

/// What an embedding run produced: which face, its score, and the vector's size and norm.
struct StudioFaceEmbeddingRows: View {
    let document: StudioFaceEmbeddingDocument

    private var norm: String {
        guard let embedding = document.face.embedding, !embedding.isEmpty else { return "—" }
        let sum = embedding.reduce(0) { $0 + Double($1) * Double($1) }
        return String(format: "%.3f", sum.squareRoot())
    }

    var body: some View {
        VStack(spacing: 0) {
            StudioResultFactRow("Face", "\(document.face.index + 1) of the picture")
            StudioResultFactRow("Score", String(format: "%.2f", document.face.detection.score))
            StudioResultFactRow("Vector", "\(document.face.embedding?.count ?? 0) dimensions")
            StudioResultFactRow("Norm", norm)
        }
    }
}

/// A comparison's verdict: the similarity large, then which faces were compared.
struct StudioFaceComparisonRows: View {
    let document: StudioFaceComparisonDocument

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(String(format: "%.3f", document.cosineSimilarity))
                    .font(.system(size: 28, weight: .semibold, design: .rounded))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text("cosine similarity")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .accessibilityElement(children: .combine)
            StudioResultHairline()
            StudioResultFactRow("Reference", "face \(document.referenceFaceIndex + 1) of \(URL(fileURLWithPath: document.referenceImage).lastPathComponent)")
            StudioResultFactRow("Candidate", "face \(document.candidateFaceIndex + 1) of \(URL(fileURLWithPath: document.candidateImage).lastPathComponent)")
        }
    }
}

/// One row per picture of a batch: its name, how many faces, or why it failed.
struct StudioFaceBatchRows: View {
    let document: StudioFaceBatchDocument

    var body: some View {
        StudioResultRowList(count: document.entries.count) {
            ForEach(Array(document.entries.enumerated()), id: \.offset) { _, entry in
                HStack(spacing: 10) {
                    Image(systemName: entry.ok ? "face.dashed" : "exclamationmark.triangle")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(entry.ok ? MereRunTheme.accent : MereRunTheme.red)
                        .frame(width: 14)
                    Text(URL(fileURLWithPath: entry.image).lastPathComponent)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .help(entry.image)
                    Text(detail(for: entry))
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(entry.ok ? MereRunTheme.textMuted : MereRunTheme.red)
                        .lineLimit(1)
                        .fixedSize()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .accessibilityElement(children: .combine)
                StudioResultHairline()
            }
        }
    }

    private func detail(for entry: StudioFaceBatchDocument.Entry) -> String {
        guard entry.ok, let result = entry.result else { return entry.error ?? "failed" }
        return result.faces.count == 1 ? "1 face" : "\(result.faces.count) faces"
    }
}
