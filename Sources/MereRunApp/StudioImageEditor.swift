import AppKit
import SwiftUI

private struct StudioMaskStroke: Identifiable {
    let id = UUID()
    var points: [CGPoint]
    let radius: CGFloat
    let erases: Bool
}

struct StudioImageEditor: View {
    let inputURL: URL
    let outputWidth: Int
    let outputHeight: Int
    @Binding var maskPath: String
    @Binding var outpaintTop: Int
    @Binding var outpaintRight: Int
    @Binding var outpaintBottom: Int
    @Binding var outpaintLeft: Int
    @Binding var feather: Int

    @Environment(\.dismiss) private var dismiss
    @State private var sourceImage: NSImage?
    @State private var strokes: [StudioMaskStroke] = []
    @State private var activeStroke: StudioMaskStroke?
    @State private var brushSize: Double = 72
    @State private var erasing = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.5))

            HStack(spacing: 0) {
                editorCanvas
                    .frame(minWidth: 560, minHeight: 480)

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                controls
                    .frame(width: 280)
            }

            Divider().overlay(MereRunTheme.border.opacity(0.5))
            footer
        }
        .frame(width: 920, height: 680)
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear {
            sourceImage = NSImage(contentsOf: inputURL)
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Image edit")
                    .font(.system(size: 17, weight: .semibold))
                Text("Paint what may change. Unpainted pixels are restored exactly.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Spacer()
            Text(inputURL.lastPathComponent)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
        }
        .padding(MereRunTheme.Spacing.lg)
    }

    @ViewBuilder
    private var editorCanvas: some View {
        if let sourceImage {
            GeometryReader { proxy in
                let imageSize = sourceImage.size
                let rect = fittedRect(imageSize: imageSize, in: proxy.size)
                ZStack {
                    MereRunTheme.surface
                    Image(nsImage: sourceImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)

                    Canvas { context, _ in
                        for stroke in strokes + [activeStroke].compactMap({ $0 }) {
                            draw(stroke, in: rect, context: &context)
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(maskGesture(in: rect))
                }
            }
            .padding(MereRunTheme.Spacing.lg)
        } else {
            ContentUnavailableView(
                "Could not open image",
                systemImage: "photo.badge.exclamationmark",
                description: Text(inputURL.path)
            )
        }
    }

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                    Text("Mask brush")
                        .font(MereRunTheme.sectionFont)
                    Picker("Brush", selection: $erasing) {
                        Text("Paint").tag(false)
                        Text("Erase").tag(true)
                    }
                    .pickerStyle(.segmented)
                    Text("Size \(Int(brushSize)) px")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Slider(value: $brushSize, in: 8...240)
                    HStack {
                        Button("Undo") {
                            if !strokes.isEmpty { strokes.removeLast() }
                        }
                        .disabled(strokes.isEmpty)
                        Button("Clear") { strokes.removeAll() }
                            .disabled(strokes.isEmpty)
                    }
                }

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
                    Text("Outpaint")
                        .font(MereRunTheme.sectionFont)
                    Text("Add editable space inside the requested output size.")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    paddingStepper("Top", value: $outpaintTop)
                    paddingStepper("Right", value: $outpaintRight)
                    paddingStepper("Bottom", value: $outpaintBottom)
                    paddingStepper("Left", value: $outpaintLeft)
                }

                Divider().overlay(MereRunTheme.border.opacity(0.5))

                Stepper("Edge feather \(feather) px", value: $feather, in: 0...128)
                    .font(MereRunTheme.captionFont)

                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.red)
                }

                if !outpaintFits {
                    Label(
                        "Padding must leave space inside the \(outputWidth)×\(outputHeight) output.",
                        systemImage: "exclamationmark.triangle.fill"
                    )
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.red)
                }

                if !maskPath.isEmpty {
                    Label(URL(fileURLWithPath: maskPath).lastPathComponent, systemImage: "checkmark.circle.fill")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.green)
                }
            }
            .padding(MereRunTheme.Spacing.lg)
        }
    }

    private var footer: some View {
        HStack {
            Button("Cancel") { dismiss() }
            Spacer()
            Button("Remove edit") {
                maskPath = ""
                outpaintTop = 0
                outpaintRight = 0
                outpaintBottom = 0
                outpaintLeft = 0
                dismiss()
            }
            .disabled(maskPath.isEmpty && !hasOutpaint)
            Button("Apply edit") { saveAndDismiss() }
                .keyboardShortcut(.defaultAction)
                .disabled(
                    sourceImage == nil
                        || !outpaintFits
                        || (strokes.isEmpty && !hasOutpaint && maskPath.isEmpty)
                )
        }
        .padding(MereRunTheme.Spacing.lg)
    }

    private var hasOutpaint: Bool {
        [outpaintTop, outpaintRight, outpaintBottom, outpaintLeft].contains(where: { $0 > 0 })
    }

    private var outpaintFits: Bool {
        outpaintLeft + outpaintRight < outputWidth
            && outpaintTop + outpaintBottom < outputHeight
    }

    private func paddingStepper(_ title: String, value: Binding<Int>) -> some View {
        Stepper("\(title) \(value.wrappedValue) px", value: value, in: 0...2048, step: 64)
            .font(MereRunTheme.captionFont)
    }

    private func fittedRect(imageSize: CGSize, in available: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(available.width / imageSize.width, available.height / imageSize.height)
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(
            x: (available.width - size.width) / 2,
            y: (available.height - size.height) / 2,
            width: size.width,
            height: size.height
        )
    }

    private func normalized(_ point: CGPoint, in rect: CGRect) -> CGPoint? {
        guard rect.contains(point), rect.width > 0, rect.height > 0 else { return nil }
        return CGPoint(
            x: (point.x - rect.minX) / rect.width,
            y: (point.y - rect.minY) / rect.height
        )
    }

    private func maskGesture(in rect: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard let point = normalized(value.location, in: rect) else { return }
                if activeStroke == nil {
                    activeStroke = StudioMaskStroke(
                        points: [point],
                        radius: CGFloat(brushSize) / max(rect.width, rect.height),
                        erases: erasing
                    )
                } else {
                    activeStroke?.points.append(point)
                }
            }
            .onEnded { _ in
                if let activeStroke {
                    strokes.append(activeStroke)
                    self.activeStroke = nil
                }
            }
    }

    private func draw(_ stroke: StudioMaskStroke, in rect: CGRect, context: inout GraphicsContext) {
        guard let first = stroke.points.first else { return }
        let lineWidth = max(2, stroke.radius * max(rect.width, rect.height))
        let color = stroke.erases ? Color.black.opacity(0.5) : MereRunTheme.accent.opacity(0.65)
        if stroke.points.count == 1 {
            let center = displayPoint(first, in: rect)
            context.fill(
                Path(
                    ellipseIn: CGRect(
                        x: center.x - (lineWidth / 2),
                        y: center.y - (lineWidth / 2),
                        width: lineWidth,
                        height: lineWidth
                    )
                ),
                with: .color(color)
            )
            return
        }
        var path = Path()
        path.move(to: displayPoint(first, in: rect))
        for point in stroke.points.dropFirst() {
            path.addLine(to: displayPoint(point, in: rect))
        }
        context.stroke(
            path,
            with: .color(color),
            style: StrokeStyle(
                lineWidth: lineWidth,
                lineCap: .round,
                lineJoin: .round
            )
        )
    }

    private func displayPoint(_ point: CGPoint, in rect: CGRect) -> CGPoint {
        CGPoint(x: rect.minX + (point.x * rect.width), y: rect.minY + (point.y * rect.height))
    }

    private func saveAndDismiss() {
        guard let sourceImage else { return }
        do {
            if !strokes.isEmpty {
                maskPath = try writeMask(for: sourceImage).path
            }
            dismiss()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func writeMask(for sourceImage: NSImage) throws -> URL {
        guard let representation = sourceImage.representations.first else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let width = max(1, representation.pixelsWide)
        let height = max(1, representation.pixelsHigh)
        guard let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bitmapFormat: [],
            bytesPerRow: width * 4,
            bitsPerPixel: 32
        ) else {
            throw CocoaError(.fileWriteUnknown)
        }

        NSGraphicsContext.saveGraphicsState()
        guard let graphicsContext = NSGraphicsContext(bitmapImageRep: bitmap) else {
            NSGraphicsContext.restoreGraphicsState()
            throw CocoaError(.fileWriteUnknown)
        }
        NSGraphicsContext.current = graphicsContext
        NSColor.black.setFill()
        NSBezierPath(rect: CGRect(x: 0, y: 0, width: width, height: height)).fill()
        for stroke in strokes {
            guard let first = stroke.points.first else { continue }
            let path = NSBezierPath()
            path.lineCapStyle = .round
            path.lineJoinStyle = .round
            path.lineWidth = max(1, stroke.radius * CGFloat(max(width, height)))
            path.move(to: maskPoint(first, width: width, height: height))
            if stroke.points.count == 1 {
                let center = maskPoint(first, width: width, height: height)
                let diameter = path.lineWidth
                (stroke.erases ? NSColor.black : NSColor.white).setFill()
                NSBezierPath(
                    ovalIn: CGRect(
                        x: center.x - (diameter / 2),
                        y: center.y - (diameter / 2),
                        width: diameter,
                        height: diameter
                    )
                ).fill()
            } else {
                for point in stroke.points.dropFirst() {
                    path.line(to: maskPoint(point, width: width, height: height))
                }
                (stroke.erases ? NSColor.black : NSColor.white).setStroke()
                path.stroke()
            }
        }
        graphicsContext.flushGraphics()
        NSGraphicsContext.restoreGraphicsState()

        let root = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        .appendingPathComponent("MereRun/Image Edits/\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("mask.png")
        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try data.write(to: url, options: .atomic)
        return url
    }

    private func maskPoint(_ point: CGPoint, width: Int, height: Int) -> NSPoint {
        NSPoint(x: point.x * CGFloat(width), y: (1 - point.y) * CGFloat(height))
    }
}
