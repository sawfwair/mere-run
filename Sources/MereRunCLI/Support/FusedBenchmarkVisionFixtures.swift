import Foundation
import MediaIO
import MereRunCore

enum FusedBenchmarkVisionFixtures {
    static func resolve(
        descriptor: FusedBenchmarkCaseDescriptor,
        source: FusedBenchmarkSource
    ) throws -> FusedExternalBenchmarkCase {
        let specification = try specification(for: descriptor.sourceCaseID)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-fused-vision-\(source.version)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let imageURL = directory.appendingPathComponent(specification.fileName)
        try render(specification.kind, to: imageURL)

        return try FusedExternalBenchmarkCase(
            id: descriptor.id,
            kind: .vision,
            sourceVersion: source.version,
            originalID: descriptor.sourceCaseID,
            messages: [
                ChatMessage(
                    role: .system,
                    content: "Answer from the supplied image. Report absent or conflicting visual evidence explicitly."
                ),
                ChatMessage(
                    role: .user,
                    content: specification.prompt,
                    imageUrl: imageURL.path
                ),
            ],
            tools: nil,
            requiredPhrases: specification.requiredPhrases,
            forbiddenPhrases: specification.forbiddenPhrases,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: nil,
            tests: nil,
            imageSHA256: nil,
            contentSHA256: ""
        ).stamped()
    }

    private static func specification(
        for sourceCaseID: String
    ) throws -> FusedBenchmarkVisionSpecification {
        guard let specification = specifications[sourceCaseID] else {
            throw FusedBenchmarkError.invalidManifest(
                "missing generated vision fixture \(sourceCaseID)"
            )
        }
        return specification
    }

    private static let specifications: [String: FusedBenchmarkVisionSpecification] = [
        "MereVision/OCRGrounding": FusedBenchmarkVisionSpecification(
            kind: .ocrGrounding,
            fileName: "ocr-grounding.png",
            prompt: "Read the access code inside the blue card. Return the exact code only.",
            requiredPhrases: ["MERE42"],
            forbiddenPhrases: ["MERE24"]
        ),
        "MereVision/Conflict": FusedBenchmarkVisionSpecification(
            kind: .conflict,
            fileName: "conflicting-status.png",
            prompt: "Do the two status panels agree? Name both visible labels and describe the evidence.",
            requiredPhrases: ["READY", "STOPPED", "conflict"],
            forbiddenPhrases: ["both ready", "both stopped"]
        ),
        "MereVision/ChartExtraction": FusedBenchmarkVisionSpecification(
            kind: .chart,
            fileName: "bar-chart.png",
            prompt: "Which labeled bar is highest, and what value is printed for it?",
            requiredPhrases: ["B", "70"],
            forbiddenPhrases: ["A is highest", "C is highest"]
        ),
        "MereVision/SpatialRelation": FusedBenchmarkVisionSpecification(
            kind: .spatial,
            fileName: "spatial-relation.png",
            prompt: "Where is the blue square relative to the red square? Answer with one direction word.",
            requiredPhrases: ["right"],
            forbiddenPhrases: ["left"]
        ),
        "MereVision/Counting": FusedBenchmarkVisionSpecification(
            kind: .counting,
            fileName: "object-count.png",
            prompt: "How many black squares are visible? Answer with one digit.",
            requiredPhrases: ["5"],
            forbiddenPhrases: ["4", "6"]
        ),
        "MereVision/DocumentLayout": FusedBenchmarkVisionSpecification(
            kind: .documentLayout,
            fileName: "document-layout.png",
            prompt: "Read the total shown in the bottom-right total box. Answer with the digits only.",
            requiredPhrases: ["4812"],
            forbiddenPhrases: ["1178"]
        ),
        "MereVision/NegativeEvidence": FusedBenchmarkVisionSpecification(
            kind: .negativeEvidence,
            fileName: "negative-evidence.png",
            prompt: "Is a star visible in the image? Answer no if it is absent.",
            requiredPhrases: ["no"],
            forbiddenPhrases: ["star is visible", "yes"]
        ),
        "MereVision/MultiImageConsistency": FusedBenchmarkVisionSpecification(
            kind: .multiPanel,
            fileName: "multi-panel-consistency.png",
            prompt: "What color and shape appear in both the left and right panels?",
            requiredPhrases: ["red", "circle"],
            forbiddenPhrases: ["blue circle", "green circle"]
        ),
        "MereVision/DenseCaption": FusedBenchmarkVisionSpecification(
            kind: .denseCaption,
            fileName: "dense-caption.png",
            prompt: "Describe the green triangle's position relative to the blue square.",
            requiredPhrases: ["green", "triangle", "above", "blue", "square"],
            forbiddenPhrases: ["below"]
        ),
        "MereVision/GroundedActionBoundary": FusedBenchmarkVisionSpecification(
            kind: .actionBoundary,
            fileName: "action-boundary.png",
            prompt: "Has this item already been published? Answer from the visible status, not the button label.",
            requiredPhrases: ["no", "draft"],
            forbiddenPhrases: ["published successfully", "already published"]
        ),
    ]

    private static func render(
        _ kind: FusedBenchmarkVisionKind,
        to url: URL
    ) throws {
        var canvas = FusedBitmapCanvas(width: 384, height: 256, fill: .paper)
        switch kind {
        case .ocrGrounding:
            canvas.fillRect(x: 46, y: 72, width: 292, height: 112, color: .blue)
            canvas.strokeRect(x: 46, y: 72, width: 292, height: 112, thickness: 4, color: .navy)
            canvas.drawText("ACCESS CODE", x: 92, y: 88, scale: 3, color: .white)
            canvas.drawText("MERE42", x: 96, y: 126, scale: 5, color: .white)
        case .conflict:
            canvas.fillRect(x: 20, y: 48, width: 164, height: 160, color: .paleGreen)
            canvas.fillRect(x: 200, y: 48, width: 164, height: 160, color: .paleRed)
            canvas.strokeRect(x: 20, y: 48, width: 164, height: 160, thickness: 4, color: .green)
            canvas.strokeRect(x: 200, y: 48, width: 164, height: 160, thickness: 4, color: .red)
            canvas.drawText("STATUS", x: 54, y: 76, scale: 3, color: .black)
            canvas.drawText("READY", x: 46, y: 126, scale: 5, color: .green)
            canvas.drawText("STATUS", x: 234, y: 76, scale: 3, color: .black)
            canvas.drawText("STOPPED", x: 208, y: 126, scale: 4, color: .red)
        case .chart:
            canvas.drawText("VALUE", x: 16, y: 12, scale: 3, color: .black)
            canvas.fillRect(x: 54, y: 146, width: 62, height: 60, color: .green)
            canvas.fillRect(x: 160, y: 66, width: 62, height: 140, color: .blue)
            canvas.fillRect(x: 266, y: 106, width: 62, height: 100, color: .red)
            canvas.drawText("A 30", x: 49, y: 216, scale: 3, color: .black)
            canvas.drawText("B 70", x: 155, y: 216, scale: 3, color: .black)
            canvas.drawText("C 50", x: 261, y: 216, scale: 3, color: .black)
        case .spatial:
            canvas.fillRect(x: 50, y: 86, width: 92, height: 92, color: .red)
            canvas.fillRect(x: 242, y: 86, width: 92, height: 92, color: .blue)
            canvas.drawText("RED", x: 65, y: 190, scale: 3, color: .red)
            canvas.drawText("BLUE", x: 250, y: 190, scale: 3, color: .blue)
        case .counting:
            for x in [28, 98, 168, 238, 308] {
                canvas.fillRect(x: x, y: 96, width: 48, height: 48, color: .black)
            }
            canvas.fillCircle(centerX: 192, centerY: 190, radius: 24, color: .yellow)
        case .documentLayout:
            canvas.strokeRect(x: 24, y: 18, width: 336, height: 220, thickness: 3, color: .navy)
            canvas.drawText("INVOICE 1178", x: 42, y: 34, scale: 3, color: .navy)
            canvas.drawText("ITEM A", x: 42, y: 88, scale: 3, color: .black)
            canvas.drawText("ITEM B", x: 42, y: 122, scale: 3, color: .black)
            canvas.fillRect(x: 188, y: 168, width: 152, height: 52, color: .paleBlue)
            canvas.strokeRect(x: 188, y: 168, width: 152, height: 52, thickness: 3, color: .blue)
            canvas.drawText("TOTAL 4812", x: 198, y: 184, scale: 3, color: .black)
        case .negativeEvidence:
            canvas.fillCircle(centerX: 118, centerY: 128, radius: 56, color: .blue)
            canvas.fillRect(x: 224, y: 72, width: 112, height: 112, color: .green)
            canvas.drawText("CIRCLE", x: 70, y: 204, scale: 3, color: .blue)
            canvas.drawText("SQUARE", x: 238, y: 204, scale: 3, color: .green)
        case .multiPanel:
            canvas.strokeRect(x: 18, y: 26, width: 164, height: 204, thickness: 3, color: .navy)
            canvas.strokeRect(x: 202, y: 26, width: 164, height: 204, thickness: 3, color: .navy)
            canvas.drawText("LEFT", x: 58, y: 42, scale: 3, color: .black)
            canvas.drawText("RIGHT", x: 232, y: 42, scale: 3, color: .black)
            canvas.fillCircle(centerX: 100, centerY: 144, radius: 48, color: .red)
            canvas.fillCircle(centerX: 284, centerY: 144, radius: 48, color: .red)
        case .denseCaption:
            canvas.fillTriangle(
                x1: 192,
                y1: 38,
                x2: 132,
                y2: 126,
                x3: 252,
                y3: 126,
                color: .green
            )
            canvas.fillRect(x: 138, y: 154, width: 108, height: 76, color: .blue)
            canvas.fillRect(x: 32, y: 166, width: 58, height: 58, color: .red)
        case .actionBoundary:
            canvas.strokeRect(x: 42, y: 28, width: 300, height: 202, thickness: 4, color: .navy)
            canvas.drawText("STATUS DRAFT", x: 72, y: 64, scale: 4, color: .black)
            canvas.fillRect(x: 104, y: 134, width: 176, height: 62, color: .blue)
            canvas.drawText("PUBLISH", x: 126, y: 154, scale: 4, color: .white)
        }
        try canvas.writePNG(to: url)
    }
}

private struct FusedBenchmarkVisionSpecification {
    let kind: FusedBenchmarkVisionKind
    let fileName: String
    let prompt: String
    let requiredPhrases: [String]
    let forbiddenPhrases: [String]
}

private enum FusedBenchmarkVisionKind {
    case ocrGrounding
    case conflict
    case chart
    case spatial
    case counting
    case documentLayout
    case negativeEvidence
    case multiPanel
    case denseCaption
    case actionBoundary
}

private struct FusedBitmapColor {
    let red: UInt8
    let green: UInt8
    let blue: UInt8
    let alpha: UInt8

    static let black = FusedBitmapColor(red: 22, green: 26, blue: 32, alpha: 255)
    static let blue = FusedBitmapColor(red: 30, green: 102, blue: 230, alpha: 255)
    static let green = FusedBitmapColor(red: 18, green: 150, blue: 86, alpha: 255)
    static let navy = FusedBitmapColor(red: 26, green: 48, blue: 86, alpha: 255)
    static let paleBlue = FusedBitmapColor(red: 220, green: 235, blue: 255, alpha: 255)
    static let paleGreen = FusedBitmapColor(red: 221, green: 248, blue: 232, alpha: 255)
    static let paleRed = FusedBitmapColor(red: 255, green: 226, blue: 226, alpha: 255)
    static let paper = FusedBitmapColor(red: 248, green: 247, blue: 242, alpha: 255)
    static let red = FusedBitmapColor(red: 220, green: 46, blue: 54, alpha: 255)
    static let white = FusedBitmapColor(red: 255, green: 255, blue: 255, alpha: 255)
    static let yellow = FusedBitmapColor(red: 244, green: 194, blue: 46, alpha: 255)
}

private struct FusedBitmapCanvas {
    let width: Int
    let height: Int
    private var rgba: [UInt8]

    init(width: Int, height: Int, fill: FusedBitmapColor) {
        self.width = width
        self.height = height
        self.rgba = [UInt8](repeating: 0, count: width * height * 4)
        fillRect(x: 0, y: 0, width: width, height: height, color: fill)
    }

    mutating func fillRect(
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        color: FusedBitmapColor
    ) {
        for pixelY in max(0, y)..<min(self.height, y + height) {
            for pixelX in max(0, x)..<min(self.width, x + width) {
                setPixel(x: pixelX, y: pixelY, color: color)
            }
        }
    }

    mutating func strokeRect(
        x: Int,
        y: Int,
        width: Int,
        height: Int,
        thickness: Int,
        color: FusedBitmapColor
    ) {
        fillRect(x: x, y: y, width: width, height: thickness, color: color)
        fillRect(x: x, y: y + height - thickness, width: width, height: thickness, color: color)
        fillRect(x: x, y: y, width: thickness, height: height, color: color)
        fillRect(x: x + width - thickness, y: y, width: thickness, height: height, color: color)
    }

    mutating func fillCircle(
        centerX: Int,
        centerY: Int,
        radius: Int,
        color: FusedBitmapColor
    ) {
        for y in (centerY - radius)...(centerY + radius) {
            for x in (centerX - radius)...(centerX + radius) {
                let dx = x - centerX
                let dy = y - centerY
                if (dx * dx) + (dy * dy) <= radius * radius {
                    setPixel(x: x, y: y, color: color)
                }
            }
        }
    }

    mutating func fillTriangle(
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
        x3: Int,
        y3: Int,
        color: FusedBitmapColor
    ) {
        let minimumX = min(x1, min(x2, x3))
        let maximumX = max(x1, max(x2, x3))
        let minimumY = min(y1, min(y2, y3))
        let maximumY = max(y1, max(y2, y3))
        for y in minimumY...maximumY {
            for x in minimumX...maximumX where Self.isInsideTriangle(
                x: x,
                y: y,
                x1: x1,
                y1: y1,
                x2: x2,
                y2: y2,
                x3: x3,
                y3: y3
            ) {
                setPixel(x: x, y: y, color: color)
            }
        }
    }

    mutating func drawText(
        _ text: String,
        x: Int,
        y: Int,
        scale: Int,
        color: FusedBitmapColor
    ) {
        var cursor = x
        for character in text.uppercased() {
            let rows = Self.glyphs[character] ?? Self.glyphs["?"]!
            for (rowIndex, row) in rows.enumerated() {
                for column in 0..<5 where row & (1 << (4 - column)) != 0 {
                    fillRect(
                        x: cursor + (column * scale),
                        y: y + (rowIndex * scale),
                        width: scale,
                        height: scale,
                        color: color
                    )
                }
            }
            cursor += 6 * scale
        }
    }

    func writePNG(to url: URL) throws {
        try MediaImageIO.writePNG(
            try MediaImage(width: width, height: height, rgba8: rgba),
            to: url
        )
    }

    private mutating func setPixel(x: Int, y: Int, color: FusedBitmapColor) {
        guard x >= 0, y >= 0, x < width, y < height else { return }
        let offset = ((y * width) + x) * 4
        rgba[offset] = color.red
        rgba[offset + 1] = color.green
        rgba[offset + 2] = color.blue
        rgba[offset + 3] = color.alpha
    }

    private static func isInsideTriangle(
        x: Int,
        y: Int,
        x1: Int,
        y1: Int,
        x2: Int,
        y2: Int,
        x3: Int,
        y3: Int
    ) -> Bool {
        let first = ((x - x2) * (y1 - y2)) - ((x1 - x2) * (y - y2))
        let second = ((x - x3) * (y2 - y3)) - ((x2 - x3) * (y - y3))
        let third = ((x - x1) * (y3 - y1)) - ((x3 - x1) * (y - y1))
        return (first >= 0 && second >= 0 && third >= 0)
            || (first <= 0 && second <= 0 && third <= 0)
    }

    private static let glyphs: [Character: [UInt8]] = [
        " ": [0, 0, 0, 0, 0, 0, 0],
        "-": [0, 0, 0, 31, 0, 0, 0],
        ":": [0, 4, 4, 0, 4, 4, 0],
        "?": [14, 17, 1, 2, 4, 0, 4],
        "0": [14, 17, 19, 21, 25, 17, 14],
        "1": [4, 12, 4, 4, 4, 4, 14],
        "2": [14, 17, 1, 2, 4, 8, 31],
        "3": [30, 1, 1, 14, 1, 1, 30],
        "4": [2, 6, 10, 18, 31, 2, 2],
        "5": [31, 16, 16, 30, 1, 1, 30],
        "6": [14, 16, 16, 30, 17, 17, 14],
        "7": [31, 1, 2, 4, 8, 8, 8],
        "8": [14, 17, 17, 14, 17, 17, 14],
        "9": [14, 17, 17, 15, 1, 1, 14],
        "A": [14, 17, 17, 31, 17, 17, 17],
        "B": [30, 17, 17, 30, 17, 17, 30],
        "C": [14, 17, 16, 16, 16, 17, 14],
        "D": [30, 17, 17, 17, 17, 17, 30],
        "E": [31, 16, 16, 30, 16, 16, 31],
        "F": [31, 16, 16, 30, 16, 16, 16],
        "G": [14, 17, 16, 23, 17, 17, 15],
        "H": [17, 17, 17, 31, 17, 17, 17],
        "I": [14, 4, 4, 4, 4, 4, 14],
        "J": [7, 2, 2, 2, 18, 18, 12],
        "K": [17, 18, 20, 24, 20, 18, 17],
        "L": [16, 16, 16, 16, 16, 16, 31],
        "M": [17, 27, 21, 21, 17, 17, 17],
        "N": [17, 25, 21, 19, 17, 17, 17],
        "O": [14, 17, 17, 17, 17, 17, 14],
        "P": [30, 17, 17, 30, 16, 16, 16],
        "Q": [14, 17, 17, 17, 21, 18, 13],
        "R": [30, 17, 17, 30, 20, 18, 17],
        "S": [15, 16, 16, 14, 1, 1, 30],
        "T": [31, 4, 4, 4, 4, 4, 4],
        "U": [17, 17, 17, 17, 17, 17, 14],
        "V": [17, 17, 17, 17, 17, 10, 4],
        "W": [17, 17, 17, 21, 21, 27, 17],
        "X": [17, 17, 10, 4, 10, 17, 17],
        "Y": [17, 17, 10, 4, 4, 4, 4],
        "Z": [31, 1, 2, 4, 8, 16, 31],
    ]
}
