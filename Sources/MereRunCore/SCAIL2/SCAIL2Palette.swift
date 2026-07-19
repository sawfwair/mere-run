import Foundation
import MediaIO

public enum SCAIL2PaletteError: LocalizedError, Equatable, Sendable {
    case dimensionMismatch(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
    case invalidBinaryMask(path: String)
    case ambiguousColor(red: UInt8, green: UInt8, blue: UInt8)
    case colorOutsideTolerance(red: UInt8, green: UInt8, blue: UInt8, tolerance: UInt8)

    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let expectedWidth, let expectedHeight, let actualWidth, let actualHeight):
            "Mask dimensions must be \(expectedWidth)x\(expectedHeight); received \(actualWidth)x\(actualHeight)."
        case .invalidBinaryMask(let path):
            "Binary correction mask is empty or invalid: \(path)."
        case .ambiguousColor(let red, let green, let blue):
            "Mask pixel rgb(\(red), \(green), \(blue)) is equally close to multiple legal SCAIL-2 colors."
        case .colorOutsideTolerance(let red, let green, let blue, let tolerance):
            "Mask pixel rgb(\(red), \(green), \(blue)) is outside palette tolerance \(tolerance)."
        }
    }
}

public struct SCAIL2PaletteComposition: Hashable, Sendable {
    public let image: MediaImage
    public let overlapPixelCount: Int

    public init(image: MediaImage, overlapPixelCount: Int) {
        self.image = image
        self.overlapPixelCount = overlapPixelCount
    }
}

enum SCAIL2MaskRole {
    case mainReference
    case additionalSubjectReference
    case driving

    func background(mode: SCAIL2Mode) -> (UInt8, UInt8, UInt8, UInt8) {
        switch (mode, self) {
        case (.animation, .mainReference), (.replacement, .driving):
            SCAIL2Palette.backgroundRGBA
        case (.animation, .driving),
             (.replacement, .mainReference),
             (_, .additionalSubjectReference):
            SCAIL2Palette.hiddenBackgroundRGBA
        }
    }
}

public enum SCAIL2Palette {
    public static let backgroundRGBA: (UInt8, UInt8, UInt8, UInt8) = (255, 255, 255, 255)
    public static let hiddenBackgroundRGBA: (UInt8, UInt8, UInt8, UInt8) = (0, 0, 0, 255)
    /// ProRes 4444 can darken categorical edge pixels during YCbCr conversion.
    /// This bound accepts that codec drift; the separation requirement still
    /// rejects pixels that cannot be assigned to one palette entry confidently.
    public static let codecTolerance: UInt8 = 192
    private static let minimumPaletteSeparation = 32

    public static func binaryMask(from image: MediaImage) throws -> [UInt8] {
        var result = [UInt8](repeating: 0, count: image.width * image.height)
        var active = 0
        for pixelIndex in result.indices {
            let offset = pixelIndex * 4
            let value = max(image.rgba8[offset], max(image.rgba8[offset + 1], image.rgba8[offset + 2]))
            if value >= 128 {
                result[pixelIndex] = 1
                active += 1
            }
        }
        guard active > 0 else {
            throw SCAIL2PaletteError.invalidBinaryMask(path: "<memory>")
        }
        return result
    }

    public static func compose(
        width: Int,
        height: Int,
        subjectMasks: [(color: SCAIL2SubjectColor, mask: [UInt8])]
    ) throws -> SCAIL2PaletteComposition {
        let pixelCount = width * height
        for entry in subjectMasks where entry.mask.count != pixelCount {
            throw SCAIL2PaletteError.dimensionMismatch(
                expectedWidth: width,
                expectedHeight: height,
                actualWidth: entry.mask.count,
                actualHeight: 1
            )
        }
        var rgba = [UInt8](repeating: 255, count: pixelCount * 4)
        var occupied = [Bool](repeating: false, count: pixelCount)
        var overlapPixelCount = 0
        for entry in subjectMasks {
            let color = entry.color.rgba8
            for pixelIndex in 0..<pixelCount where entry.mask[pixelIndex] != 0 {
                if occupied[pixelIndex] {
                    overlapPixelCount += 1
                    continue
                }
                occupied[pixelIndex] = true
                let offset = pixelIndex * 4
                rgba[offset] = color.0
                rgba[offset + 1] = color.1
                rgba[offset + 2] = color.2
                rgba[offset + 3] = color.3
            }
        }
        return SCAIL2PaletteComposition(
            image: try MediaImage(width: width, height: height, rgba8: rgba),
            overlapPixelCount: overlapPixelCount
        )
    }

    public static func snapped(_ image: MediaImage, tolerance: UInt8) throws -> MediaImage {
        let legal = [(name: "background", rgba: backgroundRGBA)] + SCAIL2SubjectColor.assignmentOrder.map {
            (name: $0.rawValue, rgba: $0.rgba8)
        }
        var rgba = image.rgba8
        for pixelIndex in 0..<(image.width * image.height) {
            let offset = pixelIndex * 4
            let red = rgba[offset]
            let green = rgba[offset + 1]
            let blue = rgba[offset + 2]
            if max(red, max(green, blue)) <= 24 {
                rgba[offset] = hiddenBackgroundRGBA.0
                rgba[offset + 1] = hiddenBackgroundRGBA.1
                rgba[offset + 2] = hiddenBackgroundRGBA.2
                rgba[offset + 3] = hiddenBackgroundRGBA.3
                continue
            }
            let distances = legal.map { entry -> Int in
                let dr = Int(red) - Int(entry.rgba.0)
                let dg = Int(green) - Int(entry.rgba.1)
                let db = Int(blue) - Int(entry.rgba.2)
                return max(abs(dr), max(abs(dg), abs(db)))
            }
            guard let minimum = distances.min(), minimum <= Int(tolerance) else {
                throw SCAIL2PaletteError.colorOutsideTolerance(
                    red: red,
                    green: green,
                    blue: blue,
                    tolerance: tolerance
                )
            }
            guard let selected = distances.firstIndex(of: minimum) else {
                throw SCAIL2PaletteError.ambiguousColor(red: red, green: green, blue: blue)
            }
            let nextMinimum = distances.enumerated()
                .filter { $0.offset != selected }
                .map(\.element)
                .min() ?? Int.max
            guard nextMinimum - minimum >= minimumPaletteSeparation else {
                throw SCAIL2PaletteError.ambiguousColor(red: red, green: green, blue: blue)
            }
            rgba[offset] = legal[selected].rgba.0
            rgba[offset + 1] = legal[selected].rgba.1
            rgba[offset + 2] = legal[selected].rgba.2
            rgba[offset + 3] = 255
        }
        return try MediaImage(width: image.width, height: image.height, rgba8: rgba)
    }

    public static func subjectColors(
        in image: MediaImage,
        tolerance: UInt8
    ) throws -> Set<SCAIL2SubjectColor> {
        let snapped = try snapped(image, tolerance: tolerance)
        let colorsByRGB: [Int: SCAIL2SubjectColor] = Dictionary(
            uniqueKeysWithValues: SCAIL2SubjectColor.assignmentOrder.map { color in
                let rgba = color.rgba8
                return ((Int(rgba.0) << 16) | (Int(rgba.1) << 8) | Int(rgba.2), color)
            }
        )
        var result = Set<SCAIL2SubjectColor>()
        for pixelIndex in 0..<(snapped.width * snapped.height) {
            let offset = pixelIndex * 4
            let key = (Int(snapped.rgba8[offset]) << 16)
                | (Int(snapped.rgba8[offset + 1]) << 8)
                | Int(snapped.rgba8[offset + 2])
            if let color = colorsByRGB[key] {
                result.insert(color)
            }
        }
        return result
    }

    public static func overlay(
        image: MediaImage,
        paletteMask: MediaImage,
        alpha: Float = 0.45
    ) throws -> MediaImage {
        guard image.width == paletteMask.width, image.height == paletteMask.height else {
            throw SCAIL2PaletteError.dimensionMismatch(
                expectedWidth: image.width,
                expectedHeight: image.height,
                actualWidth: paletteMask.width,
                actualHeight: paletteMask.height
            )
        }
        let clampedAlpha = max(0, min(1, alpha))
        var rgba = image.rgba8
        for pixelIndex in 0..<(image.width * image.height) {
            let offset = pixelIndex * 4
            let maskRGB = (
                paletteMask.rgba8[offset],
                paletteMask.rgba8[offset + 1],
                paletteMask.rgba8[offset + 2]
            )
            guard maskRGB != (255, 255, 255), maskRGB != (0, 0, 0) else { continue }
            rgba[offset] = UInt8(
                Float(rgba[offset]) * (1 - clampedAlpha) + Float(maskRGB.0) * clampedAlpha
            )
            rgba[offset + 1] = UInt8(
                Float(rgba[offset + 1]) * (1 - clampedAlpha) + Float(maskRGB.1) * clampedAlpha
            )
            rgba[offset + 2] = UInt8(
                Float(rgba[offset + 2]) * (1 - clampedAlpha) + Float(maskRGB.2) * clampedAlpha
            )
        }
        return try MediaImage(width: image.width, height: image.height, rgba8: rgba)
    }
}
