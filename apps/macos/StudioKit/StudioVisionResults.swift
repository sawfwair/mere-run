import CoreGraphics
import Foundation

// The result documents of the Vision tasks moving onto the Analyze canvas (`vision pose`,
// `vision flow`), decoded here so the overlay and flow renderers in StudioUI draw from typed
// values. `StudioFaceOverlayResult` lives beside them in `StudioSpecialistResults.swift`.

// MARK: - vision pose

/// What `mere.run vision pose --json-output` writes: the picture's stored size, the coordinate
/// space the points are in, and one subject per body, hand, or face with its named landmarks.
package struct StudioPoseOverlayResult: Decodable, Equatable {
    package struct Subject: Decodable, Equatable {
        package struct Point: Decodable, Equatable {
            package let name: String
            package let x: Double
            package let y: Double
            package let confidence: Double
        }

        /// "body", "hand", or "face".
        package let kind: String
        package let index: Int
        package let points: [Point]
    }

    package let imageWidth: Int
    package let imageHeight: Int
    /// "normalized" (origin top-left) or "normalized-bottom-left".
    package let coordinateSpace: String
    package let subjects: [Subject]

    package static func load(from url: URL) -> StudioPoseOverlayResult? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }

    /// Where a landmark sits in the picture's stored pixels, origin top-left, whichever space
    /// the document declares.
    package func storedPoint(_ point: Subject.Point) -> CGPoint {
        let normalizedY = coordinateSpace == "normalized-bottom-left" ? 1 - point.y : point.y
        return CGPoint(x: point.x * Double(max(1, imageWidth)), y: normalizedY * Double(max(1, imageHeight)))
    }

    /// "2 subjects · 51 landmarks"
    package var summary: String {
        let subjectCount = subjects.count == 1 ? "1 subject" : "\(subjects.count) subjects"
        let landmarks = subjects.reduce(0) { $0 + $1.points.count }
        return "\(subjectCount) · \(landmarks) landmarks"
    }
}

// MARK: - vision flow

/// What `mere.run vision flow` writes: a Middlebury `.flo` file — the magic float `202021.25`,
/// the width and height as 32-bit integers, then one (dx, dy) float pair per pixel, row-major.
package struct StudioFlowField: Equatable {
    package let width: Int
    package let height: Int
    package let vectors: [SIMD2<Float>]

    package init(width: Int, height: Int, vectors: [SIMD2<Float>]) {
        self.width = width
        self.height = height
        self.vectors = vectors
    }

    package static func load(url: URL) throws -> StudioFlowField {
        try decode(Data(contentsOf: url))
    }

    package static func decode(_ data: Data) throws -> StudioFlowField {
        guard data.count >= 12 else { throw CocoaError(.fileReadCorruptFile) }
        func uint32(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { raw in
                UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
        }
        let magic = Float(bitPattern: uint32(0))
        let width = Int(Int32(bitPattern: uint32(4)))
        let height = Int(Int32(bitPattern: uint32(8)))
        guard abs(magic - 202_021.25) < 0.01, width > 0, height > 0,
              data.count >= 12 + width * height * 8 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var vectors: [SIMD2<Float>] = []
        vectors.reserveCapacity(width * height)
        var offset = 12
        for _ in 0..<(width * height) {
            vectors.append(SIMD2(Float(bitPattern: uint32(offset)), Float(bitPattern: uint32(offset + 4))))
            offset += 8
        }
        return StudioFlowField(width: width, height: height, vectors: vectors)
    }

    /// "640×360 dense flow"
    package var summary: String {
        "\(width)×\(height) dense flow"
    }
}
