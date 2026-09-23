import Foundation

// Vision ▸ Geometry (multi-view) and 3D ▸ InstantMesh take optional calibrated cameras as a JSON
// file. Studio edits them as one camera per view and writes the file each run needs beside the run's
// output; a file made elsewhere still imports. Both documents carry `schemaVersion` 1 and one camera
// per image, in view order.

// MARK: - DA3 multi-view geometry

/// A calibrated camera for `vision geometry-multiview --cameras`. Mirrors `DepthAnything3KnownCamera`
/// in `MereRunCore`: normalized pinhole intrinsics for the original image, and a world-to-camera
/// rotation (row-major 3×3) and translation.
package struct StudioGeometryCamera: Codable, Equatable, Identifiable {
    package var id = UUID()
    package var imageWidth: Int
    package var imageHeight: Int
    /// Focal lengths as a fraction of the image width and height.
    package var normalizedFX: Double
    package var normalizedFY: Double
    /// The principal point as a fraction of the image width and height; 0.5 is the centre.
    package var normalizedCX: Double
    package var normalizedCY: Double
    package var rotation: [Double]
    package var translation: [Double]

    package init(
        imageWidth: Int,
        imageHeight: Int,
        normalizedFX: Double = 1,
        normalizedFY: Double = 1,
        normalizedCX: Double = 0.5,
        normalizedCY: Double = 0.5,
        rotation: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: [Double] = [0, 0, 0]
    ) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.normalizedFX = normalizedFX
        self.normalizedFY = normalizedFY
        self.normalizedCX = normalizedCX
        self.normalizedCY = normalizedCY
        self.rotation = rotation
        self.translation = translation
    }

    /// A camera at the origin looking down its own axis, for a view of `width` × `height`.
    package static func identity(width: Int = 1_920, height: Int = 1_080) -> StudioGeometryCamera {
        StudioGeometryCamera(imageWidth: width, imageHeight: height)
    }

    /// The CLI's checks (`DepthAnything3CameraValidation.issue`), in the page's words; empty when
    /// the camera is valid.
    package var problems: [String] {
        var problems: [String] = []
        if imageWidth <= 0 || imageHeight <= 0 { problems.append("needs a positive image size.") }
        if normalizedFX <= 0 || normalizedFY <= 0 { problems.append("needs positive focal lengths.") }
        let numbers = [normalizedFX, normalizedFY, normalizedCX, normalizedCY] + rotation + translation
        if numbers.contains(where: { !$0.isFinite }) { problems.append("has a value that is not a number.") }
        if rotation.count != 9 || translation.count != 3 {
            problems.append("needs a 3 × 3 rotation and a 3-value translation.")
        } else if let issue = Self.rotationIssue(rotation) {
            problems.append(issue)
        }
        return problems
    }

    /// Rows and columns unit length and orthogonal, determinant +1, to the CLI's 1e-3 tolerance.
    private static func rotationIssue(_ rotation: [Double]) -> String? {
        let tolerance = 1e-3
        let rows = [Array(rotation[0..<3]), Array(rotation[3..<6]), Array(rotation[6..<9])]
        let columns = (0..<3).map { column in [rotation[column], rotation[column + 3], rotation[column + 6]] }
        func dot(_ lhs: [Double], _ rhs: [Double]) -> Double { zip(lhs, rhs).reduce(0) { $0 + $1.0 * $1.1 } }
        for axes in [rows, columns] {
            for axis in 0..<3 where abs(dot(axes[axis], axes[axis]) - 1) > tolerance {
                return "has a rotation that is not a pure rotation (a row or column is not unit length)."
            }
            for first in 0..<3 {
                for second in (first + 1)..<3 where abs(dot(axes[first], axes[second])) > tolerance {
                    return "has a rotation that is not a pure rotation (rows or columns are not perpendicular)."
                }
            }
        }
        let determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
            - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
            + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
        if !determinant.isFinite || abs(determinant - 1) > tolerance {
            return "has a rotation that mirrors (its determinant is not +1)."
        }
        return nil
    }
}

/// The document `--cameras` reads for multi-view geometry (`DepthAnything3CameraDocument`).
package struct StudioGeometryCameraDocument: Codable, Equatable {
    package var cameras: [StudioGeometryCamera]

    package init(cameras: [StudioGeometryCamera] = []) {
        self.cameras = cameras
    }

    /// The CLI's checks against the run's views: one camera per view, each valid.
    package func problems(viewCount: Int) -> [String] {
        var problems: [String] = []
        if cameras.count != viewCount {
            problems.append(StudioCameraDocuments.countProblem(cameras: cameras.count, views: viewCount))
        }
        for (index, camera) in cameras.enumerated() {
            problems += camera.problems.map { "Camera \(index + 1) \($0)" }
        }
        return problems
    }

    package func json() throws -> Data {
        let document = Wire(schemaVersion: 1, cameras: cameras.map {
            Wire.Camera(
                intrinsics: .init(
                    imageWidth: $0.imageWidth,
                    imageHeight: $0.imageHeight,
                    normalizedFX: $0.normalizedFX,
                    normalizedFY: $0.normalizedFY,
                    normalizedCX: $0.normalizedCX,
                    normalizedCY: $0.normalizedCY
                ),
                extrinsics: .init(rotation: $0.rotation, translation: $0.translation)
            )
        })
        return try StudioCameraDocuments.encoder.encode(document)
    }

    package static func importing(_ data: Data) throws -> StudioGeometryCameraDocument {
        let document = try JSONDecoder().decode(Wire.self, from: data)
        guard document.schemaVersion == 1 else { throw StudioCameraDocumentImportError.unsupportedSchema(document.schemaVersion) }
        return StudioGeometryCameraDocument(cameras: document.cameras.map {
            StudioGeometryCamera(
                imageWidth: $0.intrinsics.imageWidth,
                imageHeight: $0.intrinsics.imageHeight,
                normalizedFX: $0.intrinsics.normalizedFX,
                normalizedFY: $0.intrinsics.normalizedFY,
                normalizedCX: $0.intrinsics.normalizedCX,
                normalizedCY: $0.intrinsics.normalizedCY,
                rotation: $0.extrinsics.rotation,
                translation: $0.extrinsics.translation
            )
        })
    }

    private struct Wire: Codable {
        struct Intrinsics: Codable {
            let imageWidth: Int
            let imageHeight: Int
            let normalizedFX: Double
            let normalizedFY: Double
            let normalizedCX: Double
            let normalizedCY: Double
        }

        struct Extrinsics: Codable {
            let rotation: [Double]
            let translation: [Double]
        }

        struct Camera: Codable {
            let intrinsics: Intrinsics
            let extrinsics: Extrinsics
        }

        let schemaVersion: Int
        let cameras: [Camera]
    }
}

// MARK: - InstantMesh

/// A calibrated camera for `image reconstruct-3d-multiview --cameras`: the sixteen conditioning
/// values InstantMesh takes per view, a row-major 3 × 4 camera-to-world matrix followed by
/// `fx, fy, cx, cy`. Without a document the CLI applies the released camera rig.
package struct StudioInstantMeshCamera: Codable, Equatable, Identifiable {
    package var id = UUID()
    package var values: [Double]

    package init(values: [Double]) {
        self.values = values
    }

    /// The guide's example row: identity pose four units back, the released focal length. A new
    /// camera each time, so a document built from several keeps distinct rows.
    package static var example: StudioInstantMeshCamera {
        StudioInstantMeshCamera(values: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 4, 1.866, 1.866, 0.5, 0.5])
    }

    /// The camera-to-world matrix, row by row.
    package var poseRows: [[Double]] {
        guard values.count == 16 else { return [] }
        return (0..<3).map { Array(values[($0 * 4)..<($0 * 4 + 4)]) }
    }

    /// The CLI's check: exactly sixteen finite values.
    package var problems: [String] {
        guard values.count == 16 else { return ["needs 16 values: a 3 × 4 pose and fx, fy, cx, cy."] }
        return values.allSatisfy(\.isFinite) ? [] : ["has a value that is not a number."]
    }
}

/// The document `--cameras` reads for InstantMesh (`InstantMeshCameraDocument`).
package struct StudioInstantMeshCameraDocument: Codable, Equatable {
    package var cameras: [StudioInstantMeshCamera]

    package init(cameras: [StudioInstantMeshCamera] = []) {
        self.cameras = cameras
    }

    package func problems(viewCount: Int) -> [String] {
        var problems: [String] = []
        if cameras.count != viewCount {
            problems.append(StudioCameraDocuments.countProblem(cameras: cameras.count, views: viewCount))
        }
        for (index, camera) in cameras.enumerated() {
            problems += camera.problems.map { "Camera \(index + 1) \($0)" }
        }
        return problems
    }

    package func json() throws -> Data {
        try StudioCameraDocuments.encoder.encode(Wire(schemaVersion: 1, cameras: cameras.map(\.values)))
    }

    package static func importing(_ data: Data) throws -> StudioInstantMeshCameraDocument {
        let document = try JSONDecoder().decode(Wire.self, from: data)
        guard document.schemaVersion == 1 else { throw StudioCameraDocumentImportError.unsupportedSchema(document.schemaVersion) }
        return StudioInstantMeshCameraDocument(cameras: document.cameras.map { StudioInstantMeshCamera(values: $0) })
    }

    private struct Wire: Codable {
        let schemaVersion: Int
        let cameras: [[Double]]
    }
}

// MARK: - Shared

package enum StudioCameraDocuments {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    static func countProblem(cameras: Int, views: Int) -> String {
        "Add one camera per view: \(views) \(views == 1 ? "view" : "views"), \(cameras) \(cameras == 1 ? "camera" : "cameras")."
    }

    /// Where a run's camera file goes: beside the run's output folder, as `<folder name>.cameras.json`,
    /// so the output folder itself stays empty for the command to fill.
    package static func url(besideOutputDirectory outputDirectory: String) -> URL {
        let output = URL(fileURLWithPath: NSString(string: outputDirectory).expandingTildeInPath, isDirectory: true).standardizedFileURL
        return output.deletingLastPathComponent().appendingPathComponent("\(output.lastPathComponent).cameras.json")
    }

    /// Where a page keeps the camera file it is editing, so the Command view's Run has a real file to
    /// pass as `--cameras`; one per page.
    package static func draftURL(page: String, fileManager: FileManager = .default) -> URL {
        StudioOutputLocation.appOutputsRoot(fileManager: fileManager)
            .deletingLastPathComponent()
            .appendingPathComponent(page, isDirectory: true)
            .appendingPathComponent("cameras.json")
    }
}

package enum StudioCameraDocumentImportError: LocalizedError, Equatable {
    case unsupportedSchema(Int)

    package var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let version):
            return "This camera file uses schemaVersion \(version); Studio and the CLI read version 1."
        }
    }
}
