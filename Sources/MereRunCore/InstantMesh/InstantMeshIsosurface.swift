import Foundation
@preconcurrency import MLX

public struct InstantMeshMeshExtractionConfiguration: Equatable, Sendable {
    /// Number of cells per axis. The pinned checkpoint was trained at 128.
    public let gridResolution: Int
    public let includeVertexColors: Bool
    public let memory: InstantMeshMemoryConfiguration

    public init(
        gridResolution: Int = InstantMeshConfiguration.production.gridResolution,
        includeVertexColors: Bool = true,
        memory: InstantMeshMemoryConfiguration = .appleSilicon
    ) {
        self.gridResolution = gridResolution
        self.includeVertexColors = includeVertexColors
        self.memory = memory
    }
}

public enum InstantMeshIsosurfaceError: Error, Equatable, LocalizedError, Sendable {
    case invalidGridResolution(Int)
    case emptySurface(gridResolution: Int)
    case meshTooLarge

    public var errorDescription: String? {
        switch self {
        case .invalidGridResolution(let resolution):
            "InstantMesh extraction grid resolution must be between 2 and 256; received \(resolution)."
        case .emptySurface(let resolution):
            "InstantMesh SDF produced no isosurface on the \(resolution)-cell grid."
        case .meshTooLarge:
            "InstantMesh mesh exceeds the UInt32 indexed-mesh limit."
        }
    }
}

public struct InstantMeshMeshExtractionResult: Sendable {
    public let mesh: MeshAsset
    public let upstreamEmptyFieldRepairApplied: Bool

    public init(mesh: MeshAsset, upstreamEmptyFieldRepairApplied: Bool) {
        self.mesh = mesh
        self.upstreamEmptyFieldRepairApplied = upstreamEmptyFieldRepairApplied
    }
}

public enum InstantMeshIsosurfaceExtractor {
    /// Extracts the checkpoint SDF with a native marching-tetrahedra path.
    ///
    /// The learned SDF and deformation are exact. Tessellation intentionally
    /// does not port NVIDIA's per-file proprietary FlexiCubes implementation,
    /// so topology is not claimed to match the upstream extractor.
    public static func extractMesh(
        model: InstantMeshModel,
        sceneCode: InstantMeshSceneCode,
        sceneIndex: Int = 0,
        configuration: InstantMeshMeshExtractionConfiguration = InstantMeshMeshExtractionConfiguration()
    ) throws -> MeshAsset {
        try extractMeshWithMetadata(
            model: model,
            sceneCode: sceneCode,
            sceneIndex: sceneIndex,
            configuration: configuration
        ).mesh
    }

    public static func extractMeshWithMetadata(
        model: InstantMeshModel,
        sceneCode: InstantMeshSceneCode,
        sceneIndex: Int = 0,
        configuration: InstantMeshMeshExtractionConfiguration = InstantMeshMeshExtractionConfiguration()
    ) throws -> InstantMeshMeshExtractionResult {
        let cells = configuration.gridResolution
        guard (2...256).contains(cells) else {
            throw InstantMeshIsosurfaceError.invalidGridResolution(cells)
        }
        let pointsPerAxis = cells + 1
        let pointCount = pointsPerAxis * pointsPerAxis * pointsPerAxis
        let halfScale = model.configuration.gridScale / 2
        let denominator = Float(cells)
        var signedDistance = [Float](repeating: 0, count: pointCount)
        var deformedPositions = [Float](repeating: 0, count: pointCount * 3)

        for start in stride(from: 0, to: pointCount, by: configuration.memory.isosurfaceQueryChunkSize) {
            let end = min(start + configuration.memory.isosurfaceQueryChunkSize, pointCount)
            var positions: [Float] = []
            positions.reserveCapacity((end - start) * 3)
            for flatIndex in start..<end {
                let x = flatIndex / (pointsPerAxis * pointsPerAxis)
                let remainder = flatIndex % (pointsPerAxis * pointsPerAxis)
                let y = remainder / pointsPerAxis
                let z = remainder % pointsPerAxis
                positions.append((Float(x) / denominator * 2 - 1) * halfScale)
                positions.append((Float(y) / denominator * 2 - 1) * halfScale)
                positions.append((Float(z) / denominator * 2 - 1) * halfScale)
            }
            let positionArray = MLXArray(positions).reshaped(end - start, 3)
            let query = InstantMeshRenderer.query(
                model: model,
                sceneCode: sceneCode,
                sceneIndex: sceneIndex,
                positions: positionArray,
                chunkSize: configuration.memory.fieldQueryChunkSize
            )
            MLX.eval(query.signedDistance, query.deformation)
            let distances = query.signedDistance.reshaped(-1).asType(.float32).asArray(Float.self)
            let displacement = query.deformation.asType(.float32).asArray(Float.self)
            signedDistance.replaceSubrange(start..<end, with: distances)
            for local in 0..<(end - start) {
                let source = local * 3
                let destination = (start + local) * 3
                deformedPositions[destination] = positions[source] + displacement[source]
                deformedPositions[destination + 1] = positions[source + 1] + displacement[source + 1]
                deformedPositions[destination + 2] = positions[source + 2] + displacement[source + 2]
            }
        }
        // The pinned reconstruction graph intentionally repairs a completely
        // one-signed interior field before topology extraction. Reproduce the
        // upstream LRM behavior exactly: replace one center sample with the
        // positive sentinel and the outer two-sample shell with the negative
        // sentinel. For tiny diagnostic grids those sets can overlap, so the
        // upstream additive update semantics matter as well.
        let upstreamEmptyFieldRepairApplied = repairEmptyInteriorField(
            &signedDistance,
            pointsPerAxis: pointsPerAxis
        )
        var mesh = try polygonize(
            signedDistance: signedDistance,
            positions: deformedPositions,
            pointsPerAxis: pointsPerAxis
        )
        if configuration.includeVertexColors {
            let query = InstantMeshRenderer.query(
                model: model,
                sceneCode: sceneCode,
                sceneIndex: sceneIndex,
                positions: MLXArray(mesh.vertices).reshaped(mesh.vertexCount, 3),
                chunkSize: configuration.memory.fieldQueryChunkSize
            )
            MLX.eval(query.color)
            let rgb = query.color.asType(.float32).asArray(Float.self)
            var rgba = [UInt8](repeating: 255, count: mesh.vertexCount * 4)
            for vertex in 0..<mesh.vertexCount {
                rgba[vertex * 4] = byte(rgb[vertex * 3])
                rgba[vertex * 4 + 1] = byte(rgb[vertex * 3 + 1])
                rgba[vertex * 4 + 2] = byte(rgb[vertex * 3 + 2])
            }
            mesh = try MeshAsset(
                vertices: mesh.vertices,
                indices: mesh.indices,
                normals: mesh.normals,
                colorsRGBA8: rgba,
                coordinateSystem: mesh.coordinateSystem,
                units: mesh.units,
                inferredUnseenGeometry: mesh.inferredUnseenGeometry
            )
        }
        return InstantMeshMeshExtractionResult(
            mesh: mesh,
            upstreamEmptyFieldRepairApplied: upstreamEmptyFieldRepairApplied
        )
    }

    private static func flat(_ x: Int, _ y: Int, _ z: Int, pointsPerAxis: Int) -> Int {
        x * pointsPerAxis * pointsPerAxis + y * pointsPerAxis + z
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8(clamping: Int((min(1, max(0, value)) * 255).rounded()))
    }

    @discardableResult
    static func repairEmptyInteriorField(
        _ signedDistance: inout [Float],
        pointsPerAxis: Int
    ) -> Bool {
        precondition(pointsPerAxis >= 3)
        precondition(signedDistance.count == pointsPerAxis * pointsPerAxis * pointsPerAxis)
        var hasPositiveInterior = false
        var hasNegativeInterior = false
        for x in 1..<(pointsPerAxis - 1) {
            for y in 1..<(pointsPerAxis - 1) {
                for z in 1..<(pointsPerAxis - 1) {
                    let value = signedDistance[flat(x, y, z, pointsPerAxis: pointsPerAxis)]
                    hasPositiveInterior = hasPositiveInterior || value > 0
                    hasNegativeInterior = hasNegativeInterior || value < 0
                }
            }
        }
        guard !hasPositiveInterior || !hasNegativeInterior else { return false }

        let minimum = signedDistance.min() ?? 0
        let maximum = signedDistance.max() ?? 0
        let positiveUpdate = 1 - minimum
        let negativeUpdate = -1 - maximum
        let cells = pointsPerAxis - 1
        let centerCoordinate = cells / 2 + 1
        let centerIndex = flat(
            centerCoordinate,
            centerCoordinate,
            centerCoordinate,
            pointsPerAxis: pointsPerAxis
        )

        var replacements: [Int: Float] = [centerIndex: positiveUpdate]
        for x in 0..<pointsPerAxis {
            for y in 0..<pointsPerAxis {
                for z in 0..<pointsPerAxis where
                    x < 2 || x >= pointsPerAxis - 2
                        || y < 2 || y >= pointsPerAxis - 2
                        || z < 2 || z >= pointsPerAxis - 2 {
                    let index = flat(x, y, z, pointsPerAxis: pointsPerAxis)
                    replacements[index, default: 0] += negativeUpdate
                }
            }
        }
        for (index, value) in replacements where value != 0 {
            signedDistance[index] = value
        }
        return true
    }

    private struct GridPoint: Hashable {
        let x: Int
        let y: Int
        let z: Int
    }

    private struct EdgeKey: Hashable {
        let lower: Int
        let upper: Int

        init(_ a: Int, _ b: Int) {
            lower = min(a, b)
            upper = max(a, b)
        }
    }

    private struct Vertex3 {
        let x: Float
        let y: Float
        let z: Float

        static func + (lhs: Vertex3, rhs: Vertex3) -> Vertex3 {
            Vertex3(x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z)
        }

        static func - (lhs: Vertex3, rhs: Vertex3) -> Vertex3 {
            Vertex3(x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z)
        }

        static func * (lhs: Vertex3, rhs: Float) -> Vertex3 {
            Vertex3(x: lhs.x * rhs, y: lhs.y * rhs, z: lhs.z * rhs)
        }
    }

    static func polygonize(
        signedDistance: [Float],
        positions: [Float],
        pointsPerAxis: Int
    ) throws -> MeshAsset {
        precondition(signedDistance.count == pointsPerAxis * pointsPerAxis * pointsPerAxis)
        precondition(positions.count == signedDistance.count * 3)
        let cornerOffsets = [
            GridPoint(x: 0, y: 0, z: 0), GridPoint(x: 1, y: 0, z: 0),
            GridPoint(x: 1, y: 1, z: 0), GridPoint(x: 0, y: 1, z: 0),
            GridPoint(x: 0, y: 0, z: 1), GridPoint(x: 1, y: 0, z: 1),
            GridPoint(x: 1, y: 1, z: 1), GridPoint(x: 0, y: 1, z: 1),
        ]
        let tetrahedra = [
            [0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
            [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6],
        ]
        var vertices: [Float] = []
        var indices: [UInt32] = []
        var vertexByEdge: [EdgeKey: UInt32] = [:]

        func pointIndex(_ point: GridPoint) -> Int {
            flat(point.x, point.y, point.z, pointsPerAxis: pointsPerAxis)
        }

        func pointPosition(_ point: GridPoint) -> Vertex3 {
            let offset = pointIndex(point) * 3
            return Vertex3(x: positions[offset], y: positions[offset + 1], z: positions[offset + 2])
        }

        func vertex(
            pointA: GridPoint,
            pointB: GridPoint,
            valueA: Float,
            valueB: Float
        ) throws -> UInt32 {
            let indexA = pointIndex(pointA)
            let indexB = pointIndex(pointB)
            let key = EdgeKey(indexA, indexB)
            if let existing = vertexByEdge[key] { return existing }
            guard vertices.count / 3 < Int(UInt32.max) else {
                throw InstantMeshIsosurfaceError.meshTooLarge
            }
            let delta = valueB - valueA
            let fraction = abs(delta) < 1e-12 ? Float(0.5) : min(1, max(0, -valueA / delta))
            let a = pointPosition(pointA)
            let b = pointPosition(pointB)
            let position = a + (b - a) * fraction
            let index = UInt32(vertices.count / 3)
            vertices.append(contentsOf: [position.x, position.y, position.z])
            vertexByEdge[key] = index
            return index
        }

        func outputPosition(_ index: UInt32) -> Vertex3 {
            let offset = Int(index) * 3
            return Vertex3(x: vertices[offset], y: vertices[offset + 1], z: vertices[offset + 2])
        }

        func dot(_ a: Vertex3, _ b: Vertex3) -> Float { a.x * b.x + a.y * b.y + a.z * b.z }
        func cross(_ a: Vertex3, _ b: Vertex3) -> Vertex3 {
            Vertex3(
                x: a.y * b.z - a.z * b.y,
                y: a.z * b.x - a.x * b.z,
                z: a.x * b.y - a.y * b.x
            )
        }

        func appendTriangle(_ a: UInt32, _ b: UInt32, _ c: UInt32, insideCenter: Vertex3) {
            let pa = outputPosition(a)
            let pb = outputPosition(b)
            let pc = outputPosition(c)
            let normal = cross(pb - pa, pc - pa)
            let centroid = (pa + pb + pc) * (1 / 3)
            if dot(normal, centroid - insideCenter) >= 0 {
                indices.append(contentsOf: [a, b, c])
            } else {
                indices.append(contentsOf: [a, c, b])
            }
        }

        for x in 0..<(pointsPerAxis - 1) {
            for y in 0..<(pointsPerAxis - 1) {
                for z in 0..<(pointsPerAxis - 1) {
                    let points = cornerOffsets.map {
                        GridPoint(x: x + $0.x, y: y + $0.y, z: z + $0.z)
                    }
                    let values = points.map { signedDistance[pointIndex($0)] }
                    let insideCount = values.reduce(0) { $0 + ($1 >= 0 ? 1 : 0) }
                    if insideCount == 0 || insideCount == 8 { continue }
                    for tetrahedron in tetrahedra {
                        let inside = tetrahedron.filter { values[$0] >= 0 }
                        let outside = tetrahedron.filter { values[$0] < 0 }
                        if inside.isEmpty || outside.isEmpty { continue }
                        let insideCenter = inside.map { pointPosition(points[$0]) }
                            .reduce(Vertex3(x: 0, y: 0, z: 0), +) * (1 / Float(inside.count))
                        if inside.count == 1 || inside.count == 3 {
                            let loneInside = inside.count == 1
                            let lone = loneInside ? inside[0] : outside[0]
                            let others = loneInside ? outside : inside
                            let intersections = try others.map {
                                try vertex(
                                    pointA: points[lone], pointB: points[$0],
                                    valueA: values[lone], valueB: values[$0]
                                )
                            }
                            appendTriangle(intersections[0], intersections[1], intersections[2], insideCenter: insideCenter)
                        } else {
                            let a = inside[0]
                            let b = inside[1]
                            let c = outside[0]
                            let d = outside[1]
                            let ac = try vertex(pointA: points[a], pointB: points[c], valueA: values[a], valueB: values[c])
                            let ad = try vertex(pointA: points[a], pointB: points[d], valueA: values[a], valueB: values[d])
                            let bc = try vertex(pointA: points[b], pointB: points[c], valueA: values[b], valueB: values[c])
                            let bd = try vertex(pointA: points[b], pointB: points[d], valueA: values[b], valueB: values[d])
                            appendTriangle(ac, bc, ad, insideCenter: insideCenter)
                            appendTriangle(ad, bc, bd, insideCenter: insideCenter)
                        }
                    }
                }
            }
        }
        guard !vertices.isEmpty, !indices.isEmpty else {
            throw InstantMeshIsosurfaceError.emptySurface(gridResolution: pointsPerAxis - 1)
        }
        return try MeshAsset(
            vertices: vertices,
            indices: indices,
            normals: MeshNormals.generate(vertices: vertices, indices: indices),
            coordinateSystem: .modelXRightYUpZForward,
            units: .normalizedObjectSpace,
            inferredUnseenGeometry: true
        )
    }
}
