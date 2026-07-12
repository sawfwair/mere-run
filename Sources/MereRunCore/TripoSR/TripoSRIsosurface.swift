import Foundation
@preconcurrency import MLX

public struct TripoSRMeshExtractionConfiguration: Equatable, Sendable {
    public let resolution: Int
    public let densityThreshold: Float
    public let includeVertexColors: Bool
    public let memory: TripoSRMemoryConfiguration

    public init(
        resolution: Int = 256,
        densityThreshold: Float = TripoSRConfiguration.production.densityThreshold,
        includeVertexColors: Bool = true,
        memory: TripoSRMemoryConfiguration = .appleSilicon
    ) {
        self.resolution = resolution
        self.densityThreshold = densityThreshold
        self.includeVertexColors = includeVertexColors
        self.memory = memory
    }
}

public enum TripoSRIsosurfaceError: Error, Equatable, LocalizedError, Sendable {
    case invalidConfiguration(resolution: Int, threshold: Float)
    case emptySurface(resolution: Int, threshold: Float)
    case meshTooLarge

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let resolution, let threshold):
            "TripoSR extraction requires resolution 2...512 and a finite threshold; received "
                + "\(resolution) and \(threshold)."
        case .emptySurface(let resolution, let threshold):
            "TripoSR density produced no isosurface at resolution \(resolution) and threshold \(threshold)."
        case .meshTooLarge:
            "TripoSR mesh exceeds the UInt32 indexed-mesh limit."
        }
    }
}

public enum TripoSRIsosurfaceExtractor {
    /// Evaluates the neural field in bounded chunks and extracts an indexed
    /// triangle mesh in normalized object coordinates.
    public static func extractMesh(
        model: TripoSRModel,
        sceneCode: TripoSRSceneCode,
        sceneIndex: Int = 0,
        configuration: TripoSRMeshExtractionConfiguration = TripoSRMeshExtractionConfiguration()
    ) throws -> MeshAsset {
        let resolution = configuration.resolution
        guard (2...512).contains(resolution), configuration.densityThreshold.isFinite else {
            throw TripoSRIsosurfaceError.invalidConfiguration(
                resolution: resolution,
                threshold: configuration.densityThreshold
            )
        }
        let pointCount = resolution * resolution * resolution
        var density = [Float](repeating: 0, count: pointCount)
        let denominator = Float(resolution - 1)
        let radius = model.configuration.rendererRadius

        for start in stride(from: 0, to: pointCount, by: configuration.memory.isosurfaceChunkSize) {
            let end = min(start + configuration.memory.isosurfaceChunkSize, pointCount)
            var positions = [Float]()
            positions.reserveCapacity((end - start) * 3)
            for flatIndex in start..<end {
                let x = flatIndex / (resolution * resolution)
                let remainder = flatIndex % (resolution * resolution)
                let y = remainder / resolution
                let z = remainder % resolution
                positions.append((Float(x) / denominator * 2 - 1) * radius)
                positions.append((Float(y) / denominator * 2 - 1) * radius)
                positions.append((Float(z) / denominator * 2 - 1) * radius)
            }
            let query = TripoSRRenderer.query(
                model: model,
                sceneCode: sceneCode,
                sceneIndex: sceneIndex,
                positions: MLXArray(positions).reshaped(end - start, 3),
                chunkSize: configuration.memory.queryChunkSize
            ).activatedDensity
            MLX.eval(query)
            let values = query.reshaped(-1).asArray(Float.self)
            density.replaceSubrange(start..<end, with: values)
        }

        var mesh = try polygonize(
            density: density,
            resolution: resolution,
            radius: radius,
            threshold: configuration.densityThreshold,
            inferredUnseenGeometry: true
        )
        if configuration.includeVertexColors {
            let positionArray = MLXArray(mesh.vertices).reshaped(mesh.vertexCount, 3)
            let query = TripoSRRenderer.query(
                model: model,
                sceneCode: sceneCode,
                sceneIndex: sceneIndex,
                positions: positionArray,
                chunkSize: configuration.memory.queryChunkSize
            ).color
            MLX.eval(query)
            let rgb = query.asType(.float32).asArray(Float.self)
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
        return mesh
    }

    private static func byte(_ value: Float) -> UInt8 {
        UInt8(clamping: Int((min(1, max(0, value)) * 255).rounded()))
    }

    struct GridPoint: Hashable {
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

    struct Vertex3 {
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

    /// Deterministic marching-tetrahedra polygonizer. It avoids a native
    /// dependency and shares vertices by lattice-edge identity. The neural
    /// density and threshold match upstream; the tessellation may differ from
    /// torchmcubes while representing the same sampled isosurface.
    static func polygonize(
        density: [Float],
        resolution: Int,
        radius: Float,
        threshold: Float,
        inferredUnseenGeometry: Bool
    ) throws -> MeshAsset {
        precondition(density.count == resolution * resolution * resolution)
        let cornerOffsets = [
            GridPoint(x: 0, y: 0, z: 0),
            GridPoint(x: 1, y: 0, z: 0),
            GridPoint(x: 1, y: 1, z: 0),
            GridPoint(x: 0, y: 1, z: 0),
            GridPoint(x: 0, y: 0, z: 1),
            GridPoint(x: 1, y: 0, z: 1),
            GridPoint(x: 1, y: 1, z: 1),
            GridPoint(x: 0, y: 1, z: 1),
        ]
        let tetrahedra = [
            [0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
            [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6],
        ]
        var vertices: [Float] = []
        var indices: [UInt32] = []
        var vertexByEdge: [EdgeKey: UInt32] = [:]
        let denominator = Float(resolution - 1)

        func flat(_ point: GridPoint) -> Int {
            point.x * resolution * resolution + point.y * resolution + point.z
        }

        func objectPosition(_ point: GridPoint) -> Vertex3 {
            Vertex3(
                x: (Float(point.x) / denominator * 2 - 1) * radius,
                y: (Float(point.y) / denominator * 2 - 1) * radius,
                z: (Float(point.z) / denominator * 2 - 1) * radius
            )
        }

        func vertex(
            pointA: GridPoint,
            pointB: GridPoint,
            valueA: Float,
            valueB: Float
        ) throws -> UInt32 {
            let indexA = flat(pointA)
            let indexB = flat(pointB)
            let key = EdgeKey(indexA, indexB)
            if let existing = vertexByEdge[key] { return existing }
            guard vertices.count / 3 < Int(UInt32.max) else {
                throw TripoSRIsosurfaceError.meshTooLarge
            }
            let delta = valueB - valueA
            let fraction = abs(delta) < 1e-12 ? Float(0.5) : min(1, max(0, (threshold - valueA) / delta))
            let a = objectPosition(pointA)
            let b = objectPosition(pointB)
            let position = a + (b - a) * fraction
            let index = UInt32(vertices.count / 3)
            vertices.append(contentsOf: [position.x, position.y, position.z])
            vertexByEdge[key] = index
            return index
        }

        func position(_ index: UInt32) -> Vertex3 {
            let base = Int(index) * 3
            return Vertex3(x: vertices[base], y: vertices[base + 1], z: vertices[base + 2])
        }

        func dot(_ a: Vertex3, _ b: Vertex3) -> Float { a.x * b.x + a.y * b.y + a.z * b.z }
        func cross(_ a: Vertex3, _ b: Vertex3) -> Vertex3 {
            Vertex3(
                x: a.y * b.z - a.z * b.y,
                y: a.z * b.x - a.x * b.z,
                z: a.x * b.y - a.y * b.x
            )
        }

        func appendOrientedTriangle(_ a: UInt32, _ b: UInt32, _ c: UInt32, insideCenter: Vertex3) {
            let pa = position(a)
            let pb = position(b)
            let pc = position(c)
            let normal = cross(pb - pa, pc - pa)
            let centroid = (pa + pb + pc) * (1 / 3)
            if dot(normal, centroid - insideCenter) >= 0 {
                indices.append(contentsOf: [a, b, c])
            } else {
                indices.append(contentsOf: [a, c, b])
            }
        }

        for x in 0..<(resolution - 1) {
            for y in 0..<(resolution - 1) {
                for z in 0..<(resolution - 1) {
                    let points = cornerOffsets.map {
                        GridPoint(x: x + $0.x, y: y + $0.y, z: z + $0.z)
                    }
                    let values = points.map { density[flat($0)] }
                    let insideCount = values.reduce(0) { $0 + ($1 >= threshold ? 1 : 0) }
                    if insideCount == 0 || insideCount == 8 { continue }

                    for tetrahedron in tetrahedra {
                        let inside = tetrahedron.filter { values[$0] >= threshold }
                        let outside = tetrahedron.filter { values[$0] < threshold }
                        if inside.isEmpty || outside.isEmpty { continue }
                        let insideCenter = inside
                            .map { objectPosition(points[$0]) }
                            .reduce(Vertex3(x: 0, y: 0, z: 0), +)
                            * (1 / Float(inside.count))

                        if inside.count == 1 || inside.count == 3 {
                            let loneIsInside = inside.count == 1
                            let lone = loneIsInside ? inside[0] : outside[0]
                            let others = loneIsInside ? outside : inside
                            let intersections = try others.map {
                                try vertex(
                                    pointA: points[lone],
                                    pointB: points[$0],
                                    valueA: values[lone],
                                    valueB: values[$0]
                                )
                            }
                            appendOrientedTriangle(
                                intersections[0], intersections[1], intersections[2],
                                insideCenter: insideCenter
                            )
                        } else {
                            let a = inside[0]
                            let b = inside[1]
                            let c = outside[0]
                            let d = outside[1]
                            let ac = try vertex(pointA: points[a], pointB: points[c], valueA: values[a], valueB: values[c])
                            let ad = try vertex(pointA: points[a], pointB: points[d], valueA: values[a], valueB: values[d])
                            let bc = try vertex(pointA: points[b], pointB: points[c], valueA: values[b], valueB: values[c])
                            let bd = try vertex(pointA: points[b], pointB: points[d], valueA: values[b], valueB: values[d])
                            appendOrientedTriangle(ac, bc, ad, insideCenter: insideCenter)
                            appendOrientedTriangle(ad, bc, bd, insideCenter: insideCenter)
                        }
                    }
                }
            }
        }

        guard !vertices.isEmpty, !indices.isEmpty else {
            throw TripoSRIsosurfaceError.emptySurface(resolution: resolution, threshold: threshold)
        }
        return try MeshAsset(
            vertices: vertices,
            indices: indices,
            normals: MeshNormals.generate(vertices: vertices, indices: indices),
            coordinateSystem: .modelXRightYUpZForward,
            units: .normalizedObjectSpace,
            inferredUnseenGeometry: inferredUnseenGeometry
        )
    }
}
