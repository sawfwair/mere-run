import Foundation

/// Deterministic CPU counterpart to the reference pipeline's final small-hole
/// repair. It caps only closed manifold boundary loops below the same
/// normalized-object-space perimeter threshold and never introduces vertices.
enum Trellis2MeshHoleFiller {
    private struct EdgeKey: Hashable {
        let lower: UInt32
        let upper: UInt32

        init(_ start: UInt32, _ end: UInt32) {
            self.lower = min(start, end)
            self.upper = max(start, end)
        }
    }

    private struct EdgeRecord {
        var count: Int
        let start: UInt32
        let end: UInt32
    }

    private struct DirectedEdge {
        let start: UInt32
        let end: UInt32
    }

    private struct Point2D {
        let x: Float
        let y: Float
    }

    static func fillSmallHoles(
        in mesh: MeshAsset,
        maximumPerimeter: Float = 0.03
    ) throws -> MeshAsset {
        precondition(maximumPerimeter > 0)
        let boundaries = boundaryEdges(indices: mesh.indices)
        guard !boundaries.isEmpty else { return mesh }

        var additions = [UInt32]()
        for loop in closedLoops(edges: boundaries) {
            guard perimeter(of: loop, vertices: mesh.vertices) <= maximumPerimeter,
                  let triangles = triangulate(loop: Array(loop.reversed()), vertices: mesh.vertices) else {
                continue
            }
            additions.append(contentsOf: triangles)
        }
        guard !additions.isEmpty else { return mesh }

        let repaired = try MeshAsset(
            vertices: mesh.vertices,
            indices: mesh.indices + additions,
            normals: nil,
            colorsRGBA8: mesh.colorsRGBA8,
            textureCoordinates: mesh.textureCoordinates,
            coordinateSystem: mesh.coordinateSystem,
            units: mesh.units,
            inferredUnseenGeometry: mesh.inferredUnseenGeometry
        )
        return mesh.normals == nil ? repaired : try repaired.withGeneratedNormals()
    }

    private static func boundaryEdges(indices: [UInt32]) -> [DirectedEdge] {
        var records = [EdgeKey: EdgeRecord]()
        records.reserveCapacity(indices.count)
        for triangle in stride(from: 0, to: indices.count, by: 3) {
            let a = indices[triangle]
            let b = indices[triangle + 1]
            let c = indices[triangle + 2]
            for (start, end) in [(a, b), (b, c), (c, a)] {
                let key = EdgeKey(start, end)
                if var record = records[key] {
                    record.count += 1
                    records[key] = record
                } else {
                    records[key] = EdgeRecord(count: 1, start: start, end: end)
                }
            }
        }
        return records.values
            .filter { $0.count == 1 }
            .map { DirectedEdge(start: $0.start, end: $0.end) }
            .sorted { ($0.start, $0.end) < ($1.start, $1.end) }
    }

    private static func closedLoops(edges: [DirectedEdge]) -> [[UInt32]] {
        var outgoing = [UInt32: [Int]]()
        for (index, edge) in edges.enumerated() {
            outgoing[edge.start, default: []].append(index)
        }
        for vertex in outgoing.keys {
            outgoing[vertex]?.sort { edges[$0].end < edges[$1].end }
        }

        var used = [Bool](repeating: false, count: edges.count)
        var loops = [[UInt32]]()
        for firstIndex in edges.indices where !used[firstIndex] {
            let first = edges[firstIndex]
            var loop = [first.start]
            var seen = Set(loop)
            var current = first.end
            used[firstIndex] = true

            while current != first.start {
                guard seen.insert(current).inserted,
                      let nextIndex = outgoing[current]?.first(where: { !used[$0] }) else {
                    loop.removeAll()
                    break
                }
                loop.append(current)
                used[nextIndex] = true
                current = edges[nextIndex].end
            }
            if loop.count >= 3, current == first.start {
                loops.append(loop)
            }
        }
        return loops
    }

    private static func perimeter(of loop: [UInt32], vertices: [Float]) -> Float {
        var result: Float = 0
        for index in loop.indices {
            let start = Int(loop[index]) * 3
            let end = Int(loop[(index + 1) % loop.count]) * 3
            let deltaX = vertices[end] - vertices[start]
            let deltaY = vertices[end + 1] - vertices[start + 1]
            let deltaZ = vertices[end + 2] - vertices[start + 2]
            result += sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ)
        }
        return result
    }

    private static func triangulate(loop: [UInt32], vertices: [Float]) -> [UInt32]? {
        let points = projectedPoints(loop: loop, vertices: vertices)
        let signedArea = area(points)
        guard abs(signedArea) > Float.ulpOfOne else { return nil }
        let orientation: Float = signedArea > 0 ? 1 : -1
        var remaining = Array(loop.indices)
        var triangles = [UInt32]()
        triangles.reserveCapacity((loop.count - 2) * 3)

        while remaining.count > 3 {
            var clipped = false
            for position in remaining.indices {
                let previous = remaining[(position + remaining.count - 1) % remaining.count]
                let current = remaining[position]
                let next = remaining[(position + 1) % remaining.count]
                guard orientation * cross(points[previous], points[current], points[next]) > 1e-10 else {
                    continue
                }
                let containsVertex = remaining.contains { candidate in
                    candidate != previous && candidate != current && candidate != next
                        && pointInTriangle(
                            points[candidate],
                            a: points[previous],
                            b: points[current],
                            c: points[next],
                            orientation: orientation
                        )
                }
                guard !containsVertex else { continue }
                triangles.append(loop[previous])
                triangles.append(loop[current])
                triangles.append(loop[next])
                remaining.remove(at: position)
                clipped = true
                break
            }
            guard clipped else { return nil }
        }
        triangles.append(loop[remaining[0]])
        triangles.append(loop[remaining[1]])
        triangles.append(loop[remaining[2]])
        return triangles
    }

    private static func projectedPoints(loop: [UInt32], vertices: [Float]) -> [Point2D] {
        var normal = (x: Float(0), y: Float(0), z: Float(0))
        for index in loop.indices {
            let current = Int(loop[index]) * 3
            let next = Int(loop[(index + 1) % loop.count]) * 3
            normal.x += (vertices[current + 1] - vertices[next + 1])
                * (vertices[current + 2] + vertices[next + 2])
            normal.y += (vertices[current + 2] - vertices[next + 2])
                * (vertices[current] + vertices[next])
            normal.z += (vertices[current] - vertices[next])
                * (vertices[current + 1] + vertices[next + 1])
        }
        let axis = [abs(normal.x), abs(normal.y), abs(normal.z)]
            .enumerated()
            .max { $0.element < $1.element }?.offset ?? 2
        return loop.map { vertex in
            let base = Int(vertex) * 3
            switch axis {
            case 0: return Point2D(x: vertices[base + 1], y: vertices[base + 2])
            case 1: return Point2D(x: vertices[base], y: vertices[base + 2])
            default: return Point2D(x: vertices[base], y: vertices[base + 1])
            }
        }
    }

    private static func area(_ points: [Point2D]) -> Float {
        var result: Float = 0
        for index in points.indices {
            let next = points[(index + 1) % points.count]
            result += points[index].x * next.y - next.x * points[index].y
        }
        return result * 0.5
    }

    private static func cross(_ a: Point2D, _ b: Point2D, _ c: Point2D) -> Float {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    private static func pointInTriangle(
        _ point: Point2D,
        a: Point2D,
        b: Point2D,
        c: Point2D,
        orientation: Float
    ) -> Bool {
        orientation * cross(a, b, point) >= -1e-10
            && orientation * cross(b, c, point) >= -1e-10
            && orientation * cross(c, a, point) >= -1e-10
    }
}
