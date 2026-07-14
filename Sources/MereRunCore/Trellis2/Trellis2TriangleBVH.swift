import Foundation

/// Median-split AABB tree over an indexed triangle mesh for closest-point
/// queries. Mirrors the role of CuMesh's cuBVH in the narrow-band remesher:
/// unsigned distance for the band field and closest points for projection.
struct Trellis2TriangleBVH {
    private struct Node {
        var minX: Float, minY: Float, minZ: Float
        var maxX: Float, maxY: Float, maxZ: Float
        /// Leaf when `count > 0`: triangles [start, start + count) in `order`.
        var start: Int32
        var count: Int32
        /// Internal node children; right = left + 1 is not guaranteed.
        var left: Int32
        var right: Int32
    }

    private let nodes: [Node]
    /// Triangle indices permuted during the build.
    private let order: [Int32]
    private let vertices: [Float]
    private let indices: [UInt32]

    init(vertices: [Float], indices: [UInt32]) {
        self.vertices = vertices
        self.indices = indices
        let triangleCount = indices.count / 3

        var centroids = [Float](repeating: 0, count: triangleCount * 3)
        for triangle in 0..<triangleCount {
            for corner in 0..<3 {
                let base = Int(indices[triangle * 3 + corner]) * 3
                centroids[triangle * 3] += vertices[base] / 3
                centroids[triangle * 3 + 1] += vertices[base + 1] / 3
                centroids[triangle * 3 + 2] += vertices[base + 2] / 3
            }
        }

        var order = (0..<triangleCount).map(Int32.init)
        var nodes = [Node]()
        nodes.reserveCapacity(triangleCount / 4)

        func bounds(_ start: Int, _ count: Int) -> (Float, Float, Float, Float, Float, Float) {
            var minX = Float.infinity, minY = Float.infinity, minZ = Float.infinity
            var maxX = -Float.infinity, maxY = -Float.infinity, maxZ = -Float.infinity
            for slot in start..<(start + count) {
                let triangle = Int(order[slot])
                for corner in 0..<3 {
                    let base = Int(indices[triangle * 3 + corner]) * 3
                    minX = min(minX, vertices[base]); maxX = max(maxX, vertices[base])
                    minY = min(minY, vertices[base + 1]); maxY = max(maxY, vertices[base + 1])
                    minZ = min(minZ, vertices[base + 2]); maxZ = max(maxZ, vertices[base + 2])
                }
            }
            return (minX, minY, minZ, maxX, maxY, maxZ)
        }

        func build(_ start: Int, _ count: Int) -> Int32 {
            let (minX, minY, minZ, maxX, maxY, maxZ) = bounds(start, count)
            let nodeIndex = Int32(nodes.count)
            nodes.append(Node(
                minX: minX, minY: minY, minZ: minZ,
                maxX: maxX, maxY: maxY, maxZ: maxZ,
                start: Int32(start), count: Int32(count), left: -1, right: -1
            ))
            if count <= 8 { return nodeIndex }

            var centroidMin = [Float.infinity, .infinity, .infinity]
            var centroidMax = [-Float.infinity, -.infinity, -.infinity]
            for slot in start..<(start + count) {
                let triangle = Int(order[slot])
                for axis in 0..<3 {
                    centroidMin[axis] = min(centroidMin[axis], centroids[triangle * 3 + axis])
                    centroidMax[axis] = max(centroidMax[axis], centroids[triangle * 3 + axis])
                }
            }
            var axis = 0
            for candidate in 1..<3
                where centroidMax[candidate] - centroidMin[candidate] > centroidMax[axis] - centroidMin[axis] {
                axis = candidate
            }
            guard centroidMax[axis] - centroidMin[axis] > Float.ulpOfOne else { return nodeIndex }

            let middle = start + count / 2
            order[start..<(start + count)].sort {
                centroids[Int($0) * 3 + axis] < centroids[Int($1) * 3 + axis]
            }
            let left = build(start, middle - start)
            let right = build(middle, start + count - middle)
            nodes[Int(nodeIndex)].count = 0
            nodes[Int(nodeIndex)].left = left
            nodes[Int(nodeIndex)].right = right
            return nodeIndex
        }

        _ = build(0, triangleCount)
        self.order = order
        self.nodes = nodes
    }

    /// Closest surface point and squared distance from the query point.
    func closestPoint(x: Float, y: Float, z: Float) -> (distanceSquared: Float, x: Float, y: Float, z: Float) {
        var best = Float.infinity
        var bestPoint: (Float, Float, Float) = (x, y, z)
        var stack = [Int32]()
        stack.reserveCapacity(64)
        stack.append(0)

        while let nodeIndex = stack.popLast() {
            let node = nodes[Int(nodeIndex)]
            guard boxDistanceSquared(node, x, y, z) < best else { continue }
            if node.count > 0 {
                for slot in Int(node.start)..<Int(node.start + node.count) {
                    let triangle = Int(order[slot])
                    let candidate = closestPointOnTriangle(triangle, x, y, z)
                    let dx = candidate.0 - x, dy = candidate.1 - y, dz = candidate.2 - z
                    let distance = dx * dx + dy * dy + dz * dz
                    if distance < best {
                        best = distance
                        bestPoint = candidate
                    }
                }
            } else {
                // Visit the nearer child last so it is popped first.
                let leftDistance = boxDistanceSquared(nodes[Int(node.left)], x, y, z)
                let rightDistance = boxDistanceSquared(nodes[Int(node.right)], x, y, z)
                if leftDistance < rightDistance {
                    stack.append(node.right)
                    stack.append(node.left)
                } else {
                    stack.append(node.left)
                    stack.append(node.right)
                }
            }
        }
        return (best, bestPoint.0, bestPoint.1, bestPoint.2)
    }

    func unsignedDistance(x: Float, y: Float, z: Float) -> Float {
        closestPoint(x: x, y: y, z: z).distanceSquared.squareRoot()
    }

    private func boxDistanceSquared(_ node: Node, _ x: Float, _ y: Float, _ z: Float) -> Float {
        let dx = max(node.minX - x, 0, x - node.maxX)
        let dy = max(node.minY - y, 0, y - node.maxY)
        let dz = max(node.minZ - z, 0, z - node.maxZ)
        return dx * dx + dy * dy + dz * dz
    }

    /// Ericson's closest-point-on-triangle, returning the surface point.
    private func closestPointOnTriangle(_ triangle: Int, _ px: Float, _ py: Float, _ pz: Float) -> (Float, Float, Float) {
        let baseA = Int(indices[triangle * 3]) * 3
        let baseB = Int(indices[triangle * 3 + 1]) * 3
        let baseC = Int(indices[triangle * 3 + 2]) * 3
        let ax = vertices[baseA], ay = vertices[baseA + 1], az = vertices[baseA + 2]
        let bx = vertices[baseB], by = vertices[baseB + 1], bz = vertices[baseB + 2]
        let cx = vertices[baseC], cy = vertices[baseC + 1], cz = vertices[baseC + 2]

        let abx = bx - ax, aby = by - ay, abz = bz - az
        let acx = cx - ax, acy = cy - ay, acz = cz - az
        let apx = px - ax, apy = py - ay, apz = pz - az

        let d1 = abx * apx + aby * apy + abz * apz
        let d2 = acx * apx + acy * apy + acz * apz
        if d1 <= 0 && d2 <= 0 { return (ax, ay, az) }

        let bpx = px - bx, bpy = py - by, bpz = pz - bz
        let d3 = abx * bpx + aby * bpy + abz * bpz
        let d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0 && d4 <= d3 { return (bx, by, bz) }

        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 {
            let t = d1 / (d1 - d3)
            return (ax + t * abx, ay + t * aby, az + t * abz)
        }

        let cpx = px - cx, cpy = py - cy, cpz = pz - cz
        let d5 = abx * cpx + aby * cpy + abz * cpz
        let d6 = acx * cpx + acy * cpy + acz * cpz
        if d6 >= 0 && d5 <= d6 { return (cx, cy, cz) }

        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 {
            let t = d2 / (d2 - d6)
            return (ax + t * acx, ay + t * acy, az + t * acz)
        }

        let va = d3 * d6 - d5 * d4
        if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0 {
            let t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return (bx + t * (cx - bx), by + t * (cy - by), bz + t * (cz - bz))
        }

        let denominator = 1 / (va + vb + vc)
        let v = vb * denominator
        let w = vc * denominator
        return (ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w)
    }
}
