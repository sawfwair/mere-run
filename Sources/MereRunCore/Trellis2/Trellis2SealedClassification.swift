import Foundation

/// Sealed inside/outside classification of the O-Voxel occupancy grid via
/// morphological closing: dilate the occupied crust by `radius`, flood-fill
/// the exterior from the grid boundary, then erode by the same radius.
/// Any cavity whose every mouth is narrower than roughly `2 * radius`
/// voxels is classified interior even when the crust around it is torn, so
/// the remesher can span a membrane across openings that no boundary-loop
/// capping can close. Thin sheets are unaffected: they are handled by the
/// narrow-band envelope, not the classification.
struct Trellis2SealedClassification {
    let resolution: Int
    /// Cells that are neither exterior nor within `radius` of it.
    private let interiorBits: [UInt64]
    /// Per-level mixed-occupancy pyramid; level 0 is the full grid, each
    /// higher level halves the resolution. A set bit means the block
    /// contains both interior and non-interior cells, so an octree cell
    /// overlapping it straddles the classification boundary.
    private let mixedPyramid: [[UInt64]]

    /// `occupiedCells` are flat (x, y, z) triples in this grid's own space.
    init(
        resolution: Int,
        occupiedCells: [Int32],
        radius: Int
    ) {
        self.resolution = resolution
        let volume = resolution * resolution * resolution
        let words = (volume + 63) / 64

        func index(_ x: Int, _ y: Int, _ z: Int) -> Int {
            (x * resolution + y) * resolution + z
        }
        func get(_ bits: [UInt64], _ cell: Int) -> Bool {
            bits[cell >> 6] & (1 << UInt64(cell & 63)) != 0
        }
        func set(_ bits: inout [UInt64], _ cell: Int) {
            bits[cell >> 6] |= 1 << UInt64(cell & 63)
        }

        // Multi-source BFS expansion by `rounds` through cells not blocked.
        func expand(
            from seeds: inout [Int32],
            marked: inout [UInt64],
            rounds: Int,
            blocked: [UInt64]?
        ) {
            var frontier = seeds
            for _ in 0..<rounds {
                var next = [Int32]()
                next.reserveCapacity(frontier.count)
                for cell in frontier {
                    let c = Int(cell)
                    let z = c % resolution
                    let y = (c / resolution) % resolution
                    let x = c / (resolution * resolution)
                    for (dx, dy, dz) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                        let nx = x + dx, ny = y + dy, nz = z + dz
                        guard nx >= 0, nx < resolution, ny >= 0, ny < resolution, nz >= 0, nz < resolution else { continue }
                        let n = index(nx, ny, nz)
                        if get(marked, n) { continue }
                        if let blocked, get(blocked, n) { continue }
                        set(&marked, n)
                        next.append(Int32(n))
                    }
                }
                frontier = next
                if frontier.isEmpty { break }
            }
            seeds = frontier
        }

        // 1. Occupancy.
        var occupancy = [UInt64](repeating: 0, count: words)
        for triple in stride(from: 0, to: occupiedCells.count, by: 3) {
            let x = Int(occupiedCells[triple])
            let y = Int(occupiedCells[triple + 1])
            let z = Int(occupiedCells[triple + 2])
            guard x >= 0, x < resolution, y >= 0, y < resolution, z >= 0, z < resolution else { continue }
            set(&occupancy, index(x, y, z))
        }

        // 2. Dilate by `radius`.
        var dilated = occupancy
        var seeds = [Int32]()
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution where get(occupancy, index(x, y, z)) {
                    seeds.append(Int32(index(x, y, z)))
                }
            }
        }
        expand(from: &seeds, marked: &dilated, rounds: radius, blocked: nil)

        // 3. Exterior flood from the grid boundary through non-dilated cells.
        var exterior = [UInt64](repeating: 0, count: words)
        var boundarySeeds = [Int32]()
        for a in 0..<resolution {
            for b in 0..<resolution {
                for cell in [
                    index(0, a, b), index(resolution - 1, a, b),
                    index(a, 0, b), index(a, resolution - 1, b),
                    index(a, b, 0), index(a, b, resolution - 1),
                ] where !get(dilated, cell) && !get(exterior, cell) {
                    set(&exterior, cell)
                    boundarySeeds.append(Int32(cell))
                }
            }
        }
        expand(from: &boundarySeeds, marked: &exterior, rounds: volume, blocked: dilated)

        // 4. Erode: anything within `radius` of the exterior is surface skin,
        // not sealed interior. The void beyond the grid is exterior too, so
        // every wall cell seeds the erosion regardless of dilation — without
        // this, occupancy dilated against a domain wall pinches off pockets
        // that classify interior and clip the membrane at the grid edge.
        var nearExterior = exterior
        var exteriorSeeds = [Int32]()
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution {
                    let cell = index(x, y, z)
                    let wall = x == 0 || x == resolution - 1
                        || y == 0 || y == resolution - 1
                        || z == 0 || z == resolution - 1
                    if get(exterior, cell) || (wall && !get(nearExterior, cell)) {
                        if wall && !get(nearExterior, cell) { set(&nearExterior, cell) }
                        exteriorSeeds.append(Int32(cell))
                    }
                }
            }
        }
        expand(from: &exteriorSeeds, marked: &nearExterior, rounds: radius, blocked: nil)

        var interior = [UInt64](repeating: 0, count: words)
        for word in 0..<words {
            interior[word] = ~nearExterior[word]
        }
        // Mask tail bits beyond the volume.
        let tail = volume & 63
        if tail != 0 {
            interior[words - 1] &= (1 << UInt64(tail)) - 1
        }
        self.interiorBits = interior

        // 5. Mixed pyramid for octree refinement: a level-k block is mixed
        // when its eight children disagree or any child is mixed.
        var pyramid: [[UInt64]] = []
        var levelResolution = resolution
        var levelAll = interior
        var levelAny = interior
        while levelResolution > 1 {
            let half = levelResolution / 2
            let halfVolume = half * half * half
            let halfWords = (halfVolume + 63) / 64
            var nextAll = [UInt64](repeating: 0, count: halfWords)
            var nextAny = [UInt64](repeating: 0, count: halfWords)
            var mixed = [UInt64](repeating: 0, count: halfWords)
            func levelIndex(_ x: Int, _ y: Int, _ z: Int, _ r: Int) -> Int {
                (x * r + y) * r + z
            }
            for x in 0..<half {
                for y in 0..<half {
                    for z in 0..<half {
                        var all = true
                        var any = false
                        for corner in 0..<8 {
                            let child = levelIndex(
                                x * 2 + (corner & 1),
                                y * 2 + ((corner >> 1) & 1),
                                z * 2 + ((corner >> 2) & 1),
                                levelResolution
                            )
                            let bit = levelAll[child >> 6] & (1 << UInt64(child & 63)) != 0
                            let anyBit = levelAny[child >> 6] & (1 << UInt64(child & 63)) != 0
                            all = all && bit
                            any = any || anyBit
                        }
                        let cell = levelIndex(x, y, z, half)
                        if all { nextAll[cell >> 6] |= 1 << UInt64(cell & 63) }
                        if any { nextAny[cell >> 6] |= 1 << UInt64(cell & 63) }
                        if any != all { mixed[cell >> 6] |= 1 << UInt64(cell & 63) }
                    }
                }
            }
            pyramid.append(mixed)
            levelAll = nextAll
            levelAny = nextAny
            levelResolution = half
        }
        self.mixedPyramid = pyramid
    }

    /// Whether the cell at full-grid coordinates is sealed interior.
    func isInterior(x: Int32, y: Int32, z: Int32) -> Bool {
        let cx = min(max(Int(x), 0), resolution - 1)
        let cy = min(max(Int(y), 0), resolution - 1)
        let cz = min(max(Int(z), 0), resolution - 1)
        let cell = (cx * resolution + cy) * resolution + cz
        return interiorBits[cell >> 6] & (1 << UInt64(cell & 63)) != 0
    }

    /// Whether a voxel at `levelResolution` granularity straddles the
    /// classification boundary (so octree refinement must descend into it).
    /// The full-resolution test asks whether the cell's eight corner grid
    /// points see mixed interior-ness — exactly the condition under which
    /// dual contouring emits crossings on the cell's edges — so every cell
    /// participating in a membrane quad stays active. Coarse levels widen
    /// the pyramid's mixed bit to the 26-neighborhood, since a boundary can
    /// sit exactly on a block seam.
    func straddlesBoundary(x: Int32, y: Int32, z: Int32, levelResolution: Int) -> Bool {
        guard levelResolution < resolution else {
            let first = isInterior(x: x, y: y, z: z)
            for corner in 1..<8 {
                let mixed = isInterior(
                    x: x + Int32(corner & 1),
                    y: y + Int32((corner >> 1) & 1),
                    z: z + Int32((corner >> 2) & 1)
                ) != first
                if mixed { return true }
            }
            return false
        }
        var level = 0
        var r = resolution
        while r / 2 > levelResolution {
            r /= 2
            level += 1
        }
        // pyramid[level] has resolution `resolution >> (level + 1)`.
        let half = resolution >> (level + 1)
        guard half == levelResolution else { return true }
        let bits = mixedPyramid[level]
        for dx in Int32(-1)...1 {
            for dy in Int32(-1)...1 {
                for dz in Int32(-1)...1 {
                    let cx = min(max(Int(x + dx), 0), half - 1)
                    let cy = min(max(Int(y + dy), 0), half - 1)
                    let cz = min(max(Int(z + dz), 0), half - 1)
                    let cell = (cx * half + cy) * half + cz
                    if bits[cell >> 6] & (1 << UInt64(cell & 63)) != 0 { return true }
                }
            }
        }
        return false
    }
}
