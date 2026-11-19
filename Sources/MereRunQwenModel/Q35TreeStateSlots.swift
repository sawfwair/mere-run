/// Traverse smaller subtrees first, leaving the largest subtree until its
/// parent slot can be reused. Snapshots live only until their final child.
struct Q35TreeStateSlots: Sendable {
    let order: [Int]
    let reads: [Int]
    let writes: [Int]
    let count: Int

    init(parents: [Int]) {
        var children = Array(repeating: [Int](), count: parents.count)
        for row in parents.indices.dropFirst() { children[parents[row]].append(row) }
        var required = Array(repeating: 0, count: parents.count)
        for row in parents.indices.reversed() where !children[row].isEmpty {
            let needs = children[row].map { required[$0] }.sorted()
            let earlier = needs.count > 1 ? needs[needs.count - 2] + 1 : 0
            required[row] = max(1, max(needs.last!, earlier))
        }
        for row in parents.indices {
            children[row].sort { required[$0] == required[$1] ? $0 < $1 : required[$0] < required[$1] }
        }
        var order: [Int] = []
        func visit(_ row: Int) {
            order.append(row)
            for child in children[row] { visit(child) }
        }
        visit(0)
        var lastChild = Array(repeating: -1, count: parents.count)
        for row in order.dropFirst() { lastChild[parents[row]] = row }
        var assigned = Array(repeating: -1, count: parents.count)
        var reads = assigned
        var occupied: [Bool] = []
        for row in order {
            if row > 0 {
                let parent = parents[row]
                reads[row] = assigned[parent]
                if lastChild[parent] == row { occupied[reads[row]] = false }
            }
            if !children[row].isEmpty {
                if let free = occupied.firstIndex(of: false) {
                    assigned[row] = free
                    occupied[free] = true
                } else {
                    assigned[row] = occupied.count
                    occupied.append(true)
                }
            }
        }
        self.order = order
        self.reads = reads
        self.writes = assigned
        self.count = occupied.count
    }
}
