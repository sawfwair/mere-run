import Foundation

/// MLX splits one key per serial categorical draw. Siblings at the same depth
/// share that draw, and discarded branches consume no extra keys.
struct Q35SerialTreeKeys {
    typealias Key = SIMD2<UInt32>
    enum Order { case sorted, vocabulary }
    let order: Order
    private var state: Key
    private var start: Int
    private var keys: [Key] = []

    init(seed: UInt64, position: Int, order: Order = .sorted) {
        self.order = order
        state = Key(UInt32(truncatingIfNeeded: seed >> 32), UInt32(truncatingIfNeeded: seed))
        start = position
    }

    mutating func prepare(position: Int, count: Int) {
        precondition(position >= start && position <= start + keys.count)
        keys.removeFirst(position - start)
        start = position
        while keys.count < count {
            let even = Self.pair(key: state, left: 0, right: 2)
            let odd = Self.pair(key: state, left: 1, right: 3)
            state = Key(even.x, odd.x)
            keys.append(Key(even.y, odd.y))
        }
    }

    subscript(position: Int) -> Key { keys[position - start] }

    static func gumbel(key: Key, rank: Int, vocabulary: Int) -> Float {
        let half = vocabulary / 2
        let index = rank % half
        let bits = pair(key: key, left: UInt32(index), right: UInt32(index + half))
        let uniform = min(Float(rank < half ? bits.x : bits.y) * 0x1p-32, Float(1).nextDown)
        return -log(-log(uniform))
    }

    private static let rotations: [UInt32] = [13, 15, 26, 6, 17, 29, 16, 24]

    private static func pair(key: Key, left: UInt32, right: UInt32) -> Key {
        let parity = key.x ^ key.y ^ 0x1bd11bda
        let injections = SIMD3(key.x, key.y, parity)
        var x = left &+ key.x, y = right &+ key.y
        for group in 0..<5 {
            for step in 0..<4 {
                x &+= y
                let shift = rotations[(group % 2) * 4 + step]
                y = ((y << shift) | (y >> (32 - shift))) ^ x
            }
            x &+= injections[(group + 1) % 3]
            y &+= injections[(group + 2) % 3] &+ UInt32(group + 1)
        }
        return Key(x, y)
    }
}
