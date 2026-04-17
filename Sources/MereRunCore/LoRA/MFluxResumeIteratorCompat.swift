import Foundation

enum MFluxResumeIteratorCompat {
    struct State {
        let step: Int
        let seed: UInt64?
        let cursor: Cursor?
    }

    struct Cursor {
        var permutation: [Int]
        var position: Int
        var pythonRNG: PythonRandom?
        var localRNGState: UInt64?

        init(
            permutation: [Int],
            position: Int,
            pythonRNG: PythonRandom? = nil,
            localRNGState: UInt64? = nil
        ) {
            self.permutation = permutation
            self.position = position
            self.pythonRNG = pythonRNG
            self.localRNGState = localRNGState
        }
    }

    private struct LocalSplitMix64 {
        var state: UInt64

        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }

        mutating func nextInt(upperBound: Int) -> Int {
            precondition(upperBound > 0)
            return Int(next() % UInt64(upperBound))
        }
    }

    static func loadState(from iteratorURL: URL) -> State? {
        guard let data = try? Data(contentsOf: iteratorURL),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let step = intValue(raw["num_iterations"]) else {
            return nil
        }

        let seed = uint64Value(raw["seed"])
        let cursor: Cursor? = {
            guard let permutation = intArrayValue(raw["current_permutation"]),
                  !permutation.isEmpty else {
                return nil
            }
            let rawPosition = intValue(raw["position"]) ?? 0
            guard rawPosition >= 0, rawPosition <= permutation.count else {
                return nil
            }
            let pythonRNG = PythonRandom(serializedState: raw["rng_state"])
            let localRNGState = uint64Value(raw["zero_rng_state"])
            return Cursor(
                permutation: permutation,
                position: rawPosition,
                pythonRNG: pythonRNG,
                localRNGState: localRNGState
            )
        }()

        return State(
            step: max(step, 0),
            seed: seed,
            cursor: cursor
        )
    }

    static func nextBatchIndices(
        requestedBatchSize: Int,
        sampleCount: Int,
        cursor: inout Cursor?
    ) -> [Int]? {
        guard requestedBatchSize > 0, sampleCount > 0 else {
            return nil
        }
        guard var state = cursor else {
            return nil
        }
        guard state.permutation.count == sampleCount else {
            // mflux iterator permutation is built from the full dataset.
            // If current phase is a bucket/subset, fall back to local RNG sampling.
            cursor = nil
            return nil
        }

        if state.position >= state.permutation.count {
            if var pythonRNG = state.pythonRNG {
                var nextPermutation = Array(0..<sampleCount)
                pythonRNG.shuffle(&nextPermutation)
                state.permutation = nextPermutation
                state.position = 0
                state.pythonRNG = pythonRNG
            } else if let localRNGState = state.localRNGState {
                var rng = LocalSplitMix64(state: localRNGState)
                var nextPermutation = Array(0..<sampleCount)
                shuffleWithLocalRNG(&nextPermutation, rng: &rng)
                state.permutation = nextPermutation
                state.position = 0
                state.localRNGState = rng.state
            } else {
                cursor = nil
                return nil
            }
        }

        let remaining = state.permutation.count - state.position
        let takeCount = min(requestedBatchSize, remaining)
        guard takeCount > 0 else {
            cursor = nil
            return nil
        }

        let indices = Array(state.permutation[state.position..<(state.position + takeCount)])
        guard indices.allSatisfy({ $0 >= 0 && $0 < sampleCount }) else {
            cursor = nil
            return nil
        }

        state.position += takeCount
        if state.position >= state.permutation.count,
           state.pythonRNG == nil,
           state.localRNGState == nil {
            cursor = nil
        } else {
            cursor = state
        }
        return indices
    }

    static func advanceTrainingRNG(batchSize: Int, cursor: inout Cursor?) {
        guard batchSize > 0 else {
            return
        }
        _ = nextTrainingSeedPairs(batchSize: batchSize, cursor: &cursor)
    }

    static func nextTrainingSeedPairs(
        batchSize: Int,
        cursor: inout Cursor?
    ) -> [(time: UInt64, noise: UInt64)]? {
        guard batchSize > 0,
              var state = cursor else {
            return nil
        }
        var seedPairs: [(time: UInt64, noise: UInt64)] = []
        seedPairs.reserveCapacity(batchSize)

        if var pythonRNG = state.pythonRNG {
            for _ in 0..<batchSize {
                seedPairs.append((pythonRNG.nextSeed(), pythonRNG.nextSeed()))
            }
            state.pythonRNG = pythonRNG
            cursor = state
            return seedPairs
        }
        if let localRNGState = state.localRNGState {
            var rng = LocalSplitMix64(state: localRNGState)
            for _ in 0..<batchSize {
                let timeSeed = UInt64(UInt32(truncatingIfNeeded: rng.next()))
                let noiseSeed = UInt64(UInt32(truncatingIfNeeded: rng.next()))
                seedPairs.append((timeSeed, noiseSeed))
            }
            state.localRNGState = rng.state
            cursor = state
            return seedPairs
        }

        cursor = nil
        return nil
    }

    static func makeLocalCursor(sampleCount: Int, seed: UInt64) -> Cursor? {
        guard sampleCount > 0 else {
            return nil
        }
        var rng = LocalSplitMix64(state: seed)
        var permutation = Array(0..<sampleCount)
        shuffleWithLocalRNG(&permutation, rng: &rng)
        return Cursor(
            permutation: permutation,
            position: 0,
            pythonRNG: nil,
            localRNGState: rng.state
        )
    }

    static func makePythonCursor(sampleCount: Int, seed: UInt64) -> Cursor? {
        guard sampleCount > 0 else {
            return nil
        }
        var pythonRNG = PythonRandom(seed: seed)
        var permutation = Array(0..<sampleCount)
        pythonRNG.shuffle(&permutation)
        return Cursor(
            permutation: permutation,
            position: 0,
            pythonRNG: pythonRNG,
            localRNGState: nil
        )
    }

    struct PythonRandom {
        private static let stateCount = 624
        private static let stateSize = 625
        private static let initializationMultiplier: UInt32 = 1_812_433_253
        private static let keyMixMultiplier: UInt32 = 1_664_525
        private static let keyTwistMultiplier: UInt32 = 1_566_083_941
        private static let defaultSeedForKeyInit: UInt32 = 19_650_218

        private var mt: [UInt32]
        private var index: Int

        init(seed: UInt64) {
            self.mt = Array(repeating: 0, count: Self.stateCount)
            self.index = Self.stateCount
            var keyWords: [UInt32] = []
            var remaining = seed
            repeat {
                keyWords.append(UInt32(truncatingIfNeeded: remaining))
                remaining >>= 32
            } while remaining > 0
            initialize(with: keyWords)
        }

        init?(serializedState: Any?) {
            guard let raw = serializedState as? [Any],
                  raw.count >= 2,
                  let mtWithIndex = raw[1] as? [Any],
                  mtWithIndex.count == Self.stateSize else {
                return nil
            }

            var parsedMT: [UInt32] = []
            parsedMT.reserveCapacity(Self.stateCount)
            for rawValue in mtWithIndex.prefix(Self.stateCount) {
                guard let value = uint64Value(rawValue),
                      value <= UInt64(UInt32.max) else {
                    return nil
                }
                parsedMT.append(UInt32(value))
            }

            guard let parsedIndex = intValue(mtWithIndex[Self.stateCount]),
                  parsedIndex >= 0,
                  parsedIndex <= Self.stateCount else {
                return nil
            }

            self.mt = parsedMT
            self.index = parsedIndex
        }

        func serializedState() -> [Any] {
            var stateTuple: [Any] = mt.map { Int($0) }
            stateTuple.append(index)
            return [3, stateTuple, NSNull()]
        }

        mutating func advanceForTrainingItems(_ count: Int) {
            guard count > 0 else { return }
            for _ in 0..<count {
                _ = randIntInclusive(lower: 0, upper: UInt64(UInt32.max))
                _ = randIntInclusive(lower: 0, upper: UInt64(UInt32.max))
            }
        }

        mutating func nextSeed() -> UInt64 {
            randIntInclusive(lower: 0, upper: UInt64(UInt32.max))
        }

        mutating func shuffle(_ values: inout [Int]) {
            guard values.count > 1 else { return }
            for i in stride(from: values.count - 1, through: 1, by: -1) {
                let j = Int(randBelow(UInt64(i + 1)))
                values.swapAt(i, j)
            }
        }

        private mutating func randIntInclusive(lower: UInt64, upper: UInt64) -> UInt64 {
            precondition(upper >= lower)
            let span = upper - lower + 1
            return lower + randBelow(span)
        }

        private mutating func randBelow(_ upperBound: UInt64) -> UInt64 {
            precondition(upperBound > 0)
            let bitCount = 64 - upperBound.leadingZeroBitCount
            var candidate = getRandBits(bitCount)
            while candidate >= upperBound {
                candidate = getRandBits(bitCount)
            }
            return candidate
        }

        private mutating func getRandBits(_ bitCount: Int) -> UInt64 {
            precondition(bitCount > 0 && bitCount <= 64)

            var remaining = bitCount
            var shift = 0
            var value: UInt64 = 0

            while remaining >= 32 {
                value |= UInt64(nextUInt32()) << shift
                shift += 32
                remaining -= 32
            }

            if remaining > 0 {
                let word = nextUInt32() >> UInt32(32 - remaining)
                value |= UInt64(word) << shift
            }

            return value
        }

        private mutating func nextUInt32() -> UInt32 {
            if index >= Self.stateCount {
                twist()
            }

            var y = mt[index]
            index += 1

            y ^= y >> 11
            y ^= (y << 7) & 0x9D2C5680
            y ^= (y << 15) & 0xEFC60000
            y ^= y >> 18
            return y
        }

        private mutating func twist() {
            let upperMask: UInt32 = 0x8000_0000
            let lowerMask: UInt32 = 0x7FFF_FFFF
            let matrixA: UInt32 = 0x9908_B0DF

            for i in 0..<Self.stateCount {
                let y = (mt[i] & upperMask) | (mt[(i + 1) % Self.stateCount] & lowerMask)
                var mixed = mt[(i + 397) % Self.stateCount] ^ (y >> 1)
                if (y & 1) != 0 {
                    mixed ^= matrixA
                }
                mt[i] = mixed
            }

            index = 0
        }

        private mutating func initialize(with keyWords: [UInt32]) {
            precondition(!keyWords.isEmpty)
            initializeGenerator(seed: Self.defaultSeedForKeyInit)

            var i = 1
            var j = 0
            var k = max(Self.stateCount, keyWords.count)
            while k > 0 {
                let prev = mt[i - 1]
                let mixed = mt[i] ^ ((prev ^ (prev >> 30)) &* Self.keyMixMultiplier)
                let value = UInt64(mixed) &+ UInt64(keyWords[j]) &+ UInt64(j)
                mt[i] = UInt32(truncatingIfNeeded: value)

                i += 1
                j += 1
                if i >= Self.stateCount {
                    mt[0] = mt[Self.stateCount - 1]
                    i = 1
                }
                if j >= keyWords.count {
                    j = 0
                }
                k -= 1
            }

            k = Self.stateCount - 1
            while k > 0 {
                let prev = mt[i - 1]
                let mixed = mt[i] ^ ((prev ^ (prev >> 30)) &* Self.keyTwistMultiplier)
                let value = UInt64(mixed) &- UInt64(i)
                mt[i] = UInt32(truncatingIfNeeded: value)

                i += 1
                if i >= Self.stateCount {
                    mt[0] = mt[Self.stateCount - 1]
                    i = 1
                }
                k -= 1
            }

            mt[0] = 0x8000_0000
            index = Self.stateCount
        }

        private mutating func initializeGenerator(seed: UInt32) {
            mt[0] = seed
            for i in 1..<Self.stateCount {
                let prev = mt[i - 1]
                mt[i] = Self.initializationMultiplier &* (prev ^ (prev >> 30)) &+ UInt32(i)
            }
            index = Self.stateCount
        }
    }

    private static func intValue(_ rawValue: Any?) -> Int? {
        if let value = rawValue as? Int {
            return value
        }
        if let value = rawValue as? NSNumber {
            return value.intValue
        }
        if let value = rawValue as? String {
            return Int(value)
        }
        return nil
    }

    private static func uint64Value(_ rawValue: Any?) -> UInt64? {
        if let value = rawValue as? UInt64 {
            return value
        }
        if let value = rawValue as? Int, value >= 0 {
            return UInt64(value)
        }
        if let value = rawValue as? NSNumber, value.int64Value >= 0 {
            return value.uint64Value
        }
        if let value = rawValue as? String {
            return UInt64(value)
        }
        return nil
    }

    private static func intArrayValue(_ rawValue: Any?) -> [Int]? {
        guard let values = rawValue as? [Any] else { return nil }
        var parsed: [Int] = []
        parsed.reserveCapacity(values.count)
        for rawEntry in values {
            guard let value = intValue(rawEntry) else {
                return nil
            }
            parsed.append(value)
        }
        return parsed
    }

    private static func shuffleWithLocalRNG(_ values: inout [Int], rng: inout LocalSplitMix64) {
        guard values.count > 1 else {
            return
        }
        for i in stride(from: values.count - 1, through: 1, by: -1) {
            let j = rng.nextInt(upperBound: i + 1)
            values.swapAt(i, j)
        }
    }
}
