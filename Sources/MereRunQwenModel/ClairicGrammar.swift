import Foundation

/// Byte grammar for the released JSON slot protocol. EOS is admitted only at
/// an accepting state. A token may cross multiple JSON boundaries.
package struct ClairicGrammar {
    package struct State: Hashable {
        fileprivate let nodes: [Int]
    }

    private struct Edge {
        let bytes: ClosedRange<UInt8>
        let target: Int
    }

    private var edges: [[Edge]] = [[], []]
    private let end = 1

    package indirect enum Value {
        case string, number, integer, boolean
        case choices([String])
        case object([(String, Value)])
        case nullable(Value)
    }

    package init(_ value: Value) {
        build(value, from: 0, to: end)
        whitespace(at: 0)
        whitespace(at: end)
    }

    package var start: State { State(nodes: [0]) }

    package func accepts(_ state: State) -> Bool { state.nodes.contains(end) }

    package func advance(_ state: State, bytes: [UInt8]) -> State? {
        var nodes = state.nodes
        for byte in bytes {
            var next: [Int] = []
            for node in nodes {
                for edge in edges[node] where edge.bytes.contains(byte) && !next.contains(edge.target) {
                    next.append(edge.target)
                }
            }
            guard !next.isEmpty else { return nil }
            nodes = next.sorted()
        }
        return State(nodes: nodes)
    }

    package func leadingBytes(for state: State) -> Set<UInt8> {
        Set(state.nodes.flatMap { edges[$0].flatMap { Array($0.bytes) } })
    }

    private mutating func node() -> Int {
        edges.append([])
        return edges.count - 1
    }

    private mutating func edge(_ bytes: ClosedRange<UInt8>, _ from: Int, _ to: Int) {
        edges[from].append(Edge(bytes: bytes, target: to))
    }

    private mutating func edge(_ byte: UInt8, _ from: Int, _ to: Int) {
        edge(byte...byte, from, to)
    }

    private mutating func whitespace(at node: Int) {
        edge(9...10, node, node)
        edge(13, node, node)
        edge(32, node, node)
    }

    private mutating func literal(_ value: String, _ from: Int, _ to: Int) {
        let bytes = Array(value.utf8)
        var current = from
        for (index, byte) in bytes.enumerated() {
            let next = index == bytes.count - 1 ? to : node()
            edge(byte, current, next)
            current = next
        }
    }

    private mutating func build(_ value: Value, from: Int, to: Int) {
        switch value {
        case .string:
            string(from: from, to: to)
        case .number, .integer:
            if case .integer = value {
                number(from: from, to: to, integer: true)
            } else {
                number(from: from, to: to, integer: false)
            }
        case .boolean:
            literal("true", from, to)
            literal("false", from, to)
        case .choices(let choices):
            for choice in Set(choices).sorted() { literal(choice, from, to) }
        case .object(let properties):
            var current = node()
            edge(123, from, current)
            for (index, property) in properties.enumerated() {
                whitespace(at: current)
                let keyEnd = node()
                literal(property.0, current, keyEnd)
                whitespace(at: keyEnd)
                let start = node()
                edge(58, keyEnd, start)
                whitespace(at: start)
                let finish = node()
                build(property.1, from: start, to: finish)
                whitespace(at: finish)
                let delimiter: UInt8 = index == properties.count - 1 ? 125 : 44
                current = delimiter == 125 ? to : node()
                edge(delimiter, finish, current)
            }
        case .nullable(let wrapped):
            build(wrapped, from: from, to: to)
            literal("null", from, to)
        }
    }

    private mutating func string(from: Int, to: Int) {
        let body = node(), escape = node()
        edge(34, from, body)
        edge(34, body, to)
        edge(32...33, body, body)
        edge(35...91, body, body)
        edge(93...127, body, body)
        edge(92, body, escape)
        for byte in "\"\\/bfnrt".utf8 { edge(byte, escape, body) }

        var current = node()
        edge(117, escape, current)
        for index in 0..<4 {
            let next = index == 3 ? body : node()
            edge(48...57, current, next)
            edge(65...70, current, next)
            edge(97...102, current, next)
            current = next
        }

        let last = node(), penultimate = node()
        edge(128...191, last, body)
        edge(128...191, penultimate, last)
        edge(194...223, body, last)

        let lowThree = node(), highThree = node()
        edge(224, body, lowThree)
        edge(160...191, lowThree, last)
        edge(225...236, body, penultimate)
        edge(237, body, highThree)
        edge(128...159, highThree, last)
        edge(238...239, body, penultimate)

        let lowFour = node(), middleFour = node(), highFour = node()
        edge(240, body, lowFour)
        edge(144...191, lowFour, penultimate)
        edge(241...243, body, middleFour)
        edge(128...191, middleFour, penultimate)
        edge(244, body, highFour)
        edge(128...143, highFour, penultimate)
    }

    private mutating func number(from: Int, to: Int, integer: Bool) {
        // Keep the numeric state alongside the accepting state. Delimiters
        // and trailing whitespace consume the accepting branch only.
        let sign = node(), zero = node(), whole = node()
        edge(45, from, sign)
        for start in [from, sign] {
            edge(48, start, zero)
            edge(48...57, start, to)
            edge(49...57, start, whole)
        }
        edge(48...57, whole, whole)
        edge(48...57, whole, to)
        guard !integer else { return }

        let point = node(), fraction = node(), exponent = node(), exponentSign = node(), digits = node()
        for start in [zero, whole] { edge(46, start, point) }
        for start in [point, fraction] {
            edge(48...57, start, fraction)
            edge(48...57, start, to)
        }
        for start in [zero, whole, fraction] {
            edge(69, start, exponent)
            edge(101, start, exponent)
        }
        edge(43, exponent, exponentSign)
        edge(45, exponent, exponentSign)
        for start in [exponent, exponentSign, digits] {
            edge(48...57, start, digits)
            edge(48...57, start, to)
        }
    }
}
