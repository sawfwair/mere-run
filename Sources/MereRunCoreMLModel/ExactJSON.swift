// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// A small, exact JSON reader, written for this package and used to read tokenizer.json. Strings are returned as
// the exact UTF-8 bytes they encode (every \uXXXX escape, including surrogate pairs, is decoded to UTF-8; nothing
// is normalized or stripped). It exists because Foundation's JSONSerialization (macOS 27) drops a leading U+FEFF
// (byte-order mark) from every string VALUE, which turns the vocabulary entries "﻿", "﻿﻿", ...
// into "", "﻿", ... and breaks the tokenizer. (JSONDecoder does not have the bug, but a 32 MB vocabulary
// decoded through Codable is slow, and this reader keeps the bytes exact by construction.)
//
// Supported: the full JSON grammar (RFC 8259). Numbers are returned as Double. Object keys keep their order and
// duplicates (the vocabulary is read as an ordered list of pairs).

import Foundation

public indirect enum ExactJSON {
    case object([(key: [UInt8], value: ExactJSON)])
    case array([ExactJSON])
    case string([UInt8])
    case number(Double)
    case bool(Bool)
    case null

    public struct ParseError: Error, CustomStringConvertible { public let description: String }

    public static func parse(_ data: Data) throws -> ExactJSON {
        try data.withUnsafeBytes { raw -> ExactJSON in
            var p = Parser(bytes: raw.bindMemory(to: UInt8.self))
            p.skipWS()
            let v = try p.value()
            p.skipWS()
            guard p.i == p.bytes.count else { throw ParseError(description: "trailing characters at byte \(p.i)") }
            return v
        }
    }

    public static func parse(contentsOf url: URL) throws -> ExactJSON { try parse(Data(contentsOf: url)) }

    // MARK: accessors
    public subscript(key: String) -> ExactJSON? {
        guard case .object(let pairs) = self else { return nil }
        let k = Array(key.utf8)
        for p in pairs where p.key == k { return p.value }
        return nil
    }
    public var stringBytes: [UInt8]? { if case .string(let b) = self { return b }; return nil }
    public var string: String? { stringBytes.map { String(decoding: $0, as: UTF8.self) } }
    public var double: Double? { if case .number(let d) = self { return d }; return nil }
    public var int: Int? { double.flatMap { $0 == $0.rounded() && abs($0) < 9e15 ? Int($0) : nil } }
    public var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var isNull: Bool { if case .null = self { return true }; return false }
    public var array: [ExactJSON]? { if case .array(let a) = self { return a }; return nil }
    public var pairs: [(key: [UInt8], value: ExactJSON)]? { if case .object(let p) = self { return p }; return nil }

    // MARK: parser
    private struct Parser {
        let bytes: UnsafeBufferPointer<UInt8>
        var i = 0

        init(bytes: UnsafeBufferPointer<UInt8>) { self.bytes = bytes }

        mutating func skipWS() {
            while i < bytes.count, bytes[i] == 0x20 || bytes[i] == 0x0A || bytes[i] == 0x0D || bytes[i] == 0x09 { i += 1 }
        }

        func fail(_ what: String) -> ParseError { ParseError(description: "\(what) at byte \(i)") }

        mutating func value() throws -> ExactJSON {
            guard i < bytes.count else { throw fail("unexpected end") }
            switch bytes[i] {
            case 0x7B: return try object()
            case 0x5B: return try array()
            case 0x22: return .string(try string())
            case 0x74: try literal("true"); return .bool(true)
            case 0x66: try literal("false"); return .bool(false)
            case 0x6E: try literal("null"); return .null
            default: return .number(try number())
            }
        }

        mutating func literal(_ s: StaticString) throws {
            let n = s.utf8CodeUnitCount
            guard i + n <= bytes.count else { throw fail("bad literal") }
            for k in 0..<n where bytes[i + k] != s.utf8Start[k] { throw fail("bad literal") }
            i += n
        }

        mutating func number() throws -> Double {
            let start = i
            while i < bytes.count {
                let c = bytes[i]
                if (c >= 0x30 && c <= 0x39) || c == 0x2D || c == 0x2B || c == 0x2E || c == 0x65 || c == 0x45 { i += 1 } else { break }
            }
            guard i > start, let d = Double(String(decoding: bytes[start..<i], as: UTF8.self)) else { throw fail("bad number") }
            return d
        }

        mutating func object() throws -> ExactJSON {
            i += 1
            var pairs: [(key: [UInt8], value: ExactJSON)] = []
            skipWS()
            if i < bytes.count, bytes[i] == 0x7D { i += 1; return .object(pairs) }
            while true {
                skipWS()
                guard i < bytes.count, bytes[i] == 0x22 else { throw fail("expected key") }
                let k = try string()
                skipWS()
                guard i < bytes.count, bytes[i] == 0x3A else { throw fail("expected ':'") }
                i += 1
                skipWS()
                pairs.append((k, try value()))
                skipWS()
                guard i < bytes.count else { throw fail("unterminated object") }
                if bytes[i] == 0x2C { i += 1; continue }
                if bytes[i] == 0x7D { i += 1; return .object(pairs) }
                throw fail("expected ',' or '}'")
            }
        }

        mutating func array() throws -> ExactJSON {
            i += 1
            var items: [ExactJSON] = []
            skipWS()
            if i < bytes.count, bytes[i] == 0x5D { i += 1; return .array(items) }
            while true {
                skipWS()
                items.append(try value())
                skipWS()
                guard i < bytes.count else { throw fail("unterminated array") }
                if bytes[i] == 0x2C { i += 1; continue }
                if bytes[i] == 0x5D { i += 1; return .array(items) }
                throw fail("expected ',' or ']'")
            }
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= bytes.count else { throw fail("bad \\u escape") }
            var v: UInt32 = 0
            for _ in 0..<4 {
                let c = bytes[i]
                let d: UInt32
                switch c {
                case 0x30...0x39: d = UInt32(c - 0x30)
                case 0x41...0x46: d = UInt32(c - 0x41 + 10)
                case 0x61...0x66: d = UInt32(c - 0x61 + 10)
                default: throw fail("bad hex digit")
                }
                v = v << 4 | d
                i += 1
            }
            return v
        }

        /// Decodes a JSON string starting at the opening quote into exact UTF-8 bytes.
        mutating func string() throws -> [UInt8] {
            i += 1
            var out: [UInt8] = []
            while true {
                guard i < bytes.count else { throw fail("unterminated string") }
                let c = bytes[i]
                if c == 0x22 { i += 1; return out }
                if c != 0x5C { out.append(c); i += 1; continue }
                i += 1
                guard i < bytes.count else { throw fail("unterminated escape") }
                let e = bytes[i]
                i += 1
                switch e {
                case 0x22: out.append(0x22)
                case 0x5C: out.append(0x5C)
                case 0x2F: out.append(0x2F)
                case 0x62: out.append(0x08)
                case 0x66: out.append(0x0C)
                case 0x6E: out.append(0x0A)
                case 0x72: out.append(0x0D)
                case 0x74: out.append(0x09)
                case 0x75:
                    var cp = try hex4()
                    if cp >= 0xD800, cp <= 0xDBFF, i + 6 <= bytes.count, bytes[i] == 0x5C, bytes[i + 1] == 0x75 {
                        let save = i
                        i += 2
                        let lo = try hex4()
                        if lo >= 0xDC00, lo <= 0xDFFF {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                        } else {
                            i = save   // not a pair: keep the lone surrogate (encoded below as 3 bytes, like Rust would reject; never occurs in our files)
                        }
                    }
                    appendUTF8(cp, to: &out)
                default: throw fail("bad escape")
                }
            }
        }

        func appendUTF8(_ cp: UInt32, to out: inout [UInt8]) {
            switch cp {
            case 0..<0x80: out.append(UInt8(cp))
            case 0x80..<0x800: out.append(UInt8(0xC0 | cp >> 6)); out.append(UInt8(0x80 | cp & 0x3F))
            case 0x800..<0x10000: out.append(UInt8(0xE0 | cp >> 12)); out.append(UInt8(0x80 | (cp >> 6) & 0x3F)); out.append(UInt8(0x80 | cp & 0x3F))
            default: out.append(UInt8(0xF0 | cp >> 18)); out.append(UInt8(0x80 | (cp >> 12) & 0x3F)); out.append(UInt8(0x80 | (cp >> 6) & 0x3F)); out.append(UInt8(0x80 | cp & 0x3F))
            }
        }
    }
}

#endif
