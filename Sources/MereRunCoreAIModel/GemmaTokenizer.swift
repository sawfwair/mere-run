// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// Exact re-implementation of the Hugging Face `tokenizers` encode pipeline for the Gemma `tokenizer.json`
// shipped in this bundle:
//
//   added tokens   leftmost-longest match of the 24 special tokens on the RAW text (they are `normalized: false`)
//   normalizer     Replace " " -> "\u{2581}" on every remaining piece
//   pre_tokenizer  Split(" ") — a no-op after the normalizer, so each piece is ONE BPE word
//   model          BPE, byte_fallback (a scalar missing from the vocab becomes its UTF-8 bytes as `<0xNN>` tokens),
//                  merges applied by rank, ties broken by position (the `tokenizers` heap algorithm)
//
// Everything operates on Unicode scalars and UTF-8 bytes: no grapheme-cluster (`Character`) iteration and no
// String equality (which is canonical equivalence in Swift). The vocabulary is keyed by exact UTF-8 bytes, so
// "é" (U+00E9) and "e" + U+0301 are different keys, as they are for `tokenizers`. tokenizer.json is read with
// ExactJSON, not Foundation (which strips a leading U+FEFF from string values and corrupts three vocab entries).
// Verified byte-identical to `tokenizers` 0.23 on tests/tokenizer-cases.json (3,758 texts, 608,225 ids).

import Foundation

public final class GemmaTokenizer: Sendable {
    public struct LoadError: Error, CustomStringConvertible {
        public let description: String
    }

    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let unkID: Int32
    public let vocabSize: Int

    private let vocab: [[UInt8]: Int32]
    private let tokenBytes: [[UInt8]]            // id -> bytes (for decode)
    private let merges: [UInt64: MergeTarget]    // (left id << 32 | right id) -> rank, merged id
    private let byteIDs: [Int32]                 // 256 entries: id of "<0xNN>"
    private let asciiIDs: [Int32]                // 128 entries: id of the 1-byte token, or -1
    private let added: [AddedToken]              // longest content first
    private let addedFirstScalars: Set<UInt32>
    private let specialIDs: Set<Int32>

    private struct MergeTarget { let rank: Int32; let id: Int32 }
    private struct AddedToken { let scalars: [Unicode.Scalar]; let id: Int32 }

    /// Loads `tokenizer.json`. Throws if the file is not the Gemma BPE configuration this class implements.
    public convenience init(contentsOf url: URL) throws {
        try self.init(data: try Data(contentsOf: url))
    }

    public init(data: Data) throws {
        let root = try ExactJSON.parse(data)
        // --- configuration guards: refuse anything this implementation does not reproduce exactly ---
        guard let norm = root["normalizer"], norm["type"]?.string == "Replace",
              norm["pattern"]?["String"]?.string == " ", norm["content"]?.stringBytes == Array("\u{2581}".utf8)
        else { throw LoadError(description: "unsupported normalizer (expected Replace \" \" -> U+2581)") }
        guard let pre = root["pre_tokenizer"], pre["type"]?.string == "Split", pre["pattern"]?["String"]?.string == " "
        else { throw LoadError(description: "unsupported pre_tokenizer (expected Split \" \")") }
        guard let model = root["model"], model["type"]?.string == "BPE", model["byte_fallback"]?.bool == true,
              model["continuing_subword_prefix"]?.isNull ?? true, model["end_of_word_suffix"]?.isNull ?? true,
              model["dropout"]?.isNull ?? true, (model["ignore_merges"]?.bool ?? false) == false
        else { throw LoadError(description: "unsupported BPE model configuration") }
        guard let vocabPairs = model["vocab"]?.pairs, let mergeList = model["merges"]?.array else {
            throw LoadError(description: "tokenizer.json: missing vocab/merges")
        }

        // --- vocab: exact UTF-8 bytes -> id ---
        var vocab = [[UInt8]: Int32](minimumCapacity: vocabPairs.count)
        var maxID: Int32 = -1
        for (key, value) in vocabPairs {
            guard let id = value.int else { throw LoadError(description: "vocab value is not an integer") }
            vocab[key] = Int32(id)
            if Int32(id) > maxID { maxID = Int32(id) }
        }
        guard vocab.count == vocabPairs.count else {
            throw LoadError(description: "duplicate vocab keys (\(vocab.count) != \(vocabPairs.count))")
        }
        var tokenBytes = [[UInt8]](repeating: [], count: Int(maxID) + 1)
        for (bytes, id) in vocab { tokenBytes[Int(id)] = bytes }

        // --- merges: [a, b] (or "a b") -> rank, id of a+b ---
        var merges = [UInt64: MergeTarget](minimumCapacity: mergeList.count)
        for (rank, m) in mergeList.enumerated() {
            let a: [UInt8], b: [UInt8]
            if let pair = m.array, pair.count == 2, let x = pair[0].stringBytes, let y = pair[1].stringBytes {
                (a, b) = (x, y)
            } else if let s = m.stringBytes, let sp = s.firstIndex(of: 0x20) {
                (a, b) = (Array(s[..<sp]), Array(s[(sp + 1)...]))
            } else {
                throw LoadError(description: "merge #\(rank) has an unexpected shape")
            }
            guard let ia = vocab[a], let ib = vocab[b], let iab = vocab[a + b] else {
                throw LoadError(description: "merge #\(rank) references a token missing from the vocab")
            }
            merges[GemmaTokenizer.pairKey(ia, ib)] = MergeTarget(rank: Int32(rank), id: iab)
        }

        // --- byte fallback tokens and the ASCII fast path ---
        var byteIDs = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            guard let id = vocab[Array(String(format: "<0x%02X>", b).utf8)] else {
                throw LoadError(description: "byte fallback token <0x\(String(format: "%02X", b))> missing")
            }
            byteIDs[b] = id
        }
        var asciiIDs = [Int32](repeating: -1, count: 128)
        for c in 0..<128 { asciiIDs[c] = vocab[[UInt8(c)]] ?? -1 }

        // --- added (special) tokens: only the exact-match flavour is supported ---
        var added: [AddedToken] = []
        for t in root["added_tokens"]?.array ?? [] {
            guard let content = t["content"]?.stringBytes, let id = t["id"]?.int else { continue }
            let flags = ["normalized", "lstrip", "rstrip", "single_word"].map { t[$0]?.bool ?? false }
            guard !flags.contains(true) else {
                throw LoadError(description: "added token \(String(decoding: content, as: UTF8.self)) uses normalized/lstrip/rstrip/single_word, which is not supported")
            }
            added.append(AddedToken(scalars: Array(String(decoding: content, as: UTF8.self).unicodeScalars), id: Int32(id)))
        }
        added.sort { $0.scalars.count > $1.scalars.count }

        func special(_ name: String) throws -> Int32 {
            guard let id = vocab[Array(name.utf8)] else { throw LoadError(description: "\(name) missing from vocab") }
            return id
        }
        self.vocab = vocab
        self.tokenBytes = tokenBytes
        self.merges = merges
        self.byteIDs = byteIDs
        self.asciiIDs = asciiIDs
        self.added = added
        self.addedFirstScalars = Set(added.compactMap { $0.scalars.first?.value })
        self.specialIDs = Set(added.map { $0.id })
        self.vocabSize = vocab.count
        self.bosID = try special("<bos>")
        self.eosID = try special("<eos>")
        self.padID = try special("<pad>")
        self.unkID = try special("<unk>")
    }

    @inline(__always) private static func pairKey(_ a: Int32, _ b: Int32) -> UInt64 {
        (UInt64(UInt32(bitPattern: a)) << 32) | UInt64(UInt32(bitPattern: b))
    }

    /// Token ids of `text` without `<bos>`/`<eos>` (the same as `tokenizers` `encode(text, add_special_tokens=False)`).
    /// Special tokens written literally in the text (e.g. "<eos>") are recognised, exactly as `tokenizers` does.
    public func encode(_ text: String) -> [Int32] {
        let scalars = Array(text.unicodeScalars)
        var ids: [Int32] = []
        ids.reserveCapacity(scalars.count / 3 + 4)
        var i = 0, pieceStart = 0
        while i < scalars.count {
            if addedFirstScalars.contains(scalars[i].value), let hit = matchAdded(scalars, at: i) {
                if i > pieceStart { bpe(scalars[pieceStart..<i], into: &ids) }
                ids.append(hit.id)
                i += hit.scalars.count
                pieceStart = i
            } else {
                i += 1
            }
        }
        if pieceStart < scalars.count { bpe(scalars[pieceStart...], into: &ids) }
        return ids
    }

    /// `<bos>` + encode(text) + `<eos>`.
    public func encodeWithSpecials(_ text: String) -> [Int32] {
        [bosID] + encode(text) + [eosID]
    }

    /// Text for a sequence of ids (the bundle's decoder: "\u{2581}" -> " ", byte-fallback tokens fused into UTF-8).
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard id >= 0, Int(id) < tokenBytes.count else { continue }
            if skipSpecialTokens, specialIDs.contains(id) { continue }
            if let b = byteIDs.firstIndex(of: id) { bytes.append(UInt8(b)); continue }
            let t = tokenBytes[Int(id)]
            var j = 0
            while j < t.count {   // replace every U+2581 (E2 96 81) by a space
                if j + 2 < t.count, t[j] == 0xE2, t[j + 1] == 0x96, t[j + 2] == 0x81 { bytes.append(0x20); j += 3 }
                else { bytes.append(t[j]); j += 1 }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func matchAdded(_ s: [Unicode.Scalar], at i: Int) -> AddedToken? {
        for t in added where i + t.scalars.count <= s.count {   // sorted longest first => first hit is the longest
            var k = 0
            while k < t.scalars.count, s[i + k] == t.scalars[k] { k += 1 }
            if k == t.scalars.count { return t }
        }
        return nil
    }

    // MARK: - BPE on one word (the `tokenizers` Word::merge_all algorithm)

    private struct HeapEntry { let rank: Int32; let pos: Int32; let id: Int32 }

    private func bpe(_ piece: ArraySlice<Unicode.Scalar>, into out: inout [Int32]) {
        // 1. initial symbols: one per scalar (space -> U+2581), byte fallback for scalars missing from the vocab
        var syms: [Int32] = []
        syms.reserveCapacity(piece.count)
        var utf8 = [UInt8](repeating: 0, count: 4)
        for sc in piece {
            let v = sc.value
            if v < 128, v != 0x20 {
                let id = asciiIDs[Int(v)]
                syms.append(id >= 0 ? id : byteIDs[Int(v)])
                continue
            }
            let scalar = v == 0x20 ? Unicode.Scalar(0x2581)! : sc
            var n = 0
            for b in String(scalar).utf8 { utf8[n] = b; n += 1 }   // the UTF-8 of ONE scalar: nothing is normalized
            let key = Array(utf8[0..<n])
            if let id = vocab[key] {
                syms.append(id)
            } else {
                for b in key { syms.append(byteIDs[Int(b)]) }
            }
        }
        let n = syms.count
        if n < 2 { out.append(contentsOf: syms); return }

        // 2. merge by rank, leftmost first among equal ranks (binary heap on (rank, pos))
        var next = [Int32](repeating: 0, count: n), prev = [Int32](repeating: 0, count: n)
        var alive = [Bool](repeating: true, count: n)
        for i in 0..<n { next[i] = Int32(i + 1); prev[i] = Int32(i - 1) }
        var heap: [HeapEntry] = []
        heap.reserveCapacity(n)
        @inline(__always) func less(_ a: HeapEntry, _ b: HeapEntry) -> Bool {
            a.rank != b.rank ? a.rank < b.rank : a.pos < b.pos
        }
        func push(_ e: HeapEntry) {
            heap.append(e)
            var c = heap.count - 1
            while c > 0 {
                let p = (c - 1) >> 1
                if less(heap[c], heap[p]) { heap.swapAt(c, p); c = p } else { break }
            }
        }
        func pop() -> HeapEntry {
            let top = heap[0]
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var p = 0
                while true {
                    let l = 2 * p + 1, r = l + 1
                    var m = p
                    if l < heap.count, less(heap[l], heap[m]) { m = l }
                    if r < heap.count, less(heap[r], heap[m]) { m = r }
                    if m == p { break }
                    heap.swapAt(p, m); p = m
                }
            }
            return top
        }
        for i in 0..<(n - 1) {
            if let m = merges[GemmaTokenizer.pairKey(syms[i], syms[i + 1])] {
                push(HeapEntry(rank: m.rank, pos: Int32(i), id: m.id))
            }
        }
        while !heap.isEmpty {
            let e = pop()
            let i = Int(e.pos)
            if !alive[i] { continue }
            let j = Int(next[i])
            if j >= n || !alive[j] { continue }
            guard let m = merges[GemmaTokenizer.pairKey(syms[i], syms[j])], m.rank == e.rank else { continue }
            syms[i] = m.id
            alive[j] = false
            next[i] = next[j]
            if Int(next[i]) < n { prev[Int(next[i])] = Int32(i) }
            if prev[i] >= 0, let m2 = merges[GemmaTokenizer.pairKey(syms[Int(prev[i])], syms[i])] {
                push(HeapEntry(rank: m2.rank, pos: prev[i], id: m2.id))
            }
            if Int(next[i]) < n, let m2 = merges[GemmaTokenizer.pairKey(syms[i], syms[Int(next[i])])] {
                push(HeapEntry(rank: m2.rank, pos: Int32(i), id: m2.id))
            }
        }
        for i in 0..<n where alive[i] { out.append(syms[i]) }
    }
}

#endif
