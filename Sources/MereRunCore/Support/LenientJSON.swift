import Foundation

struct DynamicCodingKey: CodingKey, Sendable {
    let stringValue: String
    let intValue: Int?

    init(_ stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(stringValue: String) {
        self.init(stringValue)
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

struct LenientInt: Codable, Sendable, Hashable {
    let value: Int

    init(_ value: Int) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self.value = value
            return
        }
        if let value = try? container.decode(Int64.self),
           value >= Int64(Int.min),
           value <= Int64(Int.max) {
            self.value = Int(value)
            return
        }
        if let value = try? container.decode(UInt64.self),
           value <= UInt64(Int.max) {
            self.value = Int(value)
            return
        }
        if let value = try? container.decode(Double.self),
           value.isFinite,
           value >= Double(Int.min),
           value <= Double(Int.max) {
            self.value = Int(value)
            return
        }
        if let value = try? container.decode(String.self),
           let parsed = Int(value.trimmingCharacters(in: .whitespacesAndNewlines)) {
            self.value = parsed
            return
        }
        throw DecodingError.typeMismatch(
            Int.self,
            DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected an integer-compatible JSON value.")
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

struct LenientUInt64: Codable, Sendable, Hashable {
    let value: UInt64

    init(_ value: UInt64) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(UInt64.self) {
            self.value = value
            return
        }
        if let value = try? container.decode(Int.self), value >= 0 {
            self.value = UInt64(value)
            return
        }
        if let value = try? container.decode(Int64.self), value >= 0 {
            self.value = UInt64(value)
            return
        }
        if let value = try? container.decode(Double.self),
           value.isFinite,
           value >= 0,
           value <= Double(UInt64.max) {
            self.value = UInt64(value)
            return
        }
        if let value = try? container.decode(String.self),
           let parsed = UInt64(value.trimmingCharacters(in: .whitespacesAndNewlines)) {
            self.value = parsed
            return
        }
        throw DecodingError.typeMismatch(
            UInt64.self,
            DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected an unsigned integer-compatible JSON value.")
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

struct LenientDouble: Codable, Sendable, Hashable {
    let value: Double

    init(_ value: Double) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Double.self) {
            self.value = value
            return
        }
        if let value = try? container.decode(Int.self) {
            self.value = Double(value)
            return
        }
        if let value = try? container.decode(UInt64.self) {
            self.value = Double(value)
            return
        }
        if let value = try? container.decode(String.self),
           let parsed = Double(value.trimmingCharacters(in: .whitespacesAndNewlines)) {
            self.value = parsed
            return
        }
        throw DecodingError.typeMismatch(
            Double.self,
            DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected a number-compatible JSON value.")
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}
