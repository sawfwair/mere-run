import Foundation
import MLX
import MLXFast

public enum Gemma4KVQuantizationScheme: String, Sendable, Hashable {
    case uniform
    case polar
    case turboquant
}

public struct Gemma4KVCacheQuantization: Sendable, Hashable {
    public static let defaultScheme: Gemma4KVQuantizationScheme = .uniform
    public static let defaultGroupSize = 64
    public static let defaultQuantizedStart = 5_000

    public var bits: Double?
    public var scheme: Gemma4KVQuantizationScheme
    public var groupSize: Int
    public var quantizedStart: Int

    public init(
        bits: Double? = nil,
        scheme: Gemma4KVQuantizationScheme = Gemma4KVCacheQuantization.defaultScheme,
        groupSize: Int = Gemma4KVCacheQuantization.defaultGroupSize,
        quantizedStart: Int = Gemma4KVCacheQuantization.defaultQuantizedStart
    ) {
        self.bits = bits
        self.scheme = scheme
        self.groupSize = groupSize
        self.quantizedStart = quantizedStart
    }

    public var isEnabled: Bool {
        bits != nil
    }

    package var statusDescription: String {
        guard let bits else {
            return "full-precision"
        }
        return "\(scheme.rawValue):\(bits)-bit:start-\(quantizedStart)"
    }

    package func validated() throws -> Gemma4KVCacheQuantization {
        guard let bits else {
            return self
        }

        guard bits >= 2, bits <= 8 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 KV quantization bits must be between 2 and 8 (received \(bits)).")
        }
        guard groupSize > 0 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 KV quantization group size must be greater than zero.")
        }
        guard quantizedStart >= 0 else {
            throw Gemma4Error.unsupportedConfiguration("Gemma4 quantized KV start must be zero or greater.")
        }

        let roundedHalf = (bits * 2).rounded() / 2
        switch scheme {
        case .uniform:
            guard Swift.abs(bits.rounded() - bits) < 0.000_001 else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 uniform KV quantization requires an integer bit width. Use turboquant for fractional .5 widths.")
            }
        case .polar:
            guard Swift.abs(bits.rounded() - bits) < 0.000_001,
                  [2, 3, 4].contains(Int(bits.rounded())) else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 PolarKV quantization currently supports 2, 3, or 4 bits (received \(bits)).")
            }
        case .turboquant:
            guard Swift.abs(roundedHalf - bits) < 0.000_001 else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 turboquant currently supports integer and .5 bit widths (received \(bits)).")
            }
        }

        return self
    }

    var keyBits: Int? {
        guard let bits else { return nil }
        switch scheme {
        case .uniform, .polar:
            return Int(bits.rounded())
        case .turboquant:
            return Int(floor(bits))
        }
    }

    var valueBits: Int? {
        guard let bits else { return nil }
        switch scheme {
        case .uniform, .polar:
            return Int(bits.rounded())
        case .turboquant:
            return Int(ceil(bits))
        }
    }
}
