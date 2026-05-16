import Foundation

public enum DeepseekV4FlashJSONRepair {
    public static func escapingControlCharactersInsideStrings(_ data: Data) -> Data {
        var output = Data()
        output.reserveCapacity(data.count)
        var isInsideString = false
        var isEscaping = false

        for byte in data {
            guard isInsideString else {
                output.append(byte)
                if byte == UInt8(ascii: "\"") {
                    isInsideString = true
                }
                continue
            }

            if isEscaping {
                output.append(byte)
                isEscaping = false
                continue
            }

            switch byte {
            case UInt8(ascii: "\\"):
                output.append(byte)
                isEscaping = true
            case UInt8(ascii: "\""):
                output.append(byte)
                isInsideString = false
            case 0x08:
                output.append(contentsOf: "\\b".utf8)
            case 0x09:
                output.append(contentsOf: "\\t".utf8)
            case 0x0A:
                output.append(contentsOf: "\\n".utf8)
            case 0x0C:
                output.append(contentsOf: "\\f".utf8)
            case 0x0D:
                output.append(contentsOf: "\\r".utf8)
            case 0x00...0x1F:
                output.append(contentsOf: unicodeEscape(for: byte))
            default:
                output.append(byte)
            }
        }

        return output
    }

    private static func unicodeEscape(for byte: UInt8) -> [UInt8] {
        let hex = Array("0123456789abcdef".utf8)
        return [
            UInt8(ascii: "\\"),
            UInt8(ascii: "u"),
            UInt8(ascii: "0"),
            UInt8(ascii: "0"),
            hex[Int(byte >> 4)],
            hex[Int(byte & 0x0F)],
        ]
    }
}
