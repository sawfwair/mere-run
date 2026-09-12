import Foundation

struct MultipartFormData: Equatable, Sendable {
    struct Part: Equatable, Sendable {
        let name: String
        let filename: String?
        let contentType: String?
        let body: Data
    }

    enum ParseError: LocalizedError, Equatable {
        case missingBoundary
        case malformedBody
        case missingName

        var errorDescription: String? {
            switch self {
            case .missingBoundary:
                return "Missing multipart boundary."
            case .malformedBody:
                return "Malformed multipart body."
            case .missingName:
                return "Multipart part is missing a form-data name."
            }
        }
    }

    let parts: [Part]

    /// Validates transport fields without changing the order of uploaded parts.
    func validateFields(
        textFields: Set<String>,
        fileFields: Set<String>,
        unsupportedTextMessage: String,
        unsupportedFileMessage: String,
        requiresUTF8Text: Bool = true
    ) throws {
        for part in parts {
            if part.filename != nil {
                guard fileFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(part.name, unsupportedFileMessage)
                }
            } else {
                guard textFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(part.name, unsupportedTextMessage)
                }
                if requiresUTF8Text, String(data: part.body, encoding: .utf8) == nil {
                    throw APIRequestValidationError.invalidField(part.name, "must contain valid UTF-8 text")
                }
            }
        }
        for field in textFields where parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }
    }

    static func parse(body: Data, boundary: String?) throws -> MultipartFormData {
        guard let boundary, !boundary.isEmpty else {
            throw ParseError.missingBoundary
        }
        let marker = Data("--\(boundary)".utf8)
        guard !marker.isEmpty,
              let firstMarker = body.range(of: marker, options: [], in: body.startIndex..<body.endIndex) else {
            throw ParseError.malformedBody
        }

        var parts: [Part] = []
        var markerRange = firstMarker
        while true {
            let afterMarker = markerRange.upperBound
            if body.hasBytes(Data("--".utf8), at: afterMarker) {
                break
            }
            let partStart = body.indexAfterLineBreak(at: afterMarker)
            guard let nextMarker = body.range(of: marker, options: [], in: partStart..<body.endIndex) else {
                throw ParseError.malformedBody
            }
            let partEnd = body.indexTrimmingLineBreak(before: nextMarker.lowerBound)
            if partStart < partEnd {
                parts.append(try parsePart(Data(body[partStart..<partEnd])))
            }
            markerRange = nextMarker
        }

        return MultipartFormData(parts: parts)
    }

    func field(_ name: String) -> String? {
        guard let part = parts.first(where: { $0.name == name && $0.filename == nil }) else {
            return nil
        }
        return String(data: part.body, encoding: .utf8)
    }

    func file(named name: String) -> Part? {
        parts.first { $0.name == name && $0.filename != nil }
    }

    func files(named name: String) -> [Part] {
        parts.filter { $0.name == name && $0.filename != nil }
    }

    private static func parsePart(_ data: Data) throws -> Part {
        let separator = Data("\r\n\r\n".utf8)
        let fallbackSeparator = Data("\n\n".utf8)
        let separatorRange = data.range(of: separator, options: [], in: data.startIndex..<data.endIndex)
            ?? data.range(of: fallbackSeparator, options: [], in: data.startIndex..<data.endIndex)
        guard let separatorRange,
              let headerText = String(data: data[data.startIndex..<separatorRange.lowerBound], encoding: .utf8) else {
            throw ParseError.malformedBody
        }
        let body = Data(data[separatorRange.upperBound..<data.endIndex])
        let headers = parseHeaders(headerText)
        guard let disposition = headers["content-disposition"] else {
            throw ParseError.missingName
        }
        let params = parseDispositionParameters(disposition)
        guard let name = params["name"], !name.isEmpty else {
            throw ParseError.missingName
        }
        return Part(
            name: name,
            filename: params["filename"],
            contentType: headers["content-type"],
            body: body
        )
    }

    private static func parseHeaders(_ text: String) -> [String: String] {
        var headers: [String: String] = [:]
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[name] = value
        }
        return headers
    }

    private static func parseDispositionParameters(_ value: String) -> [String: String] {
        var params: [String: String] = [:]
        for part in value.split(separator: ";", omittingEmptySubsequences: false).dropFirst() {
            let pair = part.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            var rawValue = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if rawValue.hasPrefix("\""), rawValue.hasSuffix("\""), rawValue.count >= 2 {
                rawValue.removeFirst()
                rawValue.removeLast()
            }
            params[key] = rawValue
        }
        return params
    }
}

private extension Data {
    func hasBytes(_ bytes: Data, at index: Data.Index) -> Bool {
        guard index >= startIndex, index + bytes.count <= endIndex else {
            return false
        }
        return self[index..<(index + bytes.count)].elementsEqual(bytes)
    }

    func indexAfterLineBreak(at index: Data.Index) -> Data.Index {
        if hasBytes(Data("\r\n".utf8), at: index) {
            return index + 2
        }
        if hasBytes(Data("\n".utf8), at: index) {
            return index + 1
        }
        return index
    }

    func indexTrimmingLineBreak(before index: Data.Index) -> Data.Index {
        if index >= 2, self[(index - 2)..<index].elementsEqual(Data("\r\n".utf8)) {
            return index - 2
        }
        if index >= 1, self[(index - 1)..<index].elementsEqual(Data("\n".utf8)) {
            return index - 1
        }
        return index
    }
}
