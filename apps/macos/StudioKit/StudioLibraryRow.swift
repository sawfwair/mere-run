import Foundation

/// One element of `library.json`. A row this build can read is an item; any other row is kept
/// as the JSON it was, so a rewrite never drops history another version of the app wrote.
///
/// The file stays a top-level array of row objects: older builds keep reading it, and a row a
/// newer build wrote (an unknown mode, a field with a new type) rides along untouched until a
/// build that understands it loads the file.
package enum StudioLibraryRow: Codable {
    case item(StudioLibraryItem)
    case preserved(StudioLibraryJSON)

    package init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let item = try? container.decode(StudioLibraryItem.self) {
            self = .item(item)
            return
        }
        let json = try container.decode(StudioLibraryJSON.self)
        if let migrated = try? Self.migratingOlderDraft(json) {
            self = .item(migrated)
        } else {
            self = .preserved(json)
        }
    }

    package func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .item(let item): try container.encode(item)
        case .preserved(let json): try container.encode(json)
        }
    }

    /// The one lenient boundary for `CommandDraft`, which stays synthesized everywhere else: a
    /// draft written before a field existed takes the current default for that field. A key
    /// that is present keeps its stored value, so a type the synthesized decoder rejects still
    /// fails here and the row is preserved rather than silently reinterpreted.
    private static func migratingOlderDraft(_ json: StudioLibraryJSON) throws -> StudioLibraryItem? {
        guard case .object(var row) = json, case .object(let draft)? = row["commandDraft"] else { return nil }
        let defaults = try JSONDecoder.mereRunApp.decode(
            [String: StudioLibraryJSON].self,
            from: JSONEncoder.mereRunApp.encode(CommandDraft())
        )
        row["commandDraft"] = .object(defaults.merging(draft) { _, stored in stored })
        let merged = try JSONEncoder.mereRunApp.encode(StudioLibraryJSON.object(row))
        return try JSONDecoder.mereRunApp.decode(StudioLibraryItem.self, from: merged)
    }
}

/// A JSON value carried through `library.json` without interpretation. Integers stay integers
/// so a seed or a count survives a rewrite exactly.
package indirect enum StudioLibraryJSON: Codable, Equatable {
    case string(String)
    case integer(Int)
    case number(Double)
    case bool(Bool)
    case object([String: StudioLibraryJSON])
    case array([StudioLibraryJSON])
    case null

    package init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([StudioLibraryJSON].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: StudioLibraryJSON].self))
        }
    }

    package func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}
