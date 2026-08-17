import Foundation

/// Text utilities for the worker event stream.
///
/// Relay serves `/api/graph-jobs/{id}/events` as SSE-framed lines over the
/// worker NDJSON; SSH executors return the raw NDJSON file. Both normalize to
/// the same event-per-line form here, so clients can poll a snapshot or hold
/// the stream open and feed it through the same parser.
public enum RelayEventText {
    /// Strips SSE framing (`event:`/`id:` fields, `data:` prefixes, and the
    /// `data: [DONE]` sentinel) and blank lines, returning bare event lines.
    public static func normalizedEventLines(_ raw: String) -> [String] {
        raw.split(whereSeparator: \Character.isNewline).compactMap { substring in
            let line = substring.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("event:"), !line.hasPrefix("id:"), line != "data: [DONE]" else {
                return nil
            }
            if line.hasPrefix("data:") {
                return line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            }
            return line
        }
    }

    /// Decodes normalized event lines into typed run events, skipping lines
    /// that are not valid event JSON (the stream may interleave non-event
    /// diagnostics).
    public static func decodedEvents(_ raw: String) -> [GraphRunEvent] {
        let decoder = WorkflowBundleCodec.decoder()
        return normalizedEventLines(raw).compactMap { line in
            try? decoder.decode(GraphRunEvent.self, from: Data(line.utf8))
        }
    }
}
