#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation

public enum LoRATrainingFingerprint {
    public static func sha256Hex(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

public struct LoRALossPoint: Codable, Hashable, Sendable {
    public let step: Int
    public let loss: Float

    public init(step: Int, loss: Float) {
        self.step = step
        self.loss = loss
    }
}

public final class LoRATrainingMetricsLogger: @unchecked Sendable {
    public let csvURL: URL
    public let htmlURL: URL

    private var points: [LoRALossPoint]

    public init(baseOutputURL: URL, resumeExisting: Bool) throws {
        self.csvURL = baseOutputURL
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("csv")
        self.htmlURL = baseOutputURL
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("html")

        if resumeExisting {
            self.points = try Self.loadPoints(from: csvURL)
        } else {
            self.points = []
        }

        try writeCSV()
    }

    public func record(step: Int, loss: Float) throws {
        if let index = points.firstIndex(where: { $0.step == step }) {
            points[index] = LoRALossPoint(step: step, loss: loss)
            try writeCSV()
            return
        }

        points.append(LoRALossPoint(step: step, loss: loss))
        let line = "\(step),\(loss)\n"
        if let handle = try? FileHandle(forWritingTo: csvURL) {
            try handle.seekToEnd()
            if let data = line.data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
            try handle.close()
        } else {
            try writeCSV()
        }
    }

    public func writeArtifacts() throws {
        try writeCSV()
        try writeHTML()
    }

    public func allPoints() -> [LoRALossPoint] {
        points
    }

    private func writeCSV() throws {
        var text = "step,loss\n"
        for point in points.sorted(by: { $0.step < $1.step }) {
            text += "\(point.step),\(point.loss)\n"
        }
        try text.write(to: csvURL, atomically: true, encoding: .utf8)
    }

    private func writeHTML() throws {
        let sorted = points.sorted(by: { $0.step < $1.step })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let jsonData = try encoder.encode(sorted)
        let json = String(decoding: jsonData, as: UTF8.self)

        let html = """
        <!doctype html>
        <html lang="en">
        <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>LoRA Training Loss</title>
        <style>
        :root { color-scheme: light; }
        body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; }
        .wrap { max-width: 960px; margin: 32px auto; padding: 0 16px; }
        h1 { font-size: 22px; margin: 0 0 8px; }
        p { margin: 0 0 20px; color: #475569; }
        .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; }
        svg { width: 100%; height: 420px; display: block; }
        .axis { stroke: #94a3b8; stroke-width: 1; }
        .line { fill: none; stroke: #0f766e; stroke-width: 2.25; }
        .dot { fill: #0f766e; }
        .meta { margin-top: 12px; font-size: 12px; color: #64748b; }
        </style>
        </head>
        <body>
        <div class="wrap">
          <h1>LoRA Training Loss</h1>
          <p>Points logged at training progress intervals.</p>
          <div class="card">
            <svg id="chart" viewBox="0 0 920 420" preserveAspectRatio="none"></svg>
            <div class="meta" id="meta"></div>
          </div>
        </div>
        <script>
        const points = \(json);
        const svg = document.getElementById('chart');
        const meta = document.getElementById('meta');
        const W = 920, H = 420;
        const m = { l: 56, r: 16, t: 16, b: 40 };
        const cw = W - m.l - m.r;
        const ch = H - m.t - m.b;

        function mk(tag, attrs = {}) {
          const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
          for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
          return e;
        }

        if (!points.length) {
          meta.textContent = 'No loss points were recorded.';
        } else {
          const xs = points.map(p => p.step);
          const ys = points.map(p => p.loss);
          const minX = Math.min(...xs), maxX = Math.max(...xs);
          const minY = Math.min(...ys), maxY = Math.max(...ys);

          const x = (v) => m.l + ((v - minX) / Math.max(maxX - minX, 1)) * cw;
          const y = (v) => m.t + (1 - (v - minY) / Math.max(maxY - minY, 1e-9)) * ch;

          svg.appendChild(mk('line', { x1: m.l, y1: m.t + ch, x2: m.l + cw, y2: m.t + ch, class: 'axis' }));
          svg.appendChild(mk('line', { x1: m.l, y1: m.t, x2: m.l, y2: m.t + ch, class: 'axis' }));

          const d = points.map((p, i) => `${i ? 'L' : 'M'} ${x(p.step)} ${y(p.loss)}`).join(' ');
          svg.appendChild(mk('path', { d, class: 'line' }));

          for (const p of points) {
            svg.appendChild(mk('circle', { cx: x(p.step), cy: y(p.loss), r: 2.4, class: 'dot' }));
          }

          const first = points[0];
          const last = points[points.length - 1];
          meta.textContent = `points: ${points.length} | first: step ${first.step}, loss ${first.loss.toFixed(6)} | last: step ${last.step}, loss ${last.loss.toFixed(6)}`;
        }
        </script>
        </body>
        </html>
        """

        try html.write(to: htmlURL, atomically: true, encoding: .utf8)
    }

    public static func loadPoints(from csvURL: URL) throws -> [LoRALossPoint] {
        guard FileManager.default.fileExists(atPath: csvURL.path) else {
            return []
        }
        let text = try String(contentsOf: csvURL, encoding: .utf8)
        var points: [LoRALossPoint] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let row = line.trimmingCharacters(in: .whitespaces)
            if row.isEmpty || row == "step,loss" { continue }
            let columns = row.split(separator: ",", maxSplits: 1, omittingEmptySubsequences: false)
            guard columns.count == 2,
                  let step = Int(columns[0]),
                  let loss = Float(columns[1]) else {
                continue
            }
            points.append(LoRALossPoint(step: step, loss: loss))
        }
        return points
    }
}

public enum LoRATrainingResumeArtifacts {
    public static func restore(
        from resumeCheckpointURL: URL,
        sidecar: LoRATrainingCheckpointState?,
        lossStateURL: URL? = nil,
        to outputBaseURL: URL
    ) {
        let fileManager = FileManager.default
        let sourceDirectory = resumeCheckpointURL.deletingLastPathComponent()
        let outputDirectory = outputBaseURL.deletingLastPathComponent()
        let runManifest = loadRunManifest(from: sourceDirectory)
        let outputLossCSV = outputBaseURL
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("csv")
        let outputLossHTML = outputBaseURL
            .deletingPathExtension()
            .appendingPathExtension("loss")
            .appendingPathExtension("html")

        copyNamedArtifact(
            runManifest?.checkpointFiles["loss_csv"] ?? sidecar?.lossCSVFile,
            from: sourceDirectory,
            to: outputLossCSV,
            fileManager: fileManager
        )
        copyNamedArtifact(
            runManifest?.checkpointFiles["loss_html"] ?? sidecar?.lossHTMLFile,
            from: sourceDirectory,
            to: outputLossHTML,
            fileManager: fileManager
        )

        let checkpointManifestFiles = loadCheckpointManifestFiles(from: sourceDirectory)
        let resolvedLossStateURL = mfluxLossStateURL(
            explicit: lossStateURL,
            runManifest: runManifest,
            checkpointManifestFiles: checkpointManifestFiles,
            sourceDirectory: sourceDirectory
        )
        if !fileManager.fileExists(atPath: outputLossCSV.path),
           let resolvedLossStateURL {
            importLossJSONIfPossible(
                from: resolvedLossStateURL,
                to: outputLossCSV
            )
        }

        // Older checkpoints may not carry sidecar loss filenames.
        if !fileManager.fileExists(atPath: outputLossCSV.path),
           let fallbackCSV = firstArtifact(in: sourceDirectory, suffix: ".loss.csv", fileManager: fileManager) {
            copyArtifact(fallbackCSV, to: outputLossCSV, fileManager: fileManager)
        }
        if !fileManager.fileExists(atPath: outputLossHTML.path),
           let fallbackHTML = firstArtifact(in: sourceDirectory, suffix: ".loss.html", fileManager: fileManager) {
            copyArtifact(fallbackHTML, to: outputLossHTML, fileManager: fileManager)
        }

        let sampleArtifacts = collectSampleArtifacts(
            from: sourceDirectory,
            sidecar: sidecar,
            runManifest: runManifest,
            fileManager: fileManager
        )
        guard !sampleArtifacts.isEmpty else { return }

        let outputSamplesDirectory = outputDirectory.appendingPathComponent("samples", isDirectory: true)
        try? fileManager.createDirectory(at: outputSamplesDirectory, withIntermediateDirectories: true)
        for sample in sampleArtifacts {
            let destination = outputSamplesDirectory.appendingPathComponent(sample.lastPathComponent, isDirectory: false)
            copyArtifact(sample, to: destination, fileManager: fileManager)
        }
    }

    private static func loadRunManifest(from sourceDirectory: URL) -> LoRATrainingRunManifest? {
        let url = sourceDirectory.appendingPathComponent(LoRATrainingRunManifest.filename, isDirectory: false)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? LoRATrainingRunManifest.decode(from: data)
    }

    private static func loadCheckpointManifestFiles(from sourceDirectory: URL) -> [String: String] {
        let checkpointManifestURL = sourceDirectory.appendingPathComponent("checkpoint.json", isDirectory: false)
        guard let data = try? Data(contentsOf: checkpointManifestURL),
              let manifest = try? JSONDecoder().decode(MFluxCheckpointManifest.self, from: data) else {
            return [:]
        }

        var normalized: [String: String] = [:]
        for (key, value) in manifest.files {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            normalized[key] = trimmed
        }
        return normalized
    }

    private static func mfluxLossStateURL(
        explicit: URL?,
        runManifest: LoRATrainingRunManifest?,
        checkpointManifestFiles: [String: String],
        sourceDirectory: URL
    ) -> URL? {
        if let explicit {
            return explicit
        }

        if let runManifestLoss = runManifest?.checkpointFiles["loss"] {
            let fileName = (runManifestLoss as NSString).lastPathComponent
            if !fileName.isEmpty {
                return sourceDirectory.appendingPathComponent(fileName, isDirectory: false)
            }
        }

        if let checkpointLoss = checkpointManifestFiles["loss"] {
            let fileName = (checkpointLoss as NSString).lastPathComponent
            if !fileName.isEmpty {
                return sourceDirectory.appendingPathComponent(fileName, isDirectory: false)
            }
        }

        return nil
    }

    private static func importLossJSONIfPossible(from sourceURL: URL, to destinationCSVURL: URL) {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else { return }
        guard let data = try? Data(contentsOf: sourceURL),
              let rawEntries = try? JSONDecoder().decode([LossEntryCandidate].self, from: data) else {
            return
        }

        var parsedEntries: [MFluxLossEntry] = []
        parsedEntries.reserveCapacity(rawEntries.count)

        for candidate in rawEntries {
            guard let entry = candidate.entry else { continue }
            parsedEntries.append(entry)
        }

        guard !parsedEntries.isEmpty else { return }
        parsedEntries.sort { lhs, rhs in lhs.step < rhs.step }

        var csv = "step,loss\n"
        for entry in parsedEntries {
            csv += "\(entry.step),\(entry.loss)\n"
        }

        do {
            try FileManager.default.createDirectory(
                at: destinationCSVURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try csv.write(to: destinationCSVURL, atomically: true, encoding: .utf8)
        } catch {
            // Best-effort import; training can continue with a fresh log.
        }
    }

    private struct LossEntryCandidate: Decodable {
        let entry: MFluxLossEntry?

        init(from decoder: Decoder) throws {
            guard let container = try? decoder.container(keyedBy: CodingKeys.self),
                  let step = try? container.decode(LenientInt.self, forKey: .step).value,
                  let loss = try? container.decode(LenientDouble.self, forKey: .loss).value else {
                self.entry = nil
                return
            }
            self.entry = MFluxLossEntry(step: step, loss: loss)
        }

        private enum CodingKeys: String, CodingKey {
            case step
            case loss
        }
    }

    private static func copyNamedArtifact(
        _ fileName: String?,
        from sourceDirectory: URL,
        to destinationURL: URL,
        fileManager: FileManager
    ) {
        guard let fileName else { return }
        let sanitized = (fileName as NSString).lastPathComponent
        guard !sanitized.isEmpty else { return }
        let source = sourceDirectory.appendingPathComponent(sanitized, isDirectory: false)
        copyArtifact(source, to: destinationURL, fileManager: fileManager)
    }

    private static func copyArtifact(
        _ sourceURL: URL,
        to destinationURL: URL,
        fileManager: FileManager
    ) {
        guard fileManager.fileExists(atPath: sourceURL.path) else { return }

        let source = sourceURL.standardizedFileURL
        let destination = destinationURL.standardizedFileURL
        guard source != destination else { return }

        do {
            try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.copyItem(at: source, to: destination)
        } catch {
            // Best-effort restoration; training can continue without restored artifacts.
        }
    }

    private static func firstArtifact(
        in directory: URL,
        suffix: String,
        fileManager: FileManager
    ) -> URL? {
        guard let entries = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return entries
            .filter { entry in
                guard (try? entry.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
                    return false
                }
                return entry.lastPathComponent.hasSuffix(suffix)
            }
            .sorted { lhs, rhs in lhs.lastPathComponent < rhs.lastPathComponent }
            .first
    }

    private static func collectSampleArtifacts(
        from sourceDirectory: URL,
        sidecar: LoRATrainingCheckpointState?,
        runManifest: LoRATrainingRunManifest?,
        fileManager: FileManager
    ) -> [URL] {
        var dedupedByName: [String: URL] = [:]

        func add(_ candidate: URL) {
            let fileName = candidate.lastPathComponent
            guard !fileName.isEmpty else { return }
            guard dedupedByName[fileName] == nil else { return }
            guard fileManager.fileExists(atPath: candidate.path) else { return }
            guard (try? candidate.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return }
            dedupedByName[fileName] = candidate
        }

        if let rawSampleFiles = runManifest?.checkpointFiles["sample_files"] {
            for rawName in rawSampleFiles.split(separator: ",") {
                let trimmed = String(rawName).trimmingCharacters(in: .whitespacesAndNewlines)
                let fileName = (trimmed as NSString).lastPathComponent
                guard !fileName.isEmpty else { continue }
                add(sourceDirectory.appendingPathComponent(fileName, isDirectory: false))
            }
        }

        if let sidecar,
           let manifestFile = sidecar.manifestFile {
            let manifestName = (manifestFile as NSString).lastPathComponent
            if !manifestName.isEmpty {
                let manifestURL = sourceDirectory.appendingPathComponent(manifestName, isDirectory: false)
                if let data = try? Data(contentsOf: manifestURL),
                   let manifest = try? JSONDecoder().decode(LoRATrainingManifest.self, from: data),
                   let rawSampleFiles = manifest.extras?["sample_files"] {
                    for rawName in rawSampleFiles.split(separator: ",") {
                        let trimmed = String(rawName).trimmingCharacters(in: .whitespacesAndNewlines)
                        let fileName = (trimmed as NSString).lastPathComponent
                        guard !fileName.isEmpty else { continue }
                        add(sourceDirectory.appendingPathComponent(fileName, isDirectory: false))
                    }
                }
            }
        }

        let sourceSamplesDirectory = sourceDirectory.appendingPathComponent("samples", isDirectory: true)
        if let entries = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: sourceSamplesDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries {
                add(entry)
            }
        }

        // Zip bundles flatten files, so sample previews may sit beside the checkpoint.
        if let sourceEntries = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: sourceDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) {
            let imageExtensions = Set(["png", "jpg", "jpeg", "webp"])
            for entry in sourceEntries {
                let fileName = entry.lastPathComponent.lowercased()
                guard fileName.hasPrefix("step-") else { continue }
                let ext = entry.pathExtension.lowercased()
                guard imageExtensions.contains(ext) else { continue }
                add(entry)
            }
        }

        return dedupedByName.keys.sorted().compactMap { dedupedByName[$0] }
    }
}
