import Foundation
import Hummingbird
import MereRunCore
import NIOCore

struct LoRATrainingRunViewer {
    let runDirectoryURL: URL

    init(runDirectoryURL: URL) {
        self.runDirectoryURL = runDirectoryURL.standardizedFileURL
    }

    func run(host: String, port: Int) async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )

        CLIStderr.write("[visualize] LoRA training viewer: http://\(host):\(port)\n")
        CLIStderr.write("[visualize] Run directory: \(runDirectoryURL.path)\n")
        CLIStderr.write("[visualize] Press Ctrl+C to stop.\n")
        try await app.runService()
    }

    nonisolated func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        router.get("/") { _, _ in
            htmlResponse(Self.dashboardHTML)
        }

        router.get("/health") { _, _ in
            try jsonResponse(["status": "ok"])
        }

        router.get("/api/run") { [self] _, _ in
            do {
                return try jsonResponse(snapshot())
            } catch {
                return errorResponse(message: error.localizedDescription)
            }
        }

        router.get("/api/events") { [self] _, _ in
            do {
                return try jsonResponse(loadEvents())
            } catch {
                return errorResponse(message: error.localizedDescription)
            }
        }

        router.get("/api/loss.csv") { [self] _, _ in
            do {
                guard let url = firstLossCSVURL() else {
                    return Response(status: .notFound)
                }
                let data = try Data(contentsOf: url)
                return binaryResponse(data, contentType: "text/csv; charset=utf-8")
            } catch {
                return errorResponse(message: error.localizedDescription)
            }
        }

        router.get("/api/artifacts/:kind/:name") { [self] _, context in
            do {
                guard let kind = context.parameters.get("kind", as: String.self),
                      let rawName = context.parameters.get("name", as: String.self),
                      let url = artifactURL(kind: kind, name: rawName) else {
                    return Response(status: .notFound)
                }
                let data = try Data(contentsOf: url)
                return binaryResponse(data, contentType: contentType(for: url))
            } catch {
                return errorResponse(message: error.localizedDescription)
            }
        }

        return router
    }

    func snapshot() throws -> LoRATrainingRunViewerSnapshot {
        let runManifest = try loadRunManifest()
        let events = try loadEvents()
        let lossPoints = try loadLossPoints()
        let artifacts = try listArtifacts()
        let latestEvent = events.last
        let status: String = {
            if latestEvent?.type == "run_failed" { return "failed" }
            if latestEvent?.type == "run_finished" { return "finished" }
            if let runManifest, runManifest.step >= runManifest.totalSteps { return "finished" }
            if !events.isEmpty { return "running" }
            return "idle"
        }()

        return LoRATrainingRunViewerSnapshot(
            runDirectory: runDirectoryURL.path,
            status: status,
            runManifest: runManifest,
            lossPoints: lossPoints,
            events: events,
            artifacts: artifacts
        )
    }

    private func loadRunManifest() throws -> LoRATrainingRunManifest? {
        let url = runDirectoryURL.appendingPathComponent(LoRATrainingRunManifest.filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        return try LoRATrainingRunManifest.decode(from: data)
    }

    private func loadEvents() throws -> [LoRATrainingRunEvent] {
        let urls = try files(in: runDirectoryURL)
            .filter { $0.lastPathComponent.hasSuffix(LoRATrainingRunEvent.filenameSuffix) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var events: [LoRATrainingRunEvent] = []
        for url in urls {
            events.append(contentsOf: try LoRATrainingRunEvent.load(from: url))
        }
        return events.sorted { lhs, rhs in
            if lhs.createdAt == rhs.createdAt {
                return lhs.sequence < rhs.sequence
            }
            return lhs.createdAt < rhs.createdAt
        }
    }

    private func loadLossPoints() throws -> [LoRALossPoint] {
        guard let url = firstLossCSVURL() else {
            return []
        }
        return try LoRATrainingMetricsLogger.loadPoints(from: url)
    }

    private func firstLossCSVURL() -> URL? {
        (try? files(in: runDirectoryURL))
            .flatMap { urls in
                urls
                    .filter { $0.lastPathComponent.hasSuffix(".loss.csv") }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
                    .first
            }
    }

    private func listArtifacts() throws -> [LoRATrainingRunArtifact] {
        var artifacts: [LoRATrainingRunArtifact] = []
        artifacts.append(contentsOf: try listArtifacts(in: runDirectoryURL, kind: "root"))

        let sampleDirectory = runDirectoryURL.appendingPathComponent("samples", isDirectory: true)
        if isDirectory(sampleDirectory) {
            artifacts.append(contentsOf: try listArtifacts(in: sampleDirectory, kind: "samples"))
        }

        let checkpointDirectory = runDirectoryURL.appendingPathComponent("checkpoints", isDirectory: true)
        if isDirectory(checkpointDirectory) {
            artifacts.append(contentsOf: try listArtifacts(in: checkpointDirectory, kind: "checkpoints"))
        }

        return artifacts.sorted {
            if $0.kind == $1.kind {
                return $0.name < $1.name
            }
            return $0.kind < $1.kind
        }
    }

    private func listArtifacts(in directory: URL, kind: String) throws -> [LoRATrainingRunArtifact] {
        try files(in: directory).compactMap { url in
            let name = url.lastPathComponent
            guard isAllowedArtifact(url),
                  let encodedName = name.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else {
                return nil
            }
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attributes?[.size] as? NSNumber)?.int64Value
            return LoRATrainingRunArtifact(
                kind: kind,
                name: name,
                path: url.path,
                sizeBytes: size,
                contentType: contentType(for: url),
                url: "/api/artifacts/\(kind)/\(encodedName)",
                isImage: isImageArtifact(url)
            )
        }
    }

    private func artifactURL(kind: String, name rawName: String) -> URL? {
        guard ["root", "samples", "checkpoints"].contains(kind) else { return nil }
        let name = rawName.removingPercentEncoding ?? rawName
        guard !name.isEmpty,
              !name.contains("/"),
              !name.contains("\\"),
              name != ".",
              name != "..",
              !name.contains("..") else {
            return nil
        }

        let directory: URL = switch kind {
        case "samples":
            runDirectoryURL.appendingPathComponent("samples", isDirectory: true)
        case "checkpoints":
            runDirectoryURL.appendingPathComponent("checkpoints", isDirectory: true)
        default:
            runDirectoryURL
        }
        let candidate = directory.appendingPathComponent(name, isDirectory: false).standardizedFileURL
        guard candidate.deletingLastPathComponent().path == directory.standardizedFileURL.path,
              FileManager.default.fileExists(atPath: candidate.path),
              isAllowedArtifact(candidate) else {
            return nil
        }
        return candidate
    }

    private func files(in directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            return values?.isRegularFile == true
        }
    }

    private func isDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    private func isAllowedArtifact(_ url: URL) -> Bool {
        [
            "csv",
            "html",
            "json",
            "jsonl",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "safetensors",
            "zip",
        ].contains(url.pathExtension.lowercased())
    }

    private func isImageArtifact(_ url: URL) -> Bool {
        ["png", "jpg", "jpeg", "webp"].contains(url.pathExtension.lowercased())
    }

    private func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "csv":
            return "text/csv; charset=utf-8"
        case "html":
            return "text/html; charset=utf-8"
        case "json", "jsonl":
            return "application/json"
        case "png":
            return "image/png"
        case "jpg", "jpeg":
            return "image/jpeg"
        case "webp":
            return "image/webp"
        case "zip":
            return "application/zip"
        default:
            return "application/octet-stream"
        }
    }

    private static let dashboardHTML = #"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>mere.run LoRA Training</title>
      <style>
        :root {
          color-scheme: light;
          --bg: #f6f7f8;
          --panel: #ffffff;
          --text: #121417;
          --muted: #626b77;
          --line: #d9dee5;
          --green: #1c7c54;
          --red: #bf3b3b;
          --blue: #2f6fbd;
          --gold: #9a6a16;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          background: var(--bg);
          color: var(--text);
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
        }
        header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          padding: 18px 24px;
          border-bottom: 1px solid var(--line);
          background: rgba(255, 255, 255, 0.92);
          position: sticky;
          top: 0;
          z-index: 2;
        }
        h1 { font-size: 18px; line-height: 1.2; margin: 0; font-weight: 650; }
        .status {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-size: 13px;
          color: var(--muted);
        }
        .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--gold); }
        .dot.finished { background: var(--green); }
        .dot.failed { background: var(--red); }
        .dot.running { background: var(--blue); }
        main {
          display: grid;
          grid-template-columns: minmax(0, 1.7fr) minmax(320px, 0.85fr);
          gap: 16px;
          padding: 16px 24px 28px;
        }
        section {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          overflow: hidden;
        }
        .section-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 12px 14px;
          border-bottom: 1px solid var(--line);
        }
        h2 { margin: 0; font-size: 13px; font-weight: 650; color: #20242a; }
        .small { font-size: 12px; color: var(--muted); }
        .metrics {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          padding: 14px;
        }
        .metric {
          border: 1px solid var(--line);
          border-radius: 7px;
          padding: 10px;
          min-width: 0;
          background: #fbfcfd;
        }
        .label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
        .value { font-size: 17px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        canvas { display: block; width: 100%; height: 360px; padding: 10px 14px 14px; }
        .samples {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 10px;
          padding: 14px;
        }
        figure { margin: 0; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: #f0f2f4; }
        figure img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; }
        figcaption { padding: 7px 8px; font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .list { padding: 8px 14px 14px; }
        .row {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto;
          gap: 10px;
          align-items: center;
          padding: 8px 0;
          border-bottom: 1px solid #edf0f3;
          font-size: 13px;
        }
        .row:last-child { border-bottom: 0; }
        a { color: #1f5f99; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .event { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #2b3139; }
        .muted { color: var(--muted); }
        @media (max-width: 980px) {
          main { grid-template-columns: 1fr; padding: 12px; }
          header { padding: 14px 12px; }
          .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
      </style>
    </head>
    <body>
      <header>
        <h1>mere.run LoRA Training</h1>
        <div class="status"><span id="status-dot" class="dot"></span><span id="status-text">loading</span></div>
      </header>
      <main>
        <div>
          <section>
            <div class="section-head">
              <h2>Run</h2>
              <span id="run-path" class="small"></span>
            </div>
            <div class="metrics">
              <div class="metric"><span class="label">step</span><span id="metric-step" class="value">-</span></div>
              <div class="metric"><span class="label">loss</span><span id="metric-loss" class="value">-</span></div>
              <div class="metric"><span class="label">progress</span><span id="metric-progress" class="value">-</span></div>
              <div class="metric"><span class="label">samples</span><span id="metric-samples" class="value">-</span></div>
            </div>
          </section>
          <section style="margin-top:16px">
            <div class="section-head">
              <h2>Loss</h2>
              <a class="small" href="/api/loss.csv">loss.csv</a>
            </div>
            <canvas id="loss-chart" width="1100" height="420"></canvas>
          </section>
          <section style="margin-top:16px">
            <div class="section-head">
              <h2>Samples</h2>
              <span id="sample-count" class="small"></span>
            </div>
            <div id="samples" class="samples"></div>
          </section>
        </div>
        <div>
          <section>
            <div class="section-head">
              <h2>Artifacts</h2>
              <span id="artifact-count" class="small"></span>
            </div>
            <div id="artifacts" class="list"></div>
          </section>
          <section style="margin-top:16px">
            <div class="section-head">
              <h2>Events</h2>
              <span id="event-count" class="small"></span>
            </div>
            <div id="events" class="list"></div>
          </section>
        </div>
      </main>
      <script>
        const els = {
          dot: document.getElementById('status-dot'),
          status: document.getElementById('status-text'),
          path: document.getElementById('run-path'),
          step: document.getElementById('metric-step'),
          loss: document.getElementById('metric-loss'),
          progress: document.getElementById('metric-progress'),
          samplesMetric: document.getElementById('metric-samples'),
          sampleCount: document.getElementById('sample-count'),
          samples: document.getElementById('samples'),
          artifactCount: document.getElementById('artifact-count'),
          artifacts: document.getElementById('artifacts'),
          eventCount: document.getElementById('event-count'),
          events: document.getElementById('events'),
          chart: document.getElementById('loss-chart')
        };

        function fmtBytes(n) {
          if (!n && n !== 0) return '';
          const units = ['B', 'KB', 'MB', 'GB'];
          let size = n, i = 0;
          while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
          return `${size.toFixed(i ? 1 : 0)} ${units[i]}`;
        }

        function latestTraining(events) {
          for (let i = events.length - 1; i >= 0; i--) {
            if (events[i].type === 'progress' && events[i].stage === 'training') return events[i];
          }
          return null;
        }

        function drawLoss(points) {
          const canvas = els.chart;
          const ctx = canvas.getContext('2d');
          const w = canvas.width, h = canvas.height;
          ctx.clearRect(0, 0, w, h);
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, w, h);
          ctx.strokeStyle = '#d9dee5';
          ctx.lineWidth = 1;
          const m = { l: 62, r: 20, t: 22, b: 46 };
          ctx.beginPath();
          ctx.moveTo(m.l, m.t);
          ctx.lineTo(m.l, h - m.b);
          ctx.lineTo(w - m.r, h - m.b);
          ctx.stroke();
          if (!points.length) {
            ctx.fillStyle = '#626b77';
            ctx.font = '18px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.fillText('Waiting for loss points', m.l + 16, m.t + 42);
            return;
          }
          const xs = points.map(p => p.step);
          const ys = points.map(p => p.loss);
          const minX = Math.min(...xs), maxX = Math.max(...xs);
          const minY = Math.min(...ys), maxY = Math.max(...ys);
          const x = v => m.l + ((v - minX) / Math.max(maxX - minX, 1)) * (w - m.l - m.r);
          const y = v => m.t + (1 - ((v - minY) / Math.max(maxY - minY, 1e-9))) * (h - m.t - m.b);
          ctx.strokeStyle = '#1c7c54';
          ctx.lineWidth = 3;
          ctx.beginPath();
          points.forEach((p, i) => {
            const px = x(p.step), py = y(p.loss);
            if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
          });
          ctx.stroke();
          ctx.fillStyle = '#121417';
          ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
          ctx.fillText(`loss ${minY.toFixed(4)} - ${maxY.toFixed(4)}`, m.l, 18);
          ctx.fillText(`step ${minX} - ${maxX}`, m.l, h - 14);
        }

        function render(snapshot) {
          const events = snapshot.events || [];
          const loss = snapshot.loss_points || [];
          const artifacts = snapshot.artifacts || [];
          const images = artifacts.filter(a => a.kind === 'samples' && a.is_image).slice(-12).reverse();
          const latest = latestTraining(events);
          const manifest = snapshot.run_manifest;
          const total = latest?.total_steps || manifest?.total_steps || 0;
          const step = latest?.step || manifest?.step || 0;
          const lastLoss = loss.length ? loss[loss.length - 1].loss : latest?.loss;

          els.dot.className = `dot ${snapshot.status}`;
          els.status.textContent = snapshot.status;
          els.path.textContent = snapshot.run_directory || '';
          els.step.textContent = total ? `${step} / ${total}` : (step || '-');
          els.loss.textContent = Number.isFinite(lastLoss) ? lastLoss.toFixed(6) : '-';
          els.progress.textContent = total ? `${Math.round((step / total) * 100)}%` : '-';
          els.samplesMetric.textContent = String(images.length);
          els.sampleCount.textContent = `${images.length} shown`;
          els.artifactCount.textContent = `${artifacts.length} files`;
          els.eventCount.textContent = `${events.length} events`;

          drawLoss(loss);

          els.samples.innerHTML = images.map(a => `
            <figure>
              <a href="${a.url}" target="_blank" rel="noreferrer"><img src="${a.url}" alt="${a.name}"></a>
              <figcaption>${a.name}</figcaption>
            </figure>
          `).join('') || '<div class="small">No sample images yet.</div>';

          els.artifacts.innerHTML = artifacts.map(a => `
            <div class="row">
              <a href="${a.url}" target="_blank" rel="noreferrer">${a.kind}/${a.name}</a>
              <span class="small">${fmtBytes(a.size_bytes)}</span>
            </div>
          `).join('') || '<div class="small">No artifacts yet.</div>';

          els.events.innerHTML = events.slice(-30).reverse().map(e => `
            <div class="row event">
              <span><span class="muted">#${e.sequence}</span> ${e.type}${e.stage ? `:${e.stage}` : ''}${e.step ? ` step ${e.step}` : ''}${e.loss ? ` loss ${Number(e.loss).toFixed(6)}` : ''}</span>
              <span class="muted">${new Date(e.created_at).toLocaleTimeString()}</span>
            </div>
          `).join('') || '<div class="small">No events yet.</div>';
        }

        async function refresh() {
          try {
            const response = await fetch('/api/run', { cache: 'no-store' });
            render(await response.json());
          } catch (error) {
            els.status.textContent = 'viewer error';
            els.dot.className = 'dot failed';
          }
        }

        refresh();
        setInterval(refresh, 2000);
      </script>
    </body>
    </html>
    """#
}

struct LoRATrainingRunViewerSnapshot: Encodable {
    let runDirectory: String
    let status: String
    let runManifest: LoRATrainingRunManifest?
    let lossPoints: [LoRALossPoint]
    let events: [LoRATrainingRunEvent]
    let artifacts: [LoRATrainingRunArtifact]

    enum CodingKeys: String, CodingKey {
        case runDirectory = "run_directory"
        case status
        case runManifest = "run_manifest"
        case lossPoints = "loss_points"
        case events
        case artifacts
    }
}

struct LoRATrainingRunArtifact: Encodable, Equatable {
    let kind: String
    let name: String
    let path: String
    let sizeBytes: Int64?
    let contentType: String
    let url: String
    let isImage: Bool

    enum CodingKeys: String, CodingKey {
        case kind
        case name
        case path
        case sizeBytes = "size_bytes"
        case contentType = "content_type"
        case url
        case isImage = "is_image"
    }
}

private func jsonResponse<T: Encodable>(_ payload: T, status: HTTPResponse.Status = .ok) throws -> Response {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(payload)
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(bytes: data))
    )
}

private func htmlResponse(_ html: String) -> Response {
    Response(
        status: .ok,
        headers: [.contentType: "text/html; charset=utf-8"],
        body: .init(byteBuffer: ByteBuffer(string: html))
    )
}

private func binaryResponse(_ data: Data, contentType: String) -> Response {
    Response(
        status: .ok,
        headers: [.contentType: contentType],
        body: .init(byteBuffer: ByteBuffer(bytes: data))
    )
}

private func errorResponse(message: String) -> Response {
    let payload = ["error": message]
    let data = (try? JSONEncoder().encode(payload)) ?? Data("{\"error\":\"viewer error\"}".utf8)
    return Response(
        status: .internalServerError,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(bytes: data))
    )
}
