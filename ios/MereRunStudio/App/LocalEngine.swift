import Foundation
import MereRunCore
import UIKit

/// The on-device lane: FLUX.2 Klein nano running natively on this iPhone
/// through the memory-optimized iOS generator. The model installs into the
/// app sandbox through the same managed store, catalog pins, and verified
/// snapshots as every other mere.run install. Experimental: first-run
/// generation exercises MLX Metal on the device.
@MainActor
final class LocalEngine: ObservableObject {
    static let modelID = "image-klein-nano"

    enum State: Equatable {
        case checking
        case notInstalled
        case downloading(String)
        case ready
        case generating
        case failed(String)
    }

    @Published private(set) var state: State = .checking
    @Published private(set) var lastImage: UIImage?

    private let generator = Flux2KleinGeneratoriOS()

    init() {
        refresh()
    }

    func refresh() {
        state = ManagedModelResolver.resolveInstalledModel(id: Self.modelID) != nil
            ? .ready
            : .notInstalled
    }

    func download() async {
        state = .downloading("Starting…")
        do {
            _ = try await ManagedModelResolver.installManagedModel(
                id: Self.modelID,
                progress: { [weak self] progress in
                    let label: String
                    switch progress {
                    case .downloadingBytes(let completed, let total):
                        if let total, total > 0 {
                            label = "\(Int(Double(completed) / Double(total) * 100))% of \(total / 1_048_576) MB"
                        } else {
                            label = "\(completed / 1_048_576) MB"
                        }
                    case .downloadingPercent(let percent, _):
                        label = "\(percent)%"
                    case .extracting:
                        label = "Preparing…"
                    }
                    Task { @MainActor [weak self] in
                        self?.state = .downloading(label)
                    }
                }
            )
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func generate(prompt: String, width: Int = 512, height: Int = 512, steps: Int = 4) async {
        guard state == .ready else { return }
        state = .generating
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("local-\(UUID().uuidString).png")
        let request = GenerationRequest(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            outputURL: output,
            model: Self.modelID
        )
        do {
            let result = try await generator.generate(request, progressHandler: nil)
            lastImage = UIImage(contentsOfFile: result.outputURL.path)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}
