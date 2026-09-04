import Combine
import Sparkle
import StudioKit
import StudioUI
import SwiftUI

enum MereRunUpdateConfiguration {
    static let feedURL = URL(string: "https://mere.run/releases/appcast.xml")!
    static let publicEDKey = "6sFs+7UqYcE7rThPAovzMDsZtKyf/h4/d8rUmPSH2rw="
}

@MainActor
final class MereRunUpdateCheckViewModel: ObservableObject {
    @Published private(set) var canCheckForUpdates = false

    init(updater: SPUUpdater) {
        updater.publisher(for: \.canCheckForUpdates)
            .receive(on: RunLoop.main)
            .assign(to: &$canCheckForUpdates)
    }
}

struct MereRunCheckForUpdatesView: View {
    @ObservedObject private var viewModel: MereRunUpdateCheckViewModel
    private let updater: SPUUpdater

    @MainActor
    init(updater: SPUUpdater) {
        self.updater = updater
        viewModel = MereRunUpdateCheckViewModel(updater: updater)
    }

    var body: some View {
        Button("Check for Updates…", action: updater.checkForUpdates)
            .disabled(!viewModel.canCheckForUpdates)
    }
}
