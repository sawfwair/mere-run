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

/// Sparkle's "check automatically" setting, observed so Settings and the updater's own first-run
/// prompt agree.
@MainActor
final class MereRunAutomaticUpdateChecks: ObservableObject {
    @Published private(set) var isEnabled: Bool
    private let updater: SPUUpdater

    init(updater: SPUUpdater) {
        self.updater = updater
        isEnabled = updater.automaticallyChecksForUpdates
        updater.publisher(for: \.automaticallyChecksForUpdates)
            .receive(on: RunLoop.main)
            .assign(to: &$isEnabled)
    }

    var binding: Binding<Bool> {
        Binding(
            get: { self.isEnabled },
            set: { isOn in
                self.updater.automaticallyChecksForUpdates = isOn
                self.isEnabled = isOn
            }
        )
    }
}

/// The Settings window with General's update toggle bound to Sparkle. Settings lives in
/// `StudioUI`, which does not link Sparkle, so the binding is handed in here.
struct MereRunSettingsWindow: View {
    @StateObject private var updateChecks: MereRunAutomaticUpdateChecks

    init(updater: SPUUpdater) {
        _updateChecks = StateObject(wrappedValue: MereRunAutomaticUpdateChecks(updater: updater))
    }

    var body: some View {
        MereRunSettingsView(automaticUpdateChecks: updateChecks.binding)
    }
}
