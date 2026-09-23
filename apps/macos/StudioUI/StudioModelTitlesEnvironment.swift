import StudioKit
import SwiftUI

private struct StudioModelTitlesKey: EnvironmentKey {
    static let defaultValue = StudioModelTitles.none
}

extension EnvironmentValues {
    /// The model inventory's titles, as `StudioModelStore` last published them. Each window root
    /// sets it from the store it observes, so every name below re-renders when titles arrive;
    /// a view that has no root above it (a preview, a bare render) names models from their ids.
    var studioModelTitles: StudioModelTitles {
        get { self[StudioModelTitlesKey.self] }
        set { self[StudioModelTitlesKey.self] = newValue }
    }
}

/// Puts the store's titles in the environment for everything below it. Roots use this rather
/// than reading `controller.modelStore.titles` themselves, because only an `@ObservedObject`
/// re-renders when the store publishes.
struct StudioModelTitlesScope<Content: View>: View {
    @ObservedObject var store: StudioModelStore
    @ViewBuilder let content: () -> Content

    var body: some View {
        content().environment(\.studioModelTitles, store.titles)
    }
}
