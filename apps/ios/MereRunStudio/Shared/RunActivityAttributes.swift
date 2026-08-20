#if canImport(ActivityKit)
import ActivityKit
import SwiftUI

/// Shared between the app (which starts and updates activities) and the
/// widget extension (which renders them).
struct RunActivityAttributes: ActivityAttributes {
    struct ContentState: Codable, Hashable {
        var stateLabel: String
        var fraction: Double?
        var detail: String?
    }

    var jobID: String
    var title: String
}

/// The bronze accent, duplicated here because the widget extension does not
/// compile the app's theme. Keep in sync with MereTheme.accent.
enum RunActivityPalette {
    static var accent: Color {
        Color(red: 0x9C / 255, green: 0x7A / 255, blue: 0x2E / 255)
    }
}
#endif
