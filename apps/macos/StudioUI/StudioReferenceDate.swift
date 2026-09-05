import Foundation
import SwiftUI

private struct StudioReferenceDateKey: EnvironmentKey {
    static let defaultValue: Date? = nil
}

extension EnvironmentValues {
    package var studioReferenceDate: Date? {
        get { self[StudioReferenceDateKey.self] }
        set { self[StudioReferenceDateKey.self] = newValue }
    }
}

/// Names generated while rendering use the same clock as the displayed dates.
@MainActor
package enum StudioDisplayClock {
    package static var fixedDate: Date?
    package static var now: Date { fixedDate ?? Date() }
}
