import StudioKit
import SwiftUI

private struct StudioTaskSessionsKey: EnvironmentKey {
    static let defaultValue: StudioTaskSessions? = nil
}

private struct StudioTaskScopeKey: EnvironmentKey {
    static let defaultValue = "studio"
}

extension EnvironmentValues {
    var studioTaskSessions: StudioTaskSessions? {
        get { self[StudioTaskSessionsKey.self] }
        set { self[StudioTaskSessionsKey.self] = newValue }
    }
    var studioTaskScope: String {
        get { self[StudioTaskScopeKey.self] }
        set { self[StudioTaskScopeKey.self] = newValue }
    }
}

/// A task-scoped value whose owner survives view replacement. Transient UI state stays in @State.
@MainActor
@propertyWrapper
struct StudioStoredValue<Value: Codable>: DynamicProperty {
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioTaskScope) private var scope
    private let field: String
    @State private var initial: Value

    init(wrappedValue: Value, _ field: String) {
        self.field = field
        _initial = State(initialValue: wrappedValue)
    }

    init(initialValue: Value, _ field: String) {
        self.init(wrappedValue: initialValue, field)
    }

    var wrappedValue: Value {
        get { sessions?.value(for: scope + "." + field, default: initial) ?? initial }
        nonmutating set {
            if let sessions { sessions.set(newValue, for: scope + "." + field) }
            else { initial = newValue }
        }
    }

    var projectedValue: Binding<Value> {
        Binding(get: { wrappedValue }, set: { wrappedValue = $0 })
    }
}
