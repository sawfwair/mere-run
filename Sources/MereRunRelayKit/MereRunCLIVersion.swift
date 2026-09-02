/// The mere.run distribution version, shared by the CLI and app shells for
/// contract version floors, worker probes, and the built-in node catalog
/// identity.
public enum MereRunCLIVersion {
    public static let current = "0.50.0"
}

/// Lenient three-component version-floor comparison used by worker
/// validation and contract gates.
public func workflowVersion(_ current: String, satisfiesMinimum minimum: String) -> Bool {
    func components(_ value: String) -> [Int] {
        value
            .split(separator: ".")
            .prefix(3)
            .map { component in
                Int(component.prefix(while: \Character.isNumber)) ?? 0
            }
    }
    let currentParts = components(current)
    let minimumParts = components(minimum)
    for index in 0..<max(currentParts.count, minimumParts.count) {
        let lhs = index < currentParts.count ? currentParts[index] : 0
        let rhs = index < minimumParts.count ? minimumParts[index] : 0
        if lhs != rhs { return lhs > rhs }
    }
    return true
}
