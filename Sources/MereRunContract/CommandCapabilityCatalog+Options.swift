import Foundation

extension MereRunCapabilityCatalog {
    typealias Group = MereRunCapabilityOptionGroup

    /// The machine-readable flags shared by every long-running generation command.
    /// `--receipt` prints the final `{"event":"result",...}` line; `--progress-json`
    /// streams `{"event":"progress",...}` lines on stderr.
    static let receiptOption = MereRunCapabilityOption(
        flag: "--receipt",
        label: "Result receipt",
        kind: .boolean,
        group: Group.run,
        tier: .expert
    )

    static let progressJSONOption = MereRunCapabilityOption(
        flag: "--progress-json",
        label: "Progress JSON",
        kind: .boolean,
        group: Group.run,
        tier: .expert
    )
}
