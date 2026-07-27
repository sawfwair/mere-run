import ArgumentParser
import MereRunContract
import MereRunCore

extension LTXVideoVariant: ExpressibleByArgument {}
extension LTXVideoQuality: ExpressibleByArgument {}
extension LTXVideoOutputMode: ExpressibleByArgument {}

extension ModelResolver.ModelID: ExpressibleByArgument {
    public init?(argument: String) {
        self.init(rawValue: argument)
    }
}
