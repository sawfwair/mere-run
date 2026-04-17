import ArgumentParser
import MereRunCore

extension ModelResolver.ModelID: ExpressibleByArgument {
    public init?(argument: String) {
        self.init(rawValue: argument)
    }
}
