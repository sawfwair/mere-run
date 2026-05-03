import Foundation

enum CLICommandDisplay {
    static var executable: String {
        let executablePath = Bundle.main.executableURL?.path
            ?? CommandLine.arguments.first
            ?? "mere.run"
        if executablePath.contains("/.build/") {
            return "swift run mere.run"
        }
        return "mere.run"
    }

    static func command(_ arguments: String) -> String {
        "\(executable) \(arguments)"
    }
}
