import Foundation
#if os(Linux)
import Glibc
#else
import Darwin
#endif

enum CLIStderr {
    static func write(_ text: String) {
        FileHandle.standardError.write(Data(text.utf8))
    }
}

enum CLIStdout {
    static func write(_ text: String) {
        FileHandle.standardOutput.write(Data(text.utf8))
    }

    static func isInteractive() -> Bool {
        isatty(STDOUT_FILENO) != 0
    }
}

enum CLIStdin {
    static func isInteractive() -> Bool {
        isatty(STDIN_FILENO) != 0
    }
}
