import Foundation

@inline(__always)
package func ltxMonotonicSeconds() -> Double {
    ProcessInfo.processInfo.systemUptime
}
