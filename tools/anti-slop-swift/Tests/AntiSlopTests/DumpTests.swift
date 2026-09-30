import XCTest
import SwiftSyntax
import SwiftParser
import SwiftOperators
@testable import AntiSlopCore

final class DumpTests: XCTestCase {
    func testDump() throws {
        let tree = Parser.parse(source: "let enabled = flag == true\n")
        let folded = OperatorTable.standardOperators.foldAll(tree) { _ in }
        var out = ""
        func walk(_ node: Syntax, depth: Int) {
            out += String(repeating: " ", count: depth) + "\(node.kind)\n"
            for child in node.children(viewMode: .sourceAccurate) { walk(child, depth: depth + 2) }
        }
        walk(Syntax(folded), depth: 0)
        print("TREE_START\n\(out)TREE_END")
    }
}
