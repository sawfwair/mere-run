import XCTest
@testable import MereRunCore

final class SAM31PromptSetTests: XCTestCase {
    func testNormalizationCreatesStableObjectIDsAcrossPromptKinds() throws {
        let promptSet = SAM31PromptSet(
            textPrompts: ["A Dog", "A Dog"],
            boxPrompts: [
                SAM31PromptBox(x1: 10, y1: 20, x2: 30, y2: 40, label: "person")
            ],
            pointPrompts: [
                SAM31PromptPoint(x: 1, y: 2, isPositive: true, label: "person"),
                SAM31PromptPoint(x: 5, y: 6, isPositive: false, label: "person"),
                SAM31PromptPoint(x: 7, y: 8, isPositive: true)
            ]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID).prefix(3), ["a-dog", "a-dog-2", "person"])
        XCTAssertEqual(Set(objects.map(\.objectID)), Set(["a-dog", "a-dog-2", "person", "person-2", "point-object"]))
        XCTAssertEqual(objects[0].promptKind, .text)
        XCTAssertEqual(objects[2].promptKind, .box)

        let pointObjects = objects.filter { $0.promptKind == .point }
        XCTAssertEqual(pointObjects.count, 2)
        XCTAssertTrue(pointObjects.contains(where: { $0.objectID == "person-2" && $0.pointPrompts.count == 2 }))
        XCTAssertTrue(pointObjects.contains(where: { $0.objectID == "point-object" && $0.pointPrompts.count == 1 }))
    }

    func testNormalizationRejectsInvalidBoxes() {
        let promptSet = SAM31PromptSet(
            boxPrompts: [
                SAM31PromptBox(x1: 10, y1: 20, x2: 5, y2: 40, label: "dog")
            ]
        )

        XCTAssertThrowsError(try promptSet.normalized()) { error in
            guard case SAM31PromptSet.ValidationError.invalidBox(let box) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(box.label, "dog")
        }
    }
}
