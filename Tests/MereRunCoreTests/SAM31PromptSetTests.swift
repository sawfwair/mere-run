import XCTest
@testable import MereRunCore

final class SAM31PromptSetTests: XCTestCase {
    private let box = SAM31PromptBox(x1: 10, y1: 20, x2: 30, y2: 40)

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
            ],
            maskPrompts: [SAM31PromptMask(path: "/tmp/mask.png")]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID), ["a-dog", "a-dog-2", "person", "point-object", "mask-object"])
        XCTAssertEqual(objects.map(\.promptKind), [.text, .text, .box, .point, .mask])
        // The labeled points refine the box of the same label rather than forming a second object.
        XCTAssertEqual(objects[2].boxPrompt?.label, "person")
        XCTAssertEqual(objects[2].pointPrompts.map(\.x), [1, 5])
        XCTAssertEqual(objects[3].pointPrompts.map(\.x), [7])
    }

    func testLabeledPointsJoinTheBoxWithTheirLabelThatContainsThemElseTheFirst() throws {
        let promptSet = SAM31PromptSet(
            boxPrompts: [
                SAM31PromptBox(x1: 0, y1: 0, x2: 10, y2: 10, label: "cup"),
                SAM31PromptBox(x1: 20, y1: 20, x2: 30, y2: 30, label: "cup"),
                SAM31PromptBox(x1: 40, y1: 40, x2: 50, y2: 50, label: "plate")
            ],
            pointPrompts: [
                SAM31PromptPoint(x: 5, y: 5, isPositive: true, label: "cup"),
                SAM31PromptPoint(x: 45, y: 45, isPositive: false, label: " plate "),
                SAM31PromptPoint(x: 25, y: 25, isPositive: false, label: "cup"),
                SAM31PromptPoint(x: 90, y: 90, isPositive: true, label: "cup")
            ]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID), ["cup", "cup-2", "plate"])
        XCTAssertEqual(objects[0].pointPrompts.map(\.x), [5, 90], "inside the first box, and outside every box")
        XCTAssertEqual(objects[1].pointPrompts.map(\.x), [25], "inside the second box of the same label")
        XCTAssertEqual(objects[2].pointPrompts.map(\.x), [45])
    }

    func testPointOnlyGroupsNeedAPositivePoint() {
        let unlabeled = SAM31PromptSet(pointPrompts: [SAM31PromptPoint(x: 1, y: 1, isPositive: false)])
        XCTAssertThrowsError(try unlabeled.normalized()) { error in
            guard case SAM31PromptSet.ValidationError.noPositivePoint(let label) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertNil(label)
            XCTAssertEqual(error.localizedDescription, "The unlabeled point prompts are all negative and match no box. Add a positive point, or a box for them to refine.")
        }
        let labeled = SAM31PromptSet(
            boxPrompts: [SAM31PromptBox(x1: 0, y1: 0, x2: 10, y2: 10, label: "cup")],
            pointPrompts: [
                SAM31PromptPoint(x: 5, y: 5, isPositive: false, label: "cup"),
                SAM31PromptPoint(x: 50, y: 50, isPositive: false, label: "shadow")
            ]
        )
        XCTAssertThrowsError(try labeled.normalized()) { error in
            guard case SAM31PromptSet.ValidationError.noPositivePoint(let label) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(label, "shadow", "the negative point on the box is fine; the stray one is not")
        }
        // A negative point refining a box, or beside a positive point, is a valid object.
        XCTAssertNoThrow(try SAM31PromptSet(
            boxPrompts: [box],
            pointPrompts: [SAM31PromptPoint(x: 15, y: 25, isPositive: false)]
        ).normalized())
        XCTAssertNoThrow(try SAM31PromptSet(pointPrompts: [
            SAM31PromptPoint(x: 1, y: 1, isPositive: false),
            SAM31PromptPoint(x: 2, y: 2, isPositive: true)
        ]).normalized())
    }

    func testLabeledPointsWithoutABoxFormOneObjectPerLabelInFirstAppearanceOrder() throws {
        let promptSet = SAM31PromptSet(
            boxPrompts: [SAM31PromptBox(x1: 0, y1: 0, x2: 10, y2: 10, label: "cup")],
            pointPrompts: [
                SAM31PromptPoint(x: 1, y: 1, isPositive: true, label: "spoon"),
                SAM31PromptPoint(x: 2, y: 2, isPositive: true, label: "bowl"),
                SAM31PromptPoint(x: 3, y: 3, isPositive: false, label: "spoon")
            ]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID), ["cup", "spoon", "bowl"])
        XCTAssertEqual(objects.map(\.promptKind), [.box, .point, .point])
        XCTAssertEqual(objects[1].pointPrompts.map(\.x), [1, 3])
        XCTAssertEqual(objects[2].pointPrompts.map(\.x), [2])
    }

    func testUnlabeledPointsFormOneObjectTogether() throws {
        let promptSet = SAM31PromptSet(
            pointPrompts: [
                SAM31PromptPoint(x: 1, y: 1, isPositive: true),
                SAM31PromptPoint(x: 2, y: 2, isPositive: false, label: "  "),
                SAM31PromptPoint(x: 3, y: 3, isPositive: true)
            ]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID), ["point-object"])
        XCTAssertEqual(objects[0].pointPrompts.map(\.isPositive), [true, false, true])
    }

    func testUnlabeledPointsRefineTheOnlyUnlabeledBox() throws {
        let promptSet = SAM31PromptSet(
            boxPrompts: [
                SAM31PromptBox(x1: 0, y1: 0, x2: 10, y2: 10, label: "cup"),
                box
            ],
            pointPrompts: [
                SAM31PromptPoint(x: 15, y: 25, isPositive: true),
                SAM31PromptPoint(x: 28, y: 38, isPositive: false),
                SAM31PromptPoint(x: 5, y: 5, isPositive: true, label: "cup")
            ]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID), ["cup", "object"])
        XCTAssertEqual(objects[0].pointPrompts.map(\.x), [5])
        XCTAssertEqual(objects[1].boxPrompt, box)
        XCTAssertEqual(objects[1].pointPrompts.map(\.x), [15, 28])
        XCTAssertEqual(objects[1].promptKind, .box)
    }

    func testUnlabeledPointsStayApartFromSeveralUnlabeledBoxes() throws {
        let promptSet = SAM31PromptSet(
            boxPrompts: [box, SAM31PromptBox(x1: 50, y1: 50, x2: 60, y2: 60)],
            pointPrompts: [SAM31PromptPoint(x: 15, y: 25, isPositive: true)]
        )

        let objects = try promptSet.normalized()

        XCTAssertEqual(objects.map(\.objectID), ["object", "object-2", "point-object"])
        XCTAssertEqual(objects.map(\.pointPrompts.count), [0, 0, 1])
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

    func testNormalizationRejectsMoreObjectsThanSupported() {
        let promptSet = SAM31PromptSet(textPrompts: ["a", "b", "c"])

        XCTAssertThrowsError(try promptSet.normalized(maxObjects: 2)) { error in
            guard case SAM31PromptSet.ValidationError.tooManyObjects(let count, let maxSupported) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(count, 3)
            XCTAssertEqual(maxSupported, 2)
        }
    }
}
