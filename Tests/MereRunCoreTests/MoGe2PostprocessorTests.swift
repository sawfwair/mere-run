import MereRunCore
import XCTest

final class MoGe2PostprocessorTests: XCTestCase {
    func testRecoversKnownFocalAndShiftFromAffinePointMap() {
        let width = 32
        let height = 24
        let focal = 1.25
        let shift = 0.7
        let aspect = Double(width) / Double(height)
        let spanX = aspect / sqrt(1 + aspect * aspect)
        let spanY = 1 / sqrt(1 + aspect * aspect)
        var points: [Float] = []
        var validity: [UInt8] = []
        for y in 0..<height {
            let v = -spanY * Double(height - 1) / Double(height)
                + 2 * spanY * Double(height - 1) / Double(height) * Double(y) / Double(height - 1)
            for x in 0..<width {
                let u = -spanX * Double(width - 1) / Double(width)
                    + 2 * spanX * Double(width - 1) / Double(width) * Double(x) / Double(width - 1)
                let cameraZ = 2 + 0.01 * Double(x) + 0.02 * Double(y)
                points.append(Float(u / focal * cameraZ))
                points.append(Float(v / focal * cameraZ))
                points.append(Float(cameraZ - shift))
                validity.append(1)
            }
        }
        let solution = MoGe2FocalShiftSolver.solve(
            affinePoints: points,
            validity: validity,
            width: width,
            height: height
        )
        XCTAssertEqual(solution.focal, focal, accuracy: 1e-4)
        XCTAssertEqual(solution.shift, shift, accuracy: 1e-4)
        XCTAssertLessThan(solution.residualMeanSquare, 1e-10)
    }

    func testKnownFocalSolvesOnlyShift() {
        let points: [Float] = [
            -0.5, 0, 1,
             0.5, 0, 1,
            -0.5, 0, 1,
             0.5, 0, 1,
        ]
        let result = MoGe2FocalShiftSolver.solve(
            affinePoints: points,
            validity: [1, 1, 1, 1],
            width: 2,
            height: 2,
            knownFocal: 1
        )
        XCTAssertTrue(result.focal.isFinite)
        XCTAssertTrue(result.shift.isFinite)
    }
}
