import Foundation
import MLX

enum ZImageCoordinateUtils {

  static func createCoordinateGrid(
    size: (Int, Int, Int),
    start: (Int, Int, Int)
  ) -> MLXArray {
    let (fSize, hSize, wSize) = size
    let (fStart, hStart, wStart) = start
    let shape = [fSize, hSize, wSize]

    let fCoords = MLXArray(Int32(fStart)..<Int32(fStart + fSize))
    let hCoords = MLXArray(Int32(hStart)..<Int32(hStart + hSize))
    let wCoords = MLXArray(Int32(wStart)..<Int32(wStart + wSize))

    let fExpanded = MLX.broadcast(fCoords.reshaped(fSize, 1, 1), to: shape)
    let hExpanded = MLX.broadcast(hCoords.reshaped(1, hSize, 1), to: shape)
    let wExpanded = MLX.broadcast(wCoords.reshaped(1, 1, wSize), to: shape)

    return MLX.stacked([fExpanded, hExpanded, wExpanded], axis: -1)
  }
}
