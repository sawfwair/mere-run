import Foundation

public struct QwenVisionGrid {
  public let temporal: Int
  public let height: Int
  public let width: Int

  public init(temporal: Int, height: Int, width: Int) {
    self.temporal = temporal
    self.height = height
    self.width = width
  }
}
