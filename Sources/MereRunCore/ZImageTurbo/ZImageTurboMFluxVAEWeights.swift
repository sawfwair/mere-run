import MLX

enum ZImageTurboMFluxVAEWeights {
  static func mapKey(_ key: String) -> String {
    key
      .replacingOccurrences(of: ".conv_in.conv.", with: ".conv_in.")
      .replacingOccurrences(of: ".conv_in.conv2d.", with: ".conv_in.")
      .replacingOccurrences(of: ".conv_out.conv.", with: ".conv_out.")
      .replacingOccurrences(of: ".conv_out.conv2d.", with: ".conv_out.")
      .replacingOccurrences(of: ".conv_norm_out.norm.", with: ".conv_norm_out.")
  }

  static func map(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
    [(mapKey(key), value)]
  }
}
