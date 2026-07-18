import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

public struct RealFFTPlan {
    public let size: Int

    #if canImport(Accelerate)
    private let acceleratePlan: vDSP.FFT<DSPSplitComplex>?
    private let dftRealBasis: [Float]?
    private let dftImaginaryBasis: [Float]?
    #endif

    public init(size: Int) throws {
        guard size > 1 else {
            throw RealFFTError.invalidSize(size)
        }
        self.size = size
        #if canImport(Accelerate)
        if size.nonzeroBitCount == 1 {
            let log2n = vDSP_Length(log2(Double(size)))
            self.acceleratePlan = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)
            self.dftRealBasis = nil
            self.dftImaginaryBasis = nil
        } else {
            self.acceleratePlan = nil
            let frequencyCount = (size / 2) + 1
            let denominator = Float(size)
            var realBasis: [Float] = []
            var imaginaryBasis: [Float] = []
            realBasis.reserveCapacity(frequencyCount * size)
            imaginaryBasis.reserveCapacity(frequencyCount * size)
            for frequency in 0..<frequencyCount {
                for sampleIndex in 0..<size {
                    let angle = -2 * Float.pi * Float(frequency * sampleIndex) / denominator
                    realBasis.append(cos(angle))
                    imaginaryBasis.append(sin(angle))
                }
            }
            self.dftRealBasis = realBasis
            self.dftImaginaryBasis = imaginaryBasis
        }
        #endif
    }

    public func powerSpectrum(_ samples: [Float]) -> [Float] {
        var frame = samples
        if frame.count < size {
            frame += [Float](repeating: 0, count: size - frame.count)
        } else if frame.count > size {
            frame = Array(frame.prefix(size))
        }

        #if canImport(Accelerate)
        if let acceleratePlan {
            return acceleratePowerSpectrum(frame, plan: acceleratePlan)
        }
        if let dftRealBasis, let dftImaginaryBasis {
            return acceleratePowerSpectrum(
                frame,
                realBasis: dftRealBasis,
                imaginaryBasis: dftImaginaryBasis
            )
        }
        #endif

        return discretePowerSpectrum(frame)
    }

    #if canImport(Accelerate)
    private func acceleratePowerSpectrum(
        _ frame: [Float],
        plan: vDSP.FFT<DSPSplitComplex>
    ) -> [Float] {
        var real = [Float](repeating: 0, count: size / 2)
        var imag = [Float](repeating: 0, count: size / 2)
        frame.withUnsafeBufferPointer { inputPtr in
            real.withUnsafeMutableBufferPointer { realPtr in
                imag.withUnsafeMutableBufferPointer { imagPtr in
                    var split = DSPSplitComplex(
                        realp: realPtr.baseAddress!,
                        imagp: imagPtr.baseAddress!
                    )
                    vDSP_ctoz(
                        UnsafePointer<DSPComplex>(OpaquePointer(inputPtr.baseAddress!)),
                        2,
                        &split,
                        1,
                        vDSP_Length(size / 2)
                    )
                    plan.forward(input: split, output: &split)
                }
            }
        }

        var magnitudes = [Float](repeating: 0, count: (size / 2) + 1)
        magnitudes[0] = real[0] * real[0]
        for index in 1..<(size / 2) {
            magnitudes[index] = (real[index] * real[index]) + (imag[index] * imag[index])
        }
        magnitudes[size / 2] = imag[0] * imag[0]
        return magnitudes
    }

    private func acceleratePowerSpectrum(
        _ frame: [Float],
        realBasis: [Float],
        imaginaryBasis: [Float]
    ) -> [Float] {
        let frequencyCount = (size / 2) + 1
        var real = [Float](repeating: 0, count: frequencyCount)
        var imaginary = [Float](repeating: 0, count: frequencyCount)
        vDSP_mmul(
            realBasis, 1,
            frame, 1,
            &real, 1,
            vDSP_Length(frequencyCount), 1, vDSP_Length(size)
        )
        vDSP_mmul(
            imaginaryBasis, 1,
            frame, 1,
            &imaginary, 1,
            vDSP_Length(frequencyCount), 1, vDSP_Length(size)
        )

        var magnitudes = [Float](repeating: 0, count: frequencyCount)
        for index in 0..<frequencyCount {
            magnitudes[index] = (real[index] * real[index]) + (imaginary[index] * imaginary[index])
        }
        return magnitudes
    }
    #endif

    private func discretePowerSpectrum(_ frame: [Float]) -> [Float] {
        var magnitudes = [Float](repeating: 0, count: (size / 2) + 1)
        let denominator = Double(size)
        for frequency in 0...(size / 2) {
            var real = 0.0
            var imaginary = 0.0
            for sampleIndex in 0..<size {
                let angle = -2.0 * Double.pi * Double(frequency * sampleIndex) / denominator
                let value = Double(frame[sampleIndex])
                real += value * cos(angle)
                imaginary += value * sin(angle)
            }
            magnitudes[frequency] = Float((real * real) + (imaginary * imaginary))
        }
        return magnitudes
    }
}

public enum RealFFTError: LocalizedError {
    case invalidSize(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidSize(let size):
            return "FFT size must be greater than 1; got \(size)."
        }
    }
}
