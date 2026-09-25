import Foundation
import MereRunContract
import Testing

@testable import MereRunCLI

/// Image routing the generated gate cases don't reach: defaults that follow another option,
/// numeric values the CLI compares as numbers, and the models the gate stops before load.

private func report(_ commandLine: [String]) throws -> MereRunFamilyResolutionReport {
    try #require(CLICapabilityGate.evaluate(commandLine: commandLine)).report
}

@Test func imageGenerateDefaultsToZImageNano() throws {
    let blank = try report(["image", "generate", "--prompt", "a mug"])
    #expect(blank.family == "zimage" && blank.model == "image-zimage-nano" && blank.source == .defaultModel)
}

@Test func theKleinRecipeChoosesTheKleinBaseWhenNoModelIsGiven() throws {
    let output = ["--output", "/tmp/style.safetensors"]
    let klein = try report(["image", "train-lora", "--data", "/tmp/data", "--recipe", "klein-fast-style"] + output)
    #expect(klein.family == "klein" && klein.model == "image-klein-base-9b" && klein.source == .defaultModel)
    #expect(klein.violations.isEmpty)

    let krea = try report(["image", "train-lora", "--data", "/tmp/data", "--recipe", "krea-fast-style"] + output)
    #expect(krea.family == "krea" && krea.model == "image-krea2-raw" && krea.violations.isEmpty)

    let mismatched = try report(["image", "train-lora", "--recipe", "klein-fast-style", "--model", "image-krea2-raw"] + output)
    #expect(mismatched.violations == [
        "--recipe klein-fast-style is not supported by Krea 2; use krea-fast-style or krea-cinematic-style."
    ])
}

@Test func kleinTrainingNeedsADatasetAndKreaTrainingOptionsStayOnKrea() throws {
    let base = ["image", "train-lora", "--output", "/tmp/style.safetensors", "--model", "image-klein-base-9b"]
    #expect(try report(base).violations == ["FLUX.2 Klein requires --data."])
    let quantized = try report(base + ["--data", "/tmp/data", "--base-quantization-bits", "8"])
    #expect(quantized.violations == ["--base-quantization-bits is not supported by FLUX.2 Klein. It applies to Krea 2."])
    let synthetic = try report(["image", "train-lora", "--output", "/tmp/style.safetensors", "--synthetic-samples", "2"])
    #expect(synthetic.family == "krea" && synthetic.violations.isEmpty, "Krea 2 trains without a dataset")
}

@Test func lightningGuidanceIsComparedAsANumber() throws {
    let lightning = ["image", "generate", "-p", "edit", "--input", "/tmp/in.png", "--model", "image-qwen-edit-2511-lightning"]
    #expect(try report(lightning + ["--cfg", "1.0", "--steps", "4"]).violations.isEmpty)
    #expect(try report(lightning + ["--cfg-scale", "1"]).violations.isEmpty)
    #expect(try report(lightning + ["--cfg", "4"]).violations == ["--cfg 4 is not supported by Qwen-Image-Edit Lightning; use 1."])
    #expect(try report(lightning + ["-s", "8"]).violations == [
        "--steps 8 is not supported by Qwen-Image-Edit Lightning; it runs 4. Remove --steps or pass 4."
    ])
}

@Test func ignoredImageOptionsWarnAndRejectedOnesFailBeforeLoad() throws {
    let krea = try report(["image", "generate", "-p", "a mug", "-m", "image-krea2-turbo", "--cfg", "4", "-n", "blurry"])
    #expect(krea.violations.isEmpty)
    #expect(krea.warnings == [
        "--negative-prompt has no effect with Krea 2. It applies to FLUX.2 Klein, Z-Image, HiDream-O1, SenseNova U1.5, "
            + "Qwen-Image 2.1, Qwen-Image-Edit and Qwen-Image-Edit Lightning.",
        "--cfg has no effect with Krea 2. It applies to FLUX.1-dev, FLUX.2 Klein, FLUX.2-dev, Z-Image, HiDream-O1, "
            + "SenseNova U1.5, Qwen-Image 2.1, Qwen-Image-Edit, Qwen-Image-Edit Lightning and Ideogram 4."
    ])
    let stacked = try report(["image", "generate", "-p", "a mug", "-m", "image-zimage-nano", "-l", "a.safetensors", "-l", "b.safetensors"])
    #expect(stacked.violations == ["Z-Image takes --lora at most 1 time; got 2."])
    let flux1 = try report(["image", "generate", "-p", "a mug", "-m", "image-flux1-dev", "-i", "/tmp/in.png"])
    #expect(flux1.violations == [
        "--input is not supported by FLUX.1-dev. It applies to FLUX.2 Klein, FLUX.2-dev, Z-Image, HiDream-O1, "
            + "SenseNova U1.5, Qwen-Image 2.1, Qwen-Image-Edit and Qwen-Image-Edit Lightning."
    ])
}

@Test func modelsTheImageCommandsCannotRunStopAtTheGate() throws {
    let shared = try report(["image", "generate", "-p", "a mug", "--model", "image-klein-shared"])
    #expect(shared.source == .excluded && shared.violations.count == 1)
    let turbo = try report(["image", "train-lora", "--output", "/tmp/style.safetensors", "--model", "image-krea2-turbo"])
    #expect(turbo.source == .excluded && turbo.violations.first?.hasPrefix("image-krea2-turbo can't run image train-lora") == true)
}
