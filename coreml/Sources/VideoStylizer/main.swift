import Foundation
import CoreML
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import StableDiffusion

struct Job: Decodable {
    let compute_units: String?
    let model: String
    let frames: String
    let styled: String
    let prompt: String
    let negative_prompt: String
    let strength: Float
    let steps: Int
    let guidance: Float
    let seed: UInt32
    let temporal_mix: Float
}
func readImage(_ url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw NSError(domain: "Cannot read image: \(url.path)", code: 1)
    }
    return image
}
func execute() throws {
    guard CommandLine.arguments.count == 2 else {
        throw NSError(domain: "Usage: video-stylizer-coreml job.json", code: 1)
    }
    let job = try JSONDecoder().decode(Job.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
    let config = MLModelConfiguration()
    config.computeUnits = job.compute_units == "cpu-and-gpu" ? .cpuAndGPU : .cpuAndNeuralEngine
    print("[coreml] Loading DreamShaper (\(job.compute_units ?? "cpu-and-neural-engine")); first load can take several minutes")
    fflush(stdout)
    let pipeline = try StableDiffusionPipeline(resourcesAt: URL(fileURLWithPath: job.model), controlNet: [], configuration: config, disableSafety: true, reduceMemory: true)
    let frames = try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: job.frames), includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix("frame_") && $0.pathExtension == "png" }.sorted { $0.path < $1.path }
    let context = CIContext(options: [.useSoftwareRenderer: false])
    var previous: CGImage?
    for (index, frame) in frames.enumerated() {
        try autoreleasepool {
            let output = URL(fileURLWithPath: job.styled).appendingPathComponent(frame.lastPathComponent)
            if FileManager.default.fileExists(atPath: output.path) {
                previous = try readImage(output)
                print("[coreml] resume \(index + 1)/\(frames.count)")
                return
            }
            var input = try readImage(frame)
            if let previous = previous, job.temporal_mix > 0 {
                let base = CIImage(cgImage: input)
                let blend = CIImage(cgImage: previous).applyingFilter("CIDissolveTransition", parameters: [kCIInputTargetImageKey: base, kCIInputTimeKey: 1 - job.temporal_mix])
                guard let mixed = context.createCGImage(blend, from: base.extent) else { throw NSError(domain: "Temporal blend failed", code: 1) }
                input = mixed
            }
            var options = StableDiffusionPipeline.Configuration(prompt: job.prompt)
            options.negativePrompt = job.negative_prompt
            options.startingImage = input
            options.strength = job.strength
            options.stepCount = job.steps
            options.guidanceScale = job.guidance
            options.seed = job.seed
            options.disableSafety = true
            guard let result = try pipeline.generateImages(configuration: options, progressHandler: { progress in
                print("[coreml] frame \(index + 1)/\(frames.count), step \(progress.step + 1)/\(progress.stepCount)")
                fflush(stdout)
                return true
            }).first ?? nil else {
                throw NSError(domain: "No image generated", code: 1)
            }
            let temporary = output.appendingPathExtension("tmp")
            guard let destination = CGImageDestinationCreateWithURL(temporary as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw NSError(domain: "Cannot save output", code: 1) }
            CGImageDestinationAddImage(destination, result, nil)
            guard CGImageDestinationFinalize(destination) else { throw NSError(domain: "PNG save failed", code: 1) }
            try FileManager.default.moveItem(at: temporary, to: output)
            previous = result
            print("[coreml] \(index + 1)/\(frames.count) -> \(output.lastPathComponent)")
            fflush(stdout)
        }
    }
}
do { try execute() } catch {
    fputs("Core ML error: \(error)\n", stderr)
    exit(1)
}
