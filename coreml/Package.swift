// swift-tools-version: 5.8
import PackageDescription
let package = Package(
    name: "VideoStylizerCoreML", platforms: [.macOS("13.1")],
    products: [.executable(name: "video-stylizer-coreml", targets: ["VideoStylizer"])],
    dependencies: [.package(url: "https://github.com/apple/ml-stable-diffusion.git", revision: "ea2805dc1945be20561c77e5f6d1d9a5a637cda2")],
    targets: [.executableTarget(name: "VideoStylizer", dependencies: [.product(name: "StableDiffusion", package: "ml-stable-diffusion")])]
)
