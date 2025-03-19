// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ParetoOptimization",
    platforms: [
        .macOS(.v13) // Updated to macOS 13 (Ventura) for better Metal support on M-series chips
    ],
    products: [
        .executable(name: "ParetoOptimization", targets: ["ParetoOptimization"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "ParetoOptimization",
            dependencies: [
                "OptimizationCore", 
                "GeneticAlgorithm", 
                "MetalAcceleration"
            ]
        ),
        .target(
            name: "OptimizationCore",
            dependencies: []
        ),
        .target(
            name: "GeneticAlgorithm",
            dependencies: ["OptimizationCore"]
        ),
        .target(
            name: "MetalAcceleration",
            dependencies: ["OptimizationCore"],
            exclude: ["dominance.metal"],  // Exclude the Metal shader file
            resources: [
                .process("Shaders")  // Add this if you have a Shaders directory
            ],
            swiftSettings: [
                .unsafeFlags(["-F/System/Library/Frameworks"]) // This ensures proper Foundation and Objective-C runtime access
            ]
        )
    ]
)