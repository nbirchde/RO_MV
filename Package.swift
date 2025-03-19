// swift-tools-version: 5.8
import PackageDescription

let package = Package(
    name: "ParetoOptimization",
    platforms: [
        .macOS(.v13)
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
            exclude: ["Resources/FlowshopShaders.metal"],  // Exclude the resource version
            sources: ["FlowshopMetalAccelerator.swift"],
            resources: [
                .copy("FlowshopMetal.metal")  // Include the main Metal file as a resource
            ]
        )
    ]
)