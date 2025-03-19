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
            resources: [
                .process("Resources/FlowshopMetal.metal")
            ]
        )
    ]
)