# Pareto Optimization Framework

A Swift framework for multi-objective optimization using Pareto optimality, with GPU acceleration support via Metal.

## Features

- Multi-objective optimization support
- GPU-accelerated dominance computation using Metal
- Generic Solution protocol for custom optimization problems
- Built-in support for numeric and categorical optimization criteria
- Efficient Pareto front calculation

## Requirements

- macOS with Metal support
- Swift 5.0 or later
- Xcode 12.0 or later

## Installation

Add this package to your Xcode project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "your-github-url-here", from: "1.0.0")
]
```

## Usage

```swift
import ParetoOptimization

// Create some solutions
let solutions = [DemoChromosome.random(), DemoChromosome.random()]

// Calculate Pareto front
let paretoFront = ParetoFront(solutions: solutions)

// Use GPU acceleration
let accelerator = MetalAccelerator.shared
let dominanceMatrix = accelerator.checkDominanceRelationships(solutions: solutions)
```

## License

MIT License