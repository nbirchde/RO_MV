import Foundation
import OptimizationCore
import GeneticAlgorithm
import MetalAcceleration

// Print welcome message
print("===================================================")
print("| Bi-objective Flowshop Scheduler with GPU Support |")
print("===================================================")

// Get command-line arguments
let args = CommandLine.arguments
var populationSize = 1000
var generations = 100

// Parse command line arguments if provided
if args.count > 1 {
    for i in 1..<args.count {
        let arg = args[i]
        if arg.starts(with: "--pop="), let value = Int(arg.dropFirst(6)) {
            populationSize = value
        } else if arg.starts(with: "--gen="), let value = Int(arg.dropFirst(6)) {
            generations = value
        }
    }
}

print("Configuration:")
print("- Population size: \(populationSize)")
print("- Generations: \(generations)")

// Hardware information
print("\n=== Hardware Information ===")
print("System: \(ProcessInfo.processInfo.hostName)")
print("Available CPU cores: \(ProcessInfo.processInfo.activeProcessorCount)")
print("Physical memory: \(ProcessInfo.processInfo.physicalMemory / (1024 * 1024)) MB")

// Check if Metal GPU acceleration is available
if MetalAccelerator.shared.isMetalAvailable {
    print("\n=== Metal GPU Information ===")
    print(MetalAccelerator.shared.deviceInfo())
}

// Run the flowshop solver
solveFlowshopProblem()