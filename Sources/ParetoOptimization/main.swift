import Foundation
import OptimizationCore
import GeneticAlgorithm
import MetalAcceleration

/// A demo chromosome with multiple optimization criteria
struct DemoChromosome: Chromosome {
    // Unique identifier for the chromosome
    let id: UUID
    
    // Genes - these represent the variables that can be mutated/crossed over
    let genes: [Double]
    
    // Criteria values computed from genes
    let criteria: [CriterionValue]
    
    /// Create a new chromosome with the given genes
    init(genes: [Double]) {
        self.id = UUID()
        self.genes = genes
        
        // Calculate criteria values from genes
        // Example: We'll create three criteria representing different trade-offs
        let cost = 100 + genes[0] * 900  // Cost: 100-1000 (lower is better)
        let time = 1 + genes[1] * 99     // Time: 1-100 (lower is better)
        let quality = 1 + Int(genes[2] * 9) // Quality: 1-10 (higher is better)
        
        self.criteria = [
            NumericCriterion(cost, lowerIsBetter: true),
            NumericCriterion(time, lowerIsBetter: true),
            NumericCriterion(Double(quality), lowerIsBetter: false)
        ]
    }
    
    /// Create a random chromosome (for initial population)
    static func random() -> DemoChromosome {
        let randomGenes = (0..<3).map { _ in Double.random(in: 0...1) }
        return DemoChromosome(genes: randomGenes)
    }
    
    /// Create a new chromosome by crossing this chromosome with another
    func crossover(with other: DemoChromosome) -> DemoChromosome {
        // Single-point crossover
        let crossoverPoint = Int.random(in: 0..<genes.count)
        var childGenes = [Double](repeating: 0.0, count: genes.count)
        
        // Take genes from first parent up to crossover point
        for i in 0..<crossoverPoint {
            childGenes[i] = genes[i]
        }
        
        // Take genes from second parent after crossover point
        for i in crossoverPoint..<genes.count {
            childGenes[i] = other.genes[i]
        }
        
        return DemoChromosome(genes: childGenes)
    }
    
    /// Create a mutated version of this chromosome
    func mutate(mutationRate: Double) -> DemoChromosome {
        var mutatedGenes = genes
        
        // Apply mutation to each gene with probability mutationRate
        for i in 0..<mutatedGenes.count {
            if Double.random(in: 0...1) < mutationRate {
                // Add or subtract up to 20% of the gene's value (but keep within 0-1 range)
                let change = Double.random(in: -0.2...0.2)
                mutatedGenes[i] = max(0.0, min(1.0, mutatedGenes[i] + change))
            }
        }
        
        return DemoChromosome(genes: mutatedGenes)
    }
    
    // Hashable implementation
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    // Equatable implementation
    static func == (lhs: DemoChromosome, rhs: DemoChromosome) -> Bool {
        return lhs.id == rhs.id
    }
    
    // String representation for printing
    func description() -> String {
        let costCriterion = criteria[0] as! NumericCriterion
        let timeCriterion = criteria[1] as! NumericCriterion
        let qualityCriterion = criteria[2] as! NumericCriterion
        
        return "Cost: \(String(format: "%.2f", costCriterion.value)), " +
               "Time: \(String(format: "%.2f", timeCriterion.value)), " +
               "Quality: \(String(format: "%.0f", qualityCriterion.value))"
    }
}

/// Wrapper for experiments to compare different optimization approaches
struct OptimizationExperiment {
    // Run a random search experiment
    static func runRandomSearch(solutionsCount: Int) -> [DemoChromosome] {
        print("\n=== Random Search Experiment ===")
        print("Generating \(solutionsCount) random solutions...")
        
        let startTime = Date()
        
        // Generate random solutions
        var randomSolutions: [DemoChromosome] = []
        randomSolutions.reserveCapacity(solutionsCount)
        
        for _ in 0..<solutionsCount {
            randomSolutions.append(DemoChromosome.random())
        }
        
        // Find Pareto front using standard approach
        let paretoFront = ParetoFront<DemoChromosome>()
        _ = paretoFront.addBatch(randomSolutions) // Use underscore to avoid unused warning
        
        let endTime = Date()
        let elapsedTime = endTime.timeIntervalSince(startTime)
        
        print("Random search completed in \(String(format: "%.2f", elapsedTime)) seconds")
        print("Evaluated \(solutionsCount) random solutions")
        print("Found \(paretoFront.solutions.count) non-dominated solutions")
        print("Solutions per second: \(String(format: "%.2f", Double(solutionsCount) / elapsedTime))")
        
        return paretoFront.solutions
    }
    
    // Run a Metal-accelerated random search experiment
    static func runMetalAcceleratedRandomSearch(solutionsCount: Int) -> [DemoChromosome] {
        print("\n=== Metal-Accelerated Random Search Experiment ===")
        print("Device Info: \(String(describing: MetalAccelerator.shared.deviceInfo()))")
        print("Generating \(solutionsCount) random solutions with Metal acceleration...")
        
        // Check if Metal is available first
        if !MetalAccelerator.shared.isMetalAvailable {
            print("Metal acceleration is not available on this device.")
            print("Falling back to CPU implementation...")
            return runRandomSearch(solutionsCount: solutionsCount)
        }
        
        let startTime = Date()
        var solutionGenTime: TimeInterval = 0
        var paretoCalcTime: TimeInterval = 0
        
        // Generate random solutions
        let genStartTime = Date()
        var randomSolutions: [DemoChromosome] = []
        randomSolutions.reserveCapacity(solutionsCount)
        
        // Generate in batches to avoid memory issues
        let batchSize = 5000 // Increased batch size for better GPU utilization
        for batchStart in stride(from: 0, to: solutionsCount, by: batchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + batchSize, solutionsCount)
                print("Generating batch \(batchStart)-\(batchEnd)...")
                
                for _ in batchStart..<batchEnd {
                    randomSolutions.append(DemoChromosome.random())
                }
            }
        }
        solutionGenTime = Date().timeIntervalSince(genStartTime)
        print("Solution generation completed in \(String(format: "%.2f", solutionGenTime)) seconds")
        
        // Find Pareto front using Metal acceleration
        print("\nCalculating Pareto front with Metal acceleration...")
        let paretoStartTime = Date()
        let paretoFront = ParetoFront<DemoChromosome>()
        
        // Process in batches to limit GPU memory usage
        let frontBatchSize = 5000 // Increased for better GPU utilization
        for batchStart in stride(from: 0, to: randomSolutions.count, by: frontBatchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + frontBatchSize, randomSolutions.count)
                let batch = Array(randomSolutions[batchStart..<batchEnd])
                print("Processing Pareto front batch \(batchStart)-\(batchEnd)...")
                _ = paretoFront.addBatchUsingMetal(batch) // Re-enabled Metal acceleration
            }
        }
        paretoCalcTime = Date().timeIntervalSince(paretoStartTime)
        
        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        
        print("\nPerformance Breakdown:")
        print("Solution Generation: \(String(format: "%.2f", solutionGenTime)) seconds")
        print("Pareto Front Calculation: \(String(format: "%.2f", paretoCalcTime)) seconds")
        print("Total Time: \(String(format: "%.2f", totalTime)) seconds")
        print("Solutions per second: \(String(format: "%.2f", Double(solutionsCount) / totalTime))")
        print("Found \(paretoFront.solutions.count) non-dominated solutions")
        
        return paretoFront.solutions
    }
    
    // Run a genetic algorithm experiment
    static func runGeneticAlgorithm(populationSize: Int, generations: Int) -> [DemoChromosome] {
        print("\n=== Genetic Algorithm Experiment ===")
        print("Population size: \(populationSize)")
        print("Generations: \(generations)")
        
        let startTime = Date()
        
        // Create genetic algorithm with our demo chromosome
        let ga = GeneticAlgorithm<DemoChromosome>(
            populationSize: populationSize,
            mutationRate: 0.1,
            crossoverRate: 0.8,
            elitismCount: populationSize / 10
        )
        
        // Run the genetic algorithm with progress reporting
        let paretoFrontSolutions = ga.evolve(generations: generations) { metrics in
            if metrics.iterations % 10 == 0 || metrics.iterations == generations {
                print("Generation \(metrics.iterations)/\(generations): \(metrics.paretoFrontSize) solutions in Pareto front")
            }
        }
        
        let endTime = Date()
        let elapsedTime = endTime.timeIntervalSince(startTime)
        let totalSolutions = populationSize * generations
        
        print("Genetic algorithm completed in \(String(format: "%.2f", elapsedTime)) seconds")
        print("Evaluated \(totalSolutions) solutions over \(generations) generations")
        print("Found \(paretoFrontSolutions.count) non-dominated solutions")
        print("Solutions per second: \(String(format: "%.2f", Double(totalSolutions) / elapsedTime))")
        
        return paretoFrontSolutions
    }
}

// Print hardware information to show optimization potential
print("=== Hardware Information ===")
print("System: \(ProcessInfo.processInfo.hostName)")
print("Available CPU cores: \(ProcessInfo.processInfo.activeProcessorCount)")
print("Physical memory: \(ProcessInfo.processInfo.physicalMemory / (1024 * 1024)) MB")
print("===========================")

// Define experiment parameters - adjusted for better performance
let randomSolutionCount = 100_000 // Reduced from 500,000 to 100,000 for faster testing
let gaPopulationSize = 500 
let gaGenerations = 20

// Menu for choosing which experiment to run
print("\nChoose an experiment to run:")
print("1. Random search with CPU")
print("2. Random search with Metal acceleration")
print("3. Genetic algorithm")
print("4. Run all experiments")
print("5. Direct Metal vs CPU comparison")
print("6. Exit")

// Helper function to print sample solutions
func printSampleSolutions(_ solutions: [DemoChromosome], method: String) {
    print("\n=== Sample Solutions from \(method) ===")
    
    if solutions.isEmpty {
        print("No solutions found!")
        return
    }
    
    let sampleSize = min(5, solutions.count)
    for i in 0..<sampleSize {
        print("Solution \(i+1): \(solutions[i].description())")
    }
}

// Helper function to find the best solution with weighted criteria
func findWeightedBest(_ solutions: [DemoChromosome], weights: [Double]) -> DemoChromosome? {
    var bestSolution: DemoChromosome? = nil
    var bestScore = Double.infinity
    
    for solution in solutions {
        var score: Double = 0.0
        
        for (i, criterion) in solution.criteria.enumerated() {
            if i < weights.count, let numericCriterion = criterion as? NumericCriterion {
                // For criteria where higher is better, negate the weight
                let effectiveWeight = numericCriterion.lowerIsBetter ? weights[i] : -weights[i]
                score += numericCriterion.value * effectiveWeight
            }
        }
        
        if score < bestScore {
            bestScore = score
            bestSolution = solution
        }
    }
    
    return bestSolution
}

// Helper function to show weighted best solutions
func showWeightedBestSolutions(_ randomSolutions: [DemoChromosome], _ metalSolutions: [DemoChromosome], _ gaSolutions: [DemoChromosome]) {
    // Example of finding weighted best solution
    let weights = [0.5, 0.3, 0.2] // Cost, Time, Quality
    print("\n=== Weighted Best Solutions ===")
    print("Weights: Cost=0.5, Time=0.3, Quality=0.2")

    if !randomSolutions.isEmpty, let bestRandom = findWeightedBest(randomSolutions, weights: weights) {
        print("Random Search Best: \(bestRandom.description())")
    }

    if !metalSolutions.isEmpty, let bestMetal = findWeightedBest(metalSolutions, weights: weights) {
        print("Metal-Accelerated Best: \(bestMetal.description())")
    }

    if !gaSolutions.isEmpty, let bestGA = findWeightedBest(gaSolutions, weights: weights) {
        print("Genetic Algorithm Best: \(bestGA.description())")
    }
}

// Add this new function for direct Metal vs CPU comparison with improved GPU performance
func runMetalVsCPUComparison() {
    print("\n=== Direct Metal vs CPU Comparison ===")
    
    // Use a more reasonable size for comparison
    let testSizes = [5_000, 10_000, 20_000]
    
    for size in testSizes {
        print("\n--- Testing with \(size) solutions ---")
        
        // Generate test data once
        print("Generating \(size) random solutions...")
        var randomSolutions: [DemoChromosome] = []
        randomSolutions.reserveCapacity(size)
        
        for _ in 0..<size {
            randomSolutions.append(DemoChromosome.random())
        }
        
        // Test CPU implementation
        print("\nTesting CPU implementation...")
        let cpuStartTime = Date()
        let cpuParetoFront = ParetoFront<DemoChromosome>()
        _ = cpuParetoFront.addBatch(randomSolutions)
        let cpuTime = Date().timeIntervalSince(cpuStartTime)
        
        // Test Metal implementation - using optimized approach
        print("\nTesting optimized Metal implementation...")
        let metalStartTime = Date()
        
        // Create GPU accelerator
        let accelerator = MetalAccelerator.shared
        
        // Get dominance matrix directly without batching inside
        let dominanceMatrix = accelerator.checkDominanceRelationshipsOptimized(solutions: randomSolutions)
        
        // Find non-dominated solutions
        let nonDominated = accelerator.findNonDominatedSolutions(
            solutions: randomSolutions, dominanceMatrix: dominanceMatrix)
            
        let metalTime = Date().timeIntervalSince(metalStartTime)
        
        // Show results
        print("\nResults:")
        print("CPU Time: \(String(format: "%.4f", cpuTime)) seconds")
        print("GPU Time: \(String(format: "%.4f", metalTime)) seconds")
        print("CPU Solutions per second: \(String(format: "%.2f", Double(size) / cpuTime))")
        print("GPU Solutions per second: \(String(format: "%.2f", Double(size) / metalTime))")
        print("Speedup: \(String(format: "%.2fx", cpuTime / metalTime))")
        print("CPU Pareto front size: \(cpuParetoFront.solutions.count)")
        print("GPU Pareto front size: \(nonDominated.count)")
    }
}

// Read user input and execute the selected experiment
if let input = readLine(), let option = Int(input) {
    var randomSearchSolutions: [DemoChromosome] = []
    var metalAcceleratedSolutions: [DemoChromosome] = []
    var gaSolutions: [DemoChromosome] = []
    
    switch option {
    case 1:
        // CPU-based random search
        randomSearchSolutions = OptimizationExperiment.runRandomSearch(solutionsCount: randomSolutionCount)
        printSampleSolutions(randomSearchSolutions, method: "Random Search")
        
    case 2:
        // Metal-accelerated random search
        metalAcceleratedSolutions = OptimizationExperiment.runMetalAcceleratedRandomSearch(solutionsCount: randomSolutionCount)
        printSampleSolutions(metalAcceleratedSolutions, method: "Metal-Accelerated Search")
        
    case 3:
        // Genetic algorithm
        gaSolutions = OptimizationExperiment.runGeneticAlgorithm(populationSize: gaPopulationSize, generations: gaGenerations)
        printSampleSolutions(gaSolutions, method: "Genetic Algorithm")
        
    case 4:
        // Run experiments sequentially, not all at once
        print("\nRunning all experiments sequentially...")
        
        // 1. Random search with CPU
        randomSearchSolutions = OptimizationExperiment.runRandomSearch(solutionsCount: randomSolutionCount)
        printSampleSolutions(randomSearchSolutions, method: "Random Search")
        
        // 2. Random search with Metal acceleration - only if previous step succeeded
        if !randomSearchSolutions.isEmpty {
            metalAcceleratedSolutions = OptimizationExperiment.runMetalAcceleratedRandomSearch(solutionsCount: randomSolutionCount)
            printSampleSolutions(metalAcceleratedSolutions, method: "Metal-Accelerated Search")
        }
        
        // 3. Genetic algorithm - only if previous steps succeeded
        if !randomSearchSolutions.isEmpty {
            gaSolutions = OptimizationExperiment.runGeneticAlgorithm(populationSize: gaPopulationSize, generations: gaGenerations)
            printSampleSolutions(gaSolutions, method: "Genetic Algorithm")
            
            // Show weighted best solutions if we have results from all methods
            if !randomSearchSolutions.isEmpty && !gaSolutions.isEmpty {
                showWeightedBestSolutions(randomSearchSolutions, metalAcceleratedSolutions, gaSolutions)
            }
        }
        
    case 5:
        // Direct Metal vs CPU comparison with controlled tests
        runMetalVsCPUComparison()
        
    case 6:
        print("Exiting...")
        exit(0)
        
    default:
        print("Invalid option")
    }
    
    // If we got here, we ran at least one experiment
    print("\nOptimization framework execution completed!")
    print("The project is ready for adaptation to your specific multi-criteria optimization problem.")
    
} else {
    print("Invalid input, please run again and select an option 1-6")
}