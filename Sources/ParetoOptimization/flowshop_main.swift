import Foundation
import OptimizationCore
import GeneticAlgorithm
import MetalAcceleration

/// Main function to solve the bi-objective flowshop scheduling problem
public func solveFlowshopProblem() {
    print("=== Bi-objective Flowshop Scheduling Solver ===")
    
    // Step 1: Load problem data
    let currentDirectory = FileManager.default.currentDirectoryPath
    print("Current directory: \(currentDirectory)")
    
    // Determine instance.csv path (try a few common locations)
    var instanceFilePath = currentDirectory + "/instance.csv"
    if !FileManager.default.fileExists(atPath: instanceFilePath) {
        instanceFilePath = currentDirectory + "/../instance.csv"
        if !FileManager.default.fileExists(atPath: instanceFilePath) {
            instanceFilePath = currentDirectory + "/../../instance.csv"
        }
    }
    
    print("Looking for instance data at: \(instanceFilePath)")
    
    guard let problemData = FlowshopProblemData.loadFromCSV(filePath: instanceFilePath) else {
        print("Error: Failed to load problem data")
        print("Falling back to test problem data")
        runWithTestData()
        return
    }
    
    print("Successfully loaded problem data from CSV:")
    print("Number of jobs: \(problemData.numJobs)")
    print("Number of machines: \(problemData.numMachines)")
    
    // Step 2: Run the genetic algorithm
    print("\nInitializing GPU-accelerated genetic algorithm...")
    
    // Configuration parameters - tuned for better performance
    let populationSize = 2000
    let generations = 100
    let mutationRate = 0.2
    let crossoverRate = 0.8
    let elitismCount = 100
    
    // Create the GA instance
    let ga = FlowshopGPUGeneticAlgorithm(
        populationSize: populationSize,
        mutationRate: mutationRate,
        crossoverRate: crossoverRate,
        elitismCount: elitismCount,
        problemData: problemData
    )
    
    print("\nStarting evolution...")
    let startTime = Date()
    
    // Run the genetic algorithm with progress reporting
    let paretoFrontSolutions = ga.evolve(generations: generations) { metrics in
        if metrics.iterations % 10 == 0 || metrics.iterations == generations {
            print("Gen \(metrics.iterations)/\(generations): \(metrics.paretoFrontSize) non-dominated solutions, time: \(String(format: "%.1f", metrics.timeElapsed))s")
        }
    }
    
    let totalTime = Date().timeIntervalSince(startTime)
    print("\nEvolution completed in \(String(format: "%.2f", totalTime)) seconds")
    print("Found \(paretoFrontSolutions.count) non-dominated solutions")
    
    // Step 3: Save the final Pareto front to a CSV file
    if !paretoFrontSolutions.isEmpty {
        saveParetoFrontToCSV(paretoFrontSolutions)
    }
    
    // Print some sample solutions from the Pareto front
    printSampleSolutions(paretoFrontSolutions)
}

/// Run with a test problem when the real instance can't be loaded
private func runWithTestData() {
    print("Creating test problem data...")
    let testData = FlowshopProblemData.createTestProblem(numJobs: 20, numMachines: 10)
    
    print("Number of jobs: \(testData.numJobs)")
    print("Number of machines: \(testData.numMachines)")
    
    // Configuration parameters - smaller for test
    let populationSize = 100
    let generations = 20
    let mutationRate = 0.05
    let crossoverRate = 0.8
    let elitismCount = 10
    
    // Create the GA instance
    let ga = FlowshopGPUGeneticAlgorithm(
        populationSize: populationSize,
        mutationRate: mutationRate,
        crossoverRate: crossoverRate,
        elitismCount: elitismCount,
        problemData: testData
    )
    
    print("\nStarting evolution...")
    let startTime = Date()
    
    // Run the genetic algorithm with progress reporting
    let paretoFrontSolutions = ga.evolve(generations: generations) { metrics in
        print("Generation \(metrics.iterations)/\(generations): \(metrics.paretoFrontSize) non-dominated solutions")
    }
    
    let totalTime = Date().timeIntervalSince(startTime)
    print("\nEvolution completed in \(String(format: "%.2f", totalTime)) seconds")
    print("Found \(paretoFrontSolutions.count) non-dominated solutions")
    
    // Print some solutions as examples
    printSampleSolutions(paretoFrontSolutions)
}

/// Print some sample solutions from the Pareto front
private func printSampleSolutions(_ solutions: [FlowshopChromosome]) {
    print("\nSample solutions from Pareto front:")
    let sampleCount = min(5, solutions.count)
    
    for i in 0..<sampleCount {
        print("Solution \(i+1): \(solutions[i].description())")
    }
    
    // Print statistics about the Pareto front
    if solutions.count > 0 {
        print("\nPareto Front Statistics:")
        
        // Extract makespan and tardiness values
        let makespans = solutions.map { ($0.criteria[0] as! NumericCriterion).value }
        let tardiness = solutions.map { ($0.criteria[1] as! NumericCriterion).value }
        
        // Calculate min, max, average
        let minMakespan = makespans.min() ?? 0
        let maxMakespan = makespans.max() ?? 0
        let avgMakespan = makespans.reduce(0, +) / Double(makespans.count)
        
        let minTardiness = tardiness.min() ?? 0
        let maxTardiness = tardiness.max() ?? 0
        let avgTardiness = tardiness.reduce(0, +) / Double(tardiness.count)
        
        print("Makespan range: \(String(format: "%.1f", minMakespan)) - \(String(format: "%.1f", maxMakespan)) (avg: \(String(format: "%.1f", avgMakespan)))")
        print("Tardiness range: \(String(format: "%.1f", minTardiness)) - \(String(format: "%.1f", maxTardiness)) (avg: \(String(format: "%.1f", avgTardiness)))")
    }
}

/// Save the Pareto front solutions to a CSV file
private func saveParetoFrontToCSV(_ solutions: [FlowshopChromosome]) {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd_HHmm"
    let timestamp = formatter.string(from: Date())
    
    // Create filename following the required format: <student1>_<student2>.csv
    let filename = "birch_delacalle_nicholas.csv"
    
    let outputDirectory = FileManager.default.currentDirectoryPath
    let filePath = outputDirectory + "/" + filename
    
    do {
        print("Saving Pareto front with \(solutions.count) solutions to \(filePath)")
        
        // Create CSV content
        var csvContent = ""
        
        for solution in solutions {
            // Convert job sequence to comma-separated format
            let sequenceString = solution.jobSequence.map { String($0) }.joined(separator: ",")
            csvContent += sequenceString + "\n"
        }
        
        // Write to file
        try csvContent.write(toFile: filePath, atomically: true, encoding: .utf8)
        
        print("Successfully saved Pareto front to \(filePath)")
        
        // Also save a version with timestamp for comparison
        let timestampedFilePath = outputDirectory + "/pareto_front_" + timestamp + ".csv"
        try csvContent.write(toFile: timestampedFilePath, atomically: true, encoding: .utf8)
    } catch {
        print("Failed to save Pareto front: \(error)")
    }
}