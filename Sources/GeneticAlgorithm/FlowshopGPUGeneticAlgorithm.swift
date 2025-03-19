import Foundation
import OptimizationCore
import MetalAcceleration

/// A GPU-accelerated genetic algorithm for flowshop scheduling problems
public class FlowshopGPUGeneticAlgorithm {
    // Configuration parameters
    private let populationSize: Int
    private let mutationRate: Double
    private let crossoverRate: Double
    private let elitismCount: Int
    
    // Problem data
    private let problemData: FlowshopProblemData
    
    // Current population
    private var population: [FlowshopChromosome] = []
    
    // Pareto front of non-dominated solutions
    private let paretoFront = ParetoFront<FlowshopChromosome>()
    
    // Metal accelerator for GPU acceleration
    private let metalAccelerator: FlowshopMetalAccelerator
    
    // Performance metrics
    public struct Metrics {
        public var iterations: Int = 0
        public var solutionsEvaluated: Int = 0
        public var paretoFrontSize: Int = 0
        public var timeElapsed: TimeInterval = 0
        public var gpuTimeElapsed: TimeInterval = 0
        public var cpuTimeElapsed: TimeInterval = 0
    }
    
    private var metrics = Metrics()
    
    // Flag to determine if the accelerator is available
    private let useGPU: Bool
    
    /// Initialize the genetic algorithm
    public init(
        populationSize: Int,
        mutationRate: Double,
        crossoverRate: Double,
        elitismCount: Int,
        problemData: FlowshopProblemData
    ) {
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.elitismCount = min(elitismCount, populationSize / 2) // Prevent elitism count from being too large
        self.problemData = problemData
        self.metalAccelerator = FlowshopMetalAccelerator.shared
        
        // Initialize problem data on GPU
        let processingTimes = problemData.processingTimes.map { $0.map { Float($0) } }
        let priorities = problemData.priorities.map { Float($0) }
        let deadlines = problemData.deadlines.map { Float($0) }
        
        let gpuInitSuccess = metalAccelerator.loadFlowshopProblemData(
            processingTimes: processingTimes,
            priorities: priorities,
            deadlines: deadlines
        )
        
        self.useGPU = gpuInitSuccess
        print("GPU acceleration \(useGPU ? "enabled" : "disabled")")
    }
    
    /// Initialize the population with random chromosomes
    private func initializePopulation() {
        print("Initializing population of \(populationSize) chromosomes...")
        
        if useGPU {
            initializePopulationGPU()
        } else {
            initializePopulationCPU()
        }
        
        print("Initialization complete. Population size: \(population.count), Pareto front size: \(paretoFront.count)")
    }
    
    /// Initialize population on the CPU
    private func initializePopulationCPU() {
        let startTime = Date()
        let batchSize = min(500, populationSize)
        var initialPopulation: [FlowshopChromosome] = []
        initialPopulation.reserveCapacity(populationSize)
        
        // Generate chromosomes in batches
        for batchStart in stride(from: 0, to: populationSize, by: batchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + batchSize, populationSize)
                let batchCount = batchEnd - batchStart
                
                print("Generating batch \(batchStart)-\(batchEnd-1) on CPU...")
                
                // Generate a batch of random chromosomes
                for _ in 0..<batchCount {
                    initialPopulation.append(FlowshopChromosome.random(with: problemData))
                }
            }
        }
        
        // Store the population
        population = initialPopulation
        
        // Add initial population to the Pareto front
        for solution in initialPopulation {
            _ = paretoFront.add(solution)
        }
        
        let totalTime = Date().timeIntervalSince(startTime)
        print("CPU initialization completed in \(String(format: "%.2f", totalTime)) seconds")
    }
    
    /// Initialize population on the GPU
    private func initializePopulationGPU() {
        let startTime = Date()
        
        // Use GPU to generate and evaluate initial population
        if let (sequences, objectives) = metalAccelerator.generateAndEvaluateRandomSolutions(count: populationSize) {
            print("GPU generated \(sequences.count) solutions")
            
            // Convert GPU solutions to chromosomes with their objectives
            var initialPopulation: [FlowshopChromosome] = []
            initialPopulation.reserveCapacity(populationSize)
            
            // Convert UInt32 sequences to Int for FlowshopChromosome and use GPU-computed objectives
            for i in 0..<sequences.count {
                let jobSequence = sequences[i].map { Int($0) }
                let (makespan, tardiness) = objectives[i]
                
                // Create chromosome with precomputed objectives
                let chromosome = FlowshopChromosome(
                    jobSequence: jobSequence,
                    processingTimes: problemData.processingTimes,
                    priorities: problemData.priorities,
                    deadlines: problemData.deadlines,
                    precomputedObjectives: (makespan: Double(makespan), tardiness: Double(tardiness))
                )
                initialPopulation.append(chromosome)
            }
            
            // Store the population
            population = initialPopulation
            
            // Add initial population to the Pareto front
            let frontStart = Date()
            for solution in initialPopulation {
                _ = paretoFront.add(solution)
            }
            let frontTime = Date().timeIntervalSince(frontStart)
            print("Pareto front initialization completed in \(String(format: "%.3f", frontTime))s")
            
            let gpuTime = Date().timeIntervalSince(startTime)
            print("GPU initialization completed in \(String(format: "%.2f", gpuTime)) seconds")
            metrics.gpuTimeElapsed += gpuTime
        } else {
            print("GPU initialization failed, falling back to CPU")
            initializePopulationCPU()
        }
    }
    
    /// Evolve the population for a specified number of generations
    public func evolve(generations: Int, progressHandler: ((Metrics) -> Void)? = nil) -> [FlowshopChromosome] {
        let startTime = Date()
        metrics = Metrics()
        
        // Initialize population if empty
        if population.isEmpty {
            initializePopulation()
        }
        
        // Setup timers for progress reporting
        let progressInterval: TimeInterval = 1.0 // Report progress every second
        var lastProgressTime = Date()
        
        // Main evolution loop
        for g in 0..<generations {
            autoreleasepool {
                // Evolve one generation
                evolveOneGeneration()
                
                // Update metrics
                metrics.iterations = g + 1
                metrics.solutionsEvaluated += populationSize
                metrics.paretoFrontSize = paretoFront.count
                metrics.timeElapsed = Date().timeIntervalSince(startTime)
                
                // Report progress at intervals
                if let progressHandler = progressHandler,
                   Date().timeIntervalSince(lastProgressTime) >= progressInterval {
                    progressHandler(metrics)
                    lastProgressTime = Date()
                }
            }
        }
        
        // Final metrics update
        metrics.timeElapsed = Date().timeIntervalSince(startTime)
        progressHandler?(metrics)
        
        return paretoFront.solutions
    }
    
    /// Evolve a single generation
    private func evolveOneGeneration() {
        let generationStart = Date()
        
        if useGPU {
            print("Starting GPU evolution cycle...")
            let gpuStart = Date()
            evolveOneGenerationGPU()
            metrics.gpuTimeElapsed += Date().timeIntervalSince(gpuStart)
        } else {
            print("Starting CPU evolution cycle...")
            let cpuStart = Date()
            evolveOneGenerationCPU()
            metrics.cpuTimeElapsed += Date().timeIntervalSince(cpuStart)
        }
        
        metrics.timeElapsed += Date().timeIntervalSince(generationStart)
    }
    
    /// Evolve one generation using GPU acceleration
    private func evolveOneGenerationGPU() {
        let start = Date()
        
        // Apply elitism - add best solutions directly to next generation
        var offspring: [FlowshopChromosome] = []
        if elitismCount > 0 {
            let elites = Array(paretoFront.solutions.prefix(min(elitismCount, paretoFront.count)))
            offspring.append(contentsOf: elites)
        }
        
        // Create remainder of offspring through selection and crossover
        let currentPopulation = population
        while offspring.count < populationSize {
            // Tournament selection for parents
            let parent1 = tournamentSelection(from: currentPopulation)
            let parent2 = tournamentSelection(from: currentPopulation)
            
            // Apply crossover with some probability
            if Double.random(in: 0...1) < crossoverRate {
                offspring.append(parent1.crossover(with: parent2))
            } else {
                // No crossover, add one parent randomly
                offspring.append(Bool.random() ? parent1 : parent2)
            }
        }
        
        // Apply mutation to non-elite offspring
        var mutatedOffspring: [FlowshopChromosome] = []
        mutatedOffspring.reserveCapacity(populationSize)
        
        // First, add elites without mutation
        let eliteCount = min(elitismCount, offspring.count)
        mutatedOffspring.append(contentsOf: offspring.prefix(eliteCount))
        
        // Apply mutation to non-elite offspring
        for i in eliteCount..<offspring.count {
            if Double.random(in: 0...1) < 0.5 { // Increased from 0.2 to 0.5 for more diversity
                mutatedOffspring.append(offspring[i].mutate(mutationRate: mutationRate))
            } else {
                mutatedOffspring.append(offspring[i])
            }
        }
        
        // Convert chromosomes to job sequences for GPU evaluation
        let sequencesToEvaluate = mutatedOffspring.map { $0.jobSequence.map { UInt32($0) } }
        
        print("Evaluating \(sequencesToEvaluate.count) solutions on GPU...")
        let evalStart = Date()
        
        // Evaluate all solutions on GPU
        if let objectives = metalAccelerator.evaluateSolutions(jobSequences: sequencesToEvaluate) {
            let evalTime = Date().timeIntervalSince(evalStart)
            print("GPU evaluation completed in \(String(format: "%.3f", evalTime))s")
            
            // Create new chromosomes with GPU-computed objectives
            var evaluatedOffspring: [FlowshopChromosome] = []
            evaluatedOffspring.reserveCapacity(mutatedOffspring.count)
            
            for (i, chromosome) in mutatedOffspring.enumerated() {
                let (makespan, tardiness) = objectives[i]
                let evaluated = chromosome.withObjectives(
                    makespan: Double(makespan),
                    tardiness: Double(tardiness)
                )
                evaluatedOffspring.append(evaluated)
            }
            
            // Replace population and update Pareto front
            population = evaluatedOffspring
            
            let frontStart = Date()
            for solution in evaluatedOffspring {
                _ = paretoFront.add(solution)
            }
            let frontTime = Date().timeIntervalSince(frontStart)
            print("Pareto front update completed in \(String(format: "%.3f", frontTime))s")
        } else {
            print("GPU evaluation failed, falling back to CPU")
            evolveOneGenerationCPU()
        }
        
        let elapsed = Date().timeIntervalSince(start)
        print("GPU generation took \(String(format: "%.3f", elapsed))s")
    }
    
    /// Evolve one generation using CPU
    private func evolveOneGenerationCPU() {
        // Create new offspring population
        let batchSize = min(500, populationSize)
        
        // Apply elitism - add best solutions directly to next generation
        var offspring: [FlowshopChromosome] = []
        if elitismCount > 0 {
            let currentFrontSolutions = paretoFront.solutions
            let elites = Array(currentFrontSolutions.prefix(min(elitismCount, currentFrontSolutions.count)))
            offspring.append(contentsOf: elites)
        }
        
        // Safe copy of current population for selection
        let currentPopulation = population
        
        // Create remainder of offspring through selection and crossover
        while offspring.count < populationSize {
            autoreleasepool {
                let remainingToCreate = min(batchSize, populationSize - offspring.count)
                
                // Generate a batch of offspring
                for _ in 0..<remainingToCreate {
                    // Tournament selection for parents
                    let parent1 = tournamentSelection(from: currentPopulation)
                    let parent2 = tournamentSelection(from: currentPopulation)
                    
                    // Apply crossover with some probability
                    if Double.random(in: 0...1) < crossoverRate {
                        offspring.append(parent1.crossover(with: parent2))
                    } else {
                        // No crossover, add one parent randomly
                        offspring.append(Bool.random() ? parent1 : parent2)
                    }
                }
            }
        }
        
        // Ensure we have exactly populationSize offspring
        if offspring.count > populationSize {
            offspring = Array(offspring.prefix(populationSize))
        }
        
        // Apply mutation
        var mutatedOffspring: [FlowshopChromosome] = []
        mutatedOffspring.reserveCapacity(populationSize)
        
        // First, add elites without mutation
        let eliteCount = min(elitismCount, offspring.count)
        mutatedOffspring.append(contentsOf: offspring.prefix(eliteCount))
        
        // Apply mutation to non-elite offspring
        for i in eliteCount..<offspring.count {
            if Double.random(in: 0...1) < 0.5 { // Increased from 0.2 to 0.5 for more diversity
                mutatedOffspring.append(offspring[i].mutate(mutationRate: mutationRate))
            } else {
                mutatedOffspring.append(offspring[i])
            }
        }
        
        // Replace the old population
        population = mutatedOffspring
        
        // Update the Pareto front
        for solution in mutatedOffspring {
            _ = paretoFront.add(solution)
        }
    }
    
    /// Tournament selection
    private func tournamentSelection(from population: [FlowshopChromosome], tournamentSize: Int = 3) -> FlowshopChromosome {
        // Handle edge cases
        guard !population.isEmpty else {
            return FlowshopChromosome.random(with: problemData) // Generate random if empty
        }
        
        if population.count == 1 {
            return population[0]
        }
        
        // Select random individuals for tournament
        var candidates: [FlowshopChromosome] = []
        let actualTournamentSize = min(tournamentSize, population.count)
        
        for _ in 0..<actualTournamentSize {
            let randomIndex = Int.random(in: 0..<population.count)
            candidates.append(population[randomIndex])
        }
        
        // Find the best candidate (non-dominated by others)
        var bestCandidate = candidates[0]
        
        for candidate in candidates.dropFirst() {
            if candidate.dominates(bestCandidate) {
                bestCandidate = candidate
            }
            // If neither dominates, randomly choose
            else if !bestCandidate.dominates(candidate) && Bool.random() {
                bestCandidate = candidate
            }
        }
        
        return bestCandidate
    }
    
    /// Get current metrics
    public func getMetrics() -> Metrics {
        return metrics
    }
}