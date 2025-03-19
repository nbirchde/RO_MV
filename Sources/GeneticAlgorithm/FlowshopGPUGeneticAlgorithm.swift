import Foundation
import OptimizationCore
import ObjectiveC

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
    private let metalAccelerator: Any? // Will be MetalAccelerator.shared at runtime
    
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
        
        // Initialize Metal accelerator dynamically to avoid circular dependency
        let acceleratorClass = NSClassFromString("ParetoOptimization.FlowshopMetalAccelerator") as? NSObject.Type
        let shared = acceleratorClass?.value(forKey: "shared")
        
        if let accelerator = shared as? AnyObject,
           let isAvailable = accelerator.value(forKey: "isMetalAvailable") as? Bool, 
           isAvailable {
            self.metalAccelerator = accelerator
            
            // Initialize problem data on GPU
            let selector = NSSelectorFromString("initializeProblemData:")
            if (accelerator as AnyObject).responds(to: selector) {
                let initMethod = (accelerator as AnyObject).method(for: selector)
                typealias InitFunction = (AnyObject, Selector, FlowshopProblemData) -> Bool
                let initImpl = unsafeBitCast(initMethod, to: InitFunction.self)
                self.useGPU = initImpl(accelerator as AnyObject, selector, problemData)
            } else {
                self.useGPU = false
            }
            
            print("GPU acceleration \(useGPU ? "enabled" : "disabled")")
        } else {
            self.metalAccelerator = nil
            self.useGPU = false
            print("GPU acceleration not available")
        }
    }
    
    /// Initialize the population with random chromosomes
    private func initializePopulation() {
        print("Initializing population of \(populationSize) chromosomes...")
        
        // For now, just use CPU initialization
        initializePopulationCPU()
        
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
        // For now, just use CPU evolution
        evolveOneGenerationCPU()
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
            if Double.random(in: 0...1) < 0.2 { // 20% chance to apply mutation
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