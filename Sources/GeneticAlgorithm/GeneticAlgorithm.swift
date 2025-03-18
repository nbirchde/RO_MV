import Foundation
import Dispatch
import OptimizationCore

/// Protocol for entities that can be evolved in a genetic algorithm
public protocol Chromosome: Solution {
    /// Create a new chromosome by crossing this chromosome with another
    /// - Parameter other: The other parent chromosome
    /// - Returns: A new child chromosome
    func crossover(with other: Self) -> Self
    
    /// Create a mutated version of this chromosome
    /// - Parameter mutationRate: Probability of mutation for each gene
    /// - Returns: A new mutated chromosome
    func mutate(mutationRate: Double) -> Self
    
    /// Create a random chromosome (for initial population)
    static func random() -> Self
}

/// Performance metrics for genetic algorithm execution
public struct PerformanceMetrics {
    public var iterations: Int = 0
    public var solutionsEvaluated: Int = 0
    public var paretoFrontSize: Int = 0
    public var timeElapsed: TimeInterval = 0
}

/// The genetic algorithm optimizer that can evolve solutions
public class GeneticAlgorithm<T: Chromosome> {
    // Configuration parameters
    public var populationSize: Int
    public var mutationRate: Double
    public var crossoverRate: Double
    public var elitismCount: Int
    
    // Current population
    private var population: [T] = []
    
    // Pareto front of non-dominated solutions
    public let paretoFront = ParetoFront<T>()
    
    // Performance metrics
    public var metrics = PerformanceMetrics()
    
    /// Initialize a new genetic algorithm with the given parameters
    /// - Parameters:
    ///   - populationSize: Number of chromosomes in each generation
    ///   - mutationRate: Probability of mutation for each gene (0.0-1.0)
    ///   - crossoverRate: Probability of crossover (0.0-1.0)
    ///   - elitismCount: Number of elite chromosomes to preserve unchanged
    public init(populationSize: Int = 100, 
                mutationRate: Double = 0.01,
                crossoverRate: Double = 0.7, 
                elitismCount: Int = 5) {
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.elitismCount = min(elitismCount, populationSize / 2) // Prevent elitism count from being too large
    }
    
    /// Initialize the population with random chromosomes
    public func initializePopulation() {
        print("Initializing population of \(populationSize) chromosomes...")
        
        // Generate in batches to manage memory efficiently
        let batchSize = min(500, populationSize)
        var initialPopulation: [T] = []
        initialPopulation.reserveCapacity(populationSize)
        
        // Generate chromosomes in batches
        for batchStart in stride(from: 0, to: populationSize, by: batchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + batchSize, populationSize)
                let batchCount = batchEnd - batchStart
                
                print("Generating batch \(batchStart)-\(batchEnd)...")
                
                // Generate a batch of random chromosomes
                var batchPopulation: [T] = []
                batchPopulation.reserveCapacity(batchCount)
                
                for _ in 0..<batchCount {
                    batchPopulation.append(T.random())
                }
                
                // Add batch to overall population
                initialPopulation.append(contentsOf: batchPopulation)
            }
        }
        
        // Store the population
        population = initialPopulation
        
        // Add initial population to the Pareto front in batches
        print("Adding initial population to Pareto front...")
        for batchStart in stride(from: 0, to: initialPopulation.count, by: batchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + batchSize, initialPopulation.count)
                let batch = Array(initialPopulation[batchStart..<batchEnd])
                _ = paretoFront.addBatch(batch)
            }
        }
        
        print("Initialization complete. Population size: \(population.count), Pareto front size: \(paretoFront.count)")
    }
    
    /// Evolve the population for a specified number of generations
    /// - Parameters:
    ///   - generations: Number of generations to evolve
    ///   - progressHandler: Optional handler for progress updates
    /// - Returns: The Pareto front of non-dominated solutions
    public func evolve(generations: Int, 
                       progressHandler: ((PerformanceMetrics) -> Void)? = nil) -> [T] {
        let startTime = Date()
        metrics = PerformanceMetrics()
        
        // Initialize if population is empty
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
        // Create new offspring population in manageable batches
        let batchSize = min(500, populationSize)
        
        // Get elite solutions from the Pareto front
        var offspring: [T] = []
        offspring.reserveCapacity(populationSize)
        
        // Apply elitism - add best solutions directly to next generation
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
        
        // Apply mutation after crossover
        var mutatedOffspring: [T] = []
        mutatedOffspring.reserveCapacity(populationSize)
        
        // First, add elites without mutation
        let eliteCount = min(elitismCount, offspring.count)
        mutatedOffspring.append(contentsOf: offspring.prefix(eliteCount))
        
        // Apply mutation to non-elite offspring in batches
        for batchStart in stride(from: eliteCount, to: offspring.count, by: batchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + batchSize, offspring.count)
                
                for i in batchStart..<batchEnd {
                    // Each chromosome has a certain chance to be mutated
                    if Double.random(in: 0...1) < 0.2 { // 20% chance to apply mutation
                        mutatedOffspring.append(offspring[i].mutate(mutationRate: mutationRate))
                    } else {
                        mutatedOffspring.append(offspring[i])
                    }
                }
            }
        }
        
        // Replace the old population with the new one
        population = mutatedOffspring
        
        // Update the Pareto front in batches
        for batchStart in stride(from: 0, to: mutatedOffspring.count, by: batchSize) {
            autoreleasepool {
                let batchEnd = min(batchStart + batchSize, mutatedOffspring.count)
                let batch = Array(mutatedOffspring[batchStart..<batchEnd])
                _ = paretoFront.addBatch(batch)
            }
        }
    }
    
    /// Select a chromosome using tournament selection
    /// - Returns: The selected chromosome
    private func tournamentSelection(from population: [T], tournamentSize: Int = 3) -> T {
        // Handle edge cases
        guard !population.isEmpty else {
            return T.random() // Generate random if empty
        }
        
        if population.count == 1 {
            return population[0]
        }
        
        // Select random individuals for tournament
        var candidates: [T] = []
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
}