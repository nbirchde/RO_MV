import Foundation
import Dispatch

/// Protocol for generating solutions using different heuristic strategies
public protocol SolutionGenerator {
    /// The type of solution being generated
    associatedtype SolutionType: Solution
    
    /// Generate a batch of solutions
    /// - Parameter count: Number of solutions to generate
    /// - Returns: Array of generated solutions
    func generateBatch(count: Int) -> [SolutionType]
    
    /// Generate a single solution
    /// - Returns: A newly generated solution
    func generateSolution() -> SolutionType
}

/// Performance metrics for optimization algorithms
public struct PerformanceMetrics {
    /// Time elapsed in seconds
    public var timeElapsed: TimeInterval = 0
    
    /// Number of solutions evaluated
    public var solutionsEvaluated: Int = 0
    
    /// Number of solutions added to Pareto front
    public var solutionsAdded: Int = 0
    
    /// Iterations completed
    public var iterations: Int = 0
    
    /// Current size of Pareto front
    public var paretoFrontSize: Int = 0
    
    public init() {}
    
    /// Calculate solutions per second
    public var solutionsPerSecond: Double {
        guard timeElapsed > 0 else { return 0 }
        return Double(solutionsEvaluated) / timeElapsed
    }
    
    /// Print metrics to console
    public func printSummary() {
        print("===== Performance Metrics =====")
        print("Time elapsed: \(String(format: "%.2f", timeElapsed)) seconds")
        print("Solutions evaluated: \(solutionsEvaluated)")
        print("Solutions in Pareto front: \(paretoFrontSize)")
        print("Iterations completed: \(iterations)")
        print("Solutions per second: \(String(format: "%.2f", solutionsPerSecond))")
        print("=============================")
    }
}

/// Base optimizer that combines a solution generator with a Pareto front
open class Optimizer<Generator: SolutionGenerator> {
    /// The solution generator
    public let generator: Generator
    
    /// The Pareto front to store non-dominated solutions
    public let paretoFront = ParetoFront<Generator.SolutionType>()
    
    /// Performance metrics for this optimization run
    public var metrics = PerformanceMetrics()
    
    /// Create a new optimizer with the given solution generator
    public init(generator: Generator) {
        self.generator = generator
    }
    
    /// Run the optimization process
    /// - Parameters:
    ///   - iterations: Number of iterations to run
    ///   - batchSize: Number of solutions to generate per iteration
    ///   - progressHandler: Optional handler to report progress
    /// - Returns: Array of non-dominated solutions (Pareto front)
    public func optimize(
        iterations: Int,
        batchSize: Int,
        progressHandler: ((PerformanceMetrics) -> Void)? = nil
    ) -> [Generator.SolutionType] {
        let startTime = Date()
        metrics = PerformanceMetrics()
        
        // Setup timers for progress reporting
        let progressInterval: TimeInterval = 1.0 // Report progress every second
        var lastProgressTime = Date()
        
        // Run optimization iterations
        for i in 0..<iterations {
            autoreleasepool {
                // Generate solutions in parallel using the generator
                let solutions = generator.generateBatch(count: batchSize)
                metrics.solutionsEvaluated += solutions.count
                
                // Add solutions to Pareto front and track how many were added
                let added = paretoFront.addBatch(solutions)
                metrics.solutionsAdded += added.count
                
                // Update metrics
                metrics.iterations = i + 1
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
        metrics.paretoFrontSize = paretoFront.count
        progressHandler?(metrics)
        
        return paretoFront.solutions
    }
}