import Foundation
import Dispatch
import simd

/// A collection of non-dominated solutions representing a Pareto frontier
public class ParetoFront<S: Solution> {
    /// The current set of non-dominated solutions
    private var _solutions: Set<S> = []
    
    /// Thread-safe access to the solutions
    private let queue = DispatchQueue(label: "com.optimization.paretofront", attributes: .concurrent)
    
    /// Public accessor for the solutions
    public var solutions: [S] {
        queue.sync { Array(_solutions) }
    }
    
    public init() {}
    
    /// Add a solution to the Pareto front
    /// - Returns: true if the solution was added (non-dominated), false if dominated
    @discardableResult
    public func add(_ solution: S) -> Bool {
        var shouldAdd = true
        var dominatedSolutions: Set<S> = []
        
        queue.sync(flags: .barrier) {
            // Check if this solution is dominated by any existing solution
            for existingSolution in _solutions {
                if existingSolution.dominates(solution) {
                    shouldAdd = false
                    break
                } else if solution.dominates(existingSolution) {
                    dominatedSolutions.insert(existingSolution)
                }
            }
            
            // If not dominated, add it and remove any solutions it dominates
            if shouldAdd {
                _solutions.subtract(dominatedSolutions)
                _solutions.insert(solution)
            }
        }
        
        return shouldAdd
    }
    
    /// Add multiple solutions to the Pareto front in parallel
    /// - Parameter solutions: Array of solutions to evaluate
    /// - Returns: Array of solutions that were added to the front
    public func addBatch(_ solutions: [S]) -> [S] {
        // First, filter out obviously dominated solutions within the new batch
        // This is a parallelized pre-processing step to reduce the number of comparisons
        let filteredBatch = filterDominatedInParallel(solutions)
        
        // Now add the filtered batch to the existing Pareto front
        var addedSolutions: [S] = []
        
        for solution in filteredBatch {
            if add(solution) {
                addedSolutions.append(solution)
            }
        }
        
        return addedSolutions
    }
    
    /// Filter dominated solutions within a batch in parallel using Swift concurrency
    /// - Parameter solutions: The solutions to filter
    /// - Returns: Non-dominated solutions from the batch
    private func filterDominatedInParallel(_ solutions: [S]) -> [S] {
        guard !solutions.isEmpty else { return [] }
        
        // For very small batches, avoid the overhead of parallelism
        if solutions.count <= 10 {
            return filterDominated(solutions)
        }
        
        // Use a simple chunking strategy to divide work among cores
        let processorCount = ProcessInfo.processInfo.activeProcessorCount
        let chunkSize = max(1, solutions.count / processorCount)
        
        var results: [[S]] = Array(repeating: [], count: processorCount)
        let group = DispatchGroup()
        let localQueue = DispatchQueue(label: "com.optimization.local", attributes: .concurrent)
        
        // Process each chunk in parallel
        for i in 0..<processorCount {
            let start = i * chunkSize
            let end = i == processorCount - 1 ? solutions.count : start + chunkSize
            guard start < solutions.count else { continue }
            
            localQueue.async(group: group) {
                let chunk = Array(solutions[start..<end])
                results[i] = self.filterDominated(chunk)
            }
        }
        
        // Wait for all parallel work to complete
        group.wait()
        
        // Combine and filter the results from each chunk
        let combinedResults = results.flatMap { $0 }
        return filterDominated(combinedResults)
    }
    
    /// Filter dominated solutions sequentially
    /// - Parameter solutions: The solutions to filter
    /// - Returns: Non-dominated solutions from the batch
    private func filterDominated(_ solutions: [S]) -> [S] {
        var nonDominated: [S] = []
        
        for solution in solutions {
            var isDominated = false
            var i = 0
            
            // Compare against existing non-dominated solutions
            while i < nonDominated.count {
                if nonDominated[i].dominates(solution) {
                    isDominated = true
                    break
                } else if solution.dominates(nonDominated[i]) {
                    // Remove dominated solution
                    nonDominated.swapAt(i, nonDominated.count - 1)
                    nonDominated.removeLast()
                    // Don't increment i since we need to check the new element at this position
                } else {
                    i += 1
                }
            }
            
            if !isDominated {
                nonDominated.append(solution)
            }
        }
        
        return nonDominated
    }
    
    /// Clear all solutions from the Pareto front
    public func clear() {
        queue.sync(flags: .barrier) {
            _solutions.removeAll()
        }
    }
    
    /// Return the number of solutions in the Pareto front
    public var count: Int {
        queue.sync {
            return _solutions.count
        }
    }
    
    /// Calculate weighted sum of all solutions based on provided weights
    /// - Parameter weights: Array of weights for each criterion
    /// - Returns: The solution with the best (minimum) weighted sum and its score
    public func weightedBest(weights: [Double]) -> (solution: S, score: Double)? {
        queue.sync {
            guard !_solutions.isEmpty else { return nil }
            
            // Calculate weighted sum for each solution in parallel
            var results: [(solution: S, score: Double)] = []
            let group = DispatchGroup()
            let resultsQueue = DispatchQueue(label: "com.optimization.results")
            
            for solution in _solutions {
                DispatchQueue.global().async(group: group) {
                    var score: Double = 0.0
                    
                    // Calculate weighted sum - this is a simplified approach
                    // In a real implementation, you might need to normalize different criteria types
                    for (i, criterion) in solution.criteria.enumerated() {
                        if let numericCriterion = criterion as? NumericCriterion, i < weights.count {
                            let weightedValue = numericCriterion.value * weights[i]
                            score += weightedValue
                        }
                        // Handle other criterion types if needed
                    }
                    
                    resultsQueue.sync {
                        results.append((solution: solution, score: score))
                    }
                }
            }
            
            group.wait()
            
            // Find best (minimum) weighted sum
            return results.min { $0.score < $1.score }
        }
    }
}