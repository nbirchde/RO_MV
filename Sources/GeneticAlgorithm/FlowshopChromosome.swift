import Foundation
import OptimizationCore

/// Represents a flowshop scheduling solution as a permutation of jobs
public struct FlowshopChromosome: Chromosome {
    // Unique identifier for the chromosome
    public let id: UUID
    
    // The permutation of jobs (0-based indices)
    public let jobSequence: [Int]
    
    // Job processing times for each machine
    private let processingTimes: [[Double]]
    
    // Job priorities and deadlines
    private let priorities: [Double]
    private let deadlines: [Double]
    
    // Pre-calculated criteria values
    public let criteria: [CriterionValue]
    
    /// Initialize with a specific job sequence and problem data
    public init(jobSequence: [Int], processingTimes: [[Double]], priorities: [Double], deadlines: [Double]) {
        self.id = UUID()
        self.jobSequence = jobSequence
        self.processingTimes = processingTimes
        self.priorities = priorities
        self.deadlines = deadlines
        
        // Calculate the criteria values for this job sequence
        let (makespan, totalWeightedTardiness) = FlowshopChromosome.calculateObjectives(
            jobSequence: jobSequence,
            processingTimes: processingTimes,
            priorities: priorities,
            deadlines: deadlines
        )
        
        // Create the criteria values - both are to be minimized
        self.criteria = [
            NumericCriterion(makespan, lowerIsBetter: true),         // Makespan
            NumericCriterion(totalWeightedTardiness, lowerIsBetter: true)  // Total weighted tardiness
        ]
    }
    
    /// Create a random chromosome with the provided problem data
    public static func random(numJobs: Int, processingTimes: [[Double]], priorities: [Double], deadlines: [Double]) -> FlowshopChromosome {
        // Generate a random permutation of jobs
        var jobIndices = Array(0..<numJobs)
        jobIndices.shuffle()
        
        return FlowshopChromosome(
            jobSequence: jobIndices,
            processingTimes: processingTimes,
            priorities: priorities,
            deadlines: deadlines
        )
    }
    
    /// Create a random chromosome (for initial population) - this is required by the Chromosome protocol
    public static func random() -> FlowshopChromosome {
        fatalError("Use random(numJobs:processingTimes:priorities:deadlines:) instead")
    }
    
    /// Calculate the makespan and total weighted tardiness for a given job sequence
    private static func calculateObjectives(
        jobSequence: [Int],
        processingTimes: [[Double]],
        priorities: [Double],
        deadlines: [Double]
    ) -> (makespan: Double, totalWeightedTardiness: Double) {
        let numJobs = jobSequence.count
        let numMachines = processingTimes[0].count
        
        // Initialize completion times for all jobs on all machines
        var completionTimes = Array(repeating: Array(repeating: 0.0, count: numMachines), count: numJobs)
        
        // Calculate completion time for first job in the sequence on first machine
        let firstJob = jobSequence[0]
        completionTimes[0][0] = processingTimes[firstJob][0]
        
        // Calculate completion times for first job on all machines
        for m in 1..<numMachines {
            completionTimes[0][m] = completionTimes[0][m-1] + processingTimes[firstJob][m]
        }
        
        // Calculate completion times for all remaining jobs
        for j in 1..<numJobs {
            let job = jobSequence[j]
            
            // First machine depends only on previous job's completion time on first machine
            completionTimes[j][0] = completionTimes[j-1][0] + processingTimes[job][0]
            
            // For other machines, we need to consider both the completion of the previous job on this machine
            // and the completion of this job on the previous machine
            for m in 1..<numMachines {
                completionTimes[j][m] = max(completionTimes[j][m-1], completionTimes[j-1][m]) + processingTimes[job][m]
            }
        }
        
        // Makespan is the completion time of the last job on the last machine
        let makespan = completionTimes[numJobs-1][numMachines-1]
        
        // Calculate total weighted tardiness
        var totalWeightedTardiness = 0.0
        
        for j in 0..<numJobs {
            let job = jobSequence[j]
            let completionTime = completionTimes[j][numMachines-1]
            let tardiness = max(0.0, completionTime - deadlines[job])
            let weightedTardiness = tardiness * priorities[job]
            totalWeightedTardiness += weightedTardiness
        }
        
        return (makespan, totalWeightedTardiness)
    }
    
    /// Create a new chromosome by crossing this chromosome with another
    public func crossover(with other: FlowshopChromosome) -> FlowshopChromosome {
        // Implement Order Crossover (OX) - a common crossover for permutation problems
        let jobCount = jobSequence.count
        
        // Select two random crossover points
        let crossPoint1 = Int.random(in: 0..<jobCount)
        let crossPoint2 = Int.random(in: 0..<jobCount)
        
        // Ensure point1 <= point2
        let start = min(crossPoint1, crossPoint2)
        let end = max(crossPoint1, crossPoint2)
        
        // Initialize offspring with placeholder values (-1)
        var offspring = Array(repeating: -1, count: jobCount)
        
        // Copy the segment from parent1 (this chromosome) between the crossover points
        for i in start...end {
            offspring[i] = jobSequence[i]
        }
        
        // Fill the remaining positions with the values from parent2 (other chromosome)
        // that are not already in the offspring
        var currentPos = (end + 1) % jobCount
        
        for job in other.jobSequence {
            // Only consider jobs not already in offspring
            if !offspring.contains(job) {
                offspring[currentPos] = job
                currentPos = (currentPos + 1) % jobCount
                
                // If we've wrapped around to the start segment, keep going until we find an empty spot
                while currentPos >= start && currentPos <= end {
                    currentPos = (currentPos + 1) % jobCount
                }
            }
        }
        
        // Create a new chromosome with the offspring sequence
        return FlowshopChromosome(
            jobSequence: offspring,
            processingTimes: processingTimes,
            priorities: priorities,
            deadlines: deadlines
        )
    }
    
    /// Create a mutated version of this chromosome
    public func mutate(mutationRate: Double) -> FlowshopChromosome {
        var mutated = jobSequence
        
        // Apply swap mutation - randomly swap jobs with probability mutationRate
        for i in 0..<jobSequence.count {
            if Double.random(in: 0...1) < mutationRate {
                let j = Int.random(in: 0..<jobSequence.count)
                mutated.swapAt(i, j)
            }
        }
        
        // Return the mutated chromosome
        return FlowshopChromosome(
            jobSequence: mutated,
            processingTimes: processingTimes,
            priorities: priorities,
            deadlines: deadlines
        )
    }
    
    // Hashable implementation
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    // Equatable implementation
    public static func == (lhs: FlowshopChromosome, rhs: FlowshopChromosome) -> Bool {
        return lhs.id == rhs.id
    }
    
    /// Get a description of this solution
    public func description() -> String {
        let makespan = (criteria[0] as! NumericCriterion).value
        let tardiness = (criteria[1] as! NumericCriterion).value
        
        return "Makespan: \(String(format: "%.1f", makespan)), " +
               "Weighted Tardiness: \(String(format: "%.1f", tardiness)), " +
               "Jobs: \(jobSequence.prefix(5))..."
    }
}

extension FlowshopChromosome {
    /// Create a random chromosome with the provided problem data using a static factory method
    public static func random(with problemData: FlowshopProblemData) -> FlowshopChromosome {
        return random(
            numJobs: problemData.processingTimes.count,
            processingTimes: problemData.processingTimes,
            priorities: problemData.priorities,
            deadlines: problemData.deadlines
        )
    }
}

/// Container for flowshop problem data
public struct FlowshopProblemData {
    public let processingTimes: [[Double]]
    public let priorities: [Double]
    public let deadlines: [Double]
    public let numJobs: Int
    public let numMachines: Int
    
    public init(processingTimes: [[Double]], priorities: [Double], deadlines: [Double]) {
        self.processingTimes = processingTimes
        self.priorities = priorities
        self.deadlines = deadlines
        self.numJobs = processingTimes.count
        self.numMachines = processingTimes[0].count
    }
    
    /// Load problem data from a CSV file
    public static func loadFromCSV(filePath: String) -> FlowshopProblemData? {
        print("Attempting to load file from path: \(filePath)")
        
        // Check if the file exists first
        guard FileManager.default.fileExists(atPath: filePath) else {
            print("Error: File does not exist at path \(filePath)")
            return nil
        }
        
        do {
            let fileContents = try String(contentsOfFile: filePath, encoding: .utf8)
            print("Successfully read file with \(fileContents.count) characters")
            
            let lines = fileContents.components(separatedBy: .newlines)
                .filter { !$0.isEmpty }
            
            print("Found \(lines.count) non-empty lines in the file")
            
            // Initialize data arrays
            let expectedNumJobs = lines.count
            var priorities: [Double] = Array(repeating: 0.0, count: expectedNumJobs)
            var deadlines: [Double] = Array(repeating: 0.0, count: expectedNumJobs)
            var processingTimes: [[Double]] = Array(repeating: Array(repeating: 0.0, count: 10), count: expectedNumJobs)
            
            // Track how many valid lines we've processed
            var validLinesCount = 0
            
            for (i, line) in lines.enumerated() {
                let values = line.components(separatedBy: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .compactMap { Double($0) }
                
                // Expected format: [priority, deadline, proc_time_m1, proc_time_m2, ..., proc_time_m10]
                if values.count >= 12 {  // 2 columns + 10 machines
                    priorities[validLinesCount] = values[0]
                    deadlines[validLinesCount] = values[1]
                    
                    // Extract the 10 processing times for each machine
                    for m in 0..<10 {
                        processingTimes[validLinesCount][m] = values[m + 2]
                    }
                    
                    validLinesCount += 1
                } else {
                    print("Warning: Line \(i+1) has incorrect format: \(values.count) values found, expected at least 12")
                }
            }
            
            // Trim arrays if we didn't process all lines
            if validLinesCount < expectedNumJobs {
                print("Trimming arrays from \(expectedNumJobs) to \(validLinesCount) valid jobs")
                priorities.removeSubrange(validLinesCount..<expectedNumJobs)
                deadlines.removeSubrange(validLinesCount..<expectedNumJobs)
                processingTimes.removeSubrange(validLinesCount..<expectedNumJobs)
            }
            
            // Validate that we have consistent data
            if !priorities.isEmpty && priorities.count == deadlines.count && priorities.count == processingTimes.count {
                print("Successfully loaded \(priorities.count) jobs with \(processingTimes[0].count) machines")
                
                // Print some sample data for verification
                let sampleSize = min(3, priorities.count)
                for i in 0..<sampleSize {
                    print("Job \(i): Priority=\(priorities[i]), Deadline=\(deadlines[i])")
                    print("  Processing times: \(processingTimes[i])")
                }
                
                return FlowshopProblemData(
                    processingTimes: processingTimes,
                    priorities: priorities,
                    deadlines: deadlines
                )
            } else {
                print("Error: Inconsistent data - priorities: \(priorities.count), deadlines: \(deadlines.count), processingTimes: \(processingTimes.count)")
                return nil
            }
        } catch {
            print("Error loading problem data: \(error)")
            return nil
        }
    }
    
    /// Create a small test problem
    public static func createTestProblem(numJobs: Int = 5, numMachines: Int = 3) -> FlowshopProblemData {
        var priorities: [Double] = []
        var deadlines: [Double] = []
        var processingTimes: [[Double]] = []
        
        // Generate random job data
        for _ in 0..<numJobs {
            // Random priority between 1-10
            priorities.append(Double(Int.random(in: 1...10)))
            
            // Random processing times between 1-20 for each machine
            var jobTimes: [Double] = []
            for _ in 0..<numMachines {
                jobTimes.append(Double(Int.random(in: 1...20)))
            }
            processingTimes.append(jobTimes)
            
            // Set deadline to a reasonable value (sum of processing times * random factor)
            let sumProcessingTimes = jobTimes.reduce(0, +)
            deadlines.append(sumProcessingTimes * Double.random(in: 1.0...2.0))
        }
        
        return FlowshopProblemData(
            processingTimes: processingTimes,
            priorities: priorities,
            deadlines: deadlines
        )
    }
}