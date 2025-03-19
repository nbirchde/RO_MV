import Foundation
import Metal
import OptimizationCore

/// Class for GPU-accelerated flowshop scheduling using Metal
public class FlowshopMetalAccelerator {
    // Singleton instance
    public static let shared = FlowshopMetalAccelerator()
    
    // Metal properties
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    public private(set) var isMetalAvailable: Bool
    
    // Problem data
    private var numJobs: UInt32 = 0
    private var numMachines: UInt32 = 10  // Default to 10 machines for flowshop
    
    // Pipeline states
    private var evaluationPipelineState: MTLComputePipelineState?
    private var generateAndEvaluatePipelineState: MTLComputePipelineState?
    private var dominancePipelineState: MTLComputePipelineState?
    
    // GPU buffers for problem data (long-lived)
    private var processingTimesBuffer: MTLBuffer?
    private var prioritiesBuffer: MTLBuffer?
    private var deadlinesBuffer: MTLBuffer?
    
    // Initialize Metal setup
    private init() {
        print("Initializing Metal Accelerator...")
        
        // Try to get default Metal device
        if let device = MTLCreateSystemDefaultDevice() {
            self.device = device
            print("✅ Found Metal device: \(device.name)")
            
            // Create command queue
            self.commandQueue = device.makeCommandQueue()
            self.isMetalAvailable = (self.commandQueue != nil)
            
            if self.isMetalAvailable {
                print("✅ Created Metal command queue")
            } else {
                print("❌ Failed to create Metal command queue")
            }
            
            // Initialize compute pipelines
            print("Setting up Metal compute pipelines...")
            setupComputePipelines()
            
            if self.evaluationPipelineState != nil {
                print("✅ Created evaluation pipeline")
            } else {
                print("❌ Failed to create evaluation pipeline")
            }
            
            if self.generateAndEvaluatePipelineState != nil {
                print("✅ Created generation pipeline")
            } else {
                print("❌ Failed to create generation pipeline")
            }
        } else {
            print("❌ No Metal device available")
            self.device = nil
            self.commandQueue = nil
            self.isMetalAvailable = false
        }
    }
    
    /// Get information about the Metal device being used
    public func deviceInfo() -> String {
        guard let device = device else {
            return "No Metal device available"
        }
        
        var info = "GPU: \(device.name)"
        info += "\nMax threads per threadgroup: \(device.maxThreadsPerThreadgroup.width)x\(device.maxThreadsPerThreadgroup.height)"
        info += "\nRecommended max working set size: \(Double(device.recommendedMaxWorkingSetSize) / (1024.0 * 1024.0)) MB"
        info += "\nHas unified memory: \(device.hasUnifiedMemory)"
        
        return info
    }
    
    /// Set up Metal compute pipelines
    private func setupComputePipelines() {
        guard let device = device else { return }
        
        // Metal shader source code
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        // Xorshift RNG for GPU
        uint xorshift(thread uint* state) {
            uint x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            return x;
        }
        
        // Fisher-Yates shuffle implementation for GPU
        void shuffle_sequence(
            thread uint* sequence,
            uint length,
            thread uint* rng_state
        ) {
            // Initialize sequence with 0..length-1
            for (uint i = 0; i < length; i++) {
                sequence[i] = i;
            }
            
            // Shuffle
            for (uint i = length - 1; i > 0; i--) {
                uint j = xorshift(rng_state) % (i + 1);
                // Swap
                uint temp = sequence[i];
                sequence[i] = sequence[j];
                sequence[j] = temp;
            }
        }
        
        kernel void generate_and_evaluate_flowshop(
            device uint* outputSequences,             // Output buffer for generated sequences
            device const float* processingTimes,      // Processing times matrix (flattened)
            device const float* priorities,           // Job priorities
            device const float* deadlines,            // Job deadlines
            device float2* objectives,                // Output: (makespan, tardiness) for each solution
            constant uint& numJobs,                   // Number of jobs
            constant uint& numMachines,               // Number of machines
            constant uint& numSolutions,              // Number of solutions to generate
            constant uint& seed,                      // Random seed
            uint id [[thread_position_in_grid]]       // Thread ID
        ) {
            if (id >= numSolutions) return;
            
            // Initialize RNG state for this thread
            uint rng_state = seed + id;
            
            // Get pointer to this thread's output sequence
            device uint* sequence = outputSequences + (id * numJobs);
            
            // Generate random permutation using thread-local array
            thread uint local_sequence[200];  // Max 200 jobs
            shuffle_sequence(local_sequence, numJobs, &rng_state);
            
            // Copy to output buffer
            for (uint i = 0; i < numJobs; i++) {
                sequence[i] = local_sequence[i];
            }
            
            // Evaluate the sequence (reuse evaluation code)
            thread float machineCompletionTimes[10];
            thread float jobCompletionTimes[200];
            
            // Initialize completion times
            for (uint m = 0; m < numMachines; m++) {
                machineCompletionTimes[m] = 0.0;
            }
            
            // Process each job through machines
            for (uint jobIdx = 0; jobIdx < numJobs; jobIdx++) {
                uint job = local_sequence[jobIdx];
                
                for (uint m = 0; m < numMachines; m++) {
                    uint procTimeIdx = job * numMachines + m;
                    float processingTime = processingTimes[procTimeIdx];
                    
                    if (m == 0) {
                        machineCompletionTimes[m] += processingTime;
                    } else {
                        machineCompletionTimes[m] = max(machineCompletionTimes[m], machineCompletionTimes[m-1]) + processingTime;
                    }
                }
                
                jobCompletionTimes[job] = machineCompletionTimes[numMachines - 1];
            }
            
            // Calculate objectives
            float makespan = machineCompletionTimes[numMachines - 1];
            float totalWeightedTardiness = 0.0;
            
            for (uint job = 0; job < numJobs; job++) {
                float completionTime = jobCompletionTimes[job];
                float tardiness = max(0.0f, completionTime - deadlines[job]);
                totalWeightedTardiness += tardiness * priorities[job];
            }
            
            objectives[id] = float2(makespan, totalWeightedTardiness);
        }
        
        // Keep existing evaluate_flowshop kernel
        kernel void evaluate_flowshop(
            device const uint* jobSequences,          // Array of job sequences to evaluate
            device const float* processingTimes,      // Processing times matrix (flattened)
            device const float* priorities,           // Job priorities
            device const float* deadlines,            // Job deadlines
            device float2* objectives,                // Output: (makespan, tardiness) for each solution
            constant uint& numJobs,                   // Number of jobs
            constant uint& numMachines,               // Number of machines
            constant uint& numSolutions,              // Number of solutions to evaluate
            uint id [[thread_position_in_grid]]       // Thread ID
        ) {
            // Check if this thread should evaluate a solution
            if (id >= numSolutions) {
                return;
            }
            
            // Get pointer to this thread's job sequence
            device const uint* sequence = jobSequences + (id * numJobs);
            
            // Allocate arrays for completion times on each machine
            thread float machineCompletionTimes[10];  // Assuming max 10 machines
            
            // Initialize machine completion times to 0
            for (uint m = 0; m < numMachines; m++) {
                machineCompletionTimes[m] = 0.0;
            }
            
            // Track job completion times (when each job exits the system)
            thread float jobCompletionTimes[200];    // Assuming max 200 jobs
            
            // Process each job in the sequence through all machines
            for (uint jobIdx = 0; jobIdx < numJobs; jobIdx++) {
                uint job = sequence[jobIdx];  // Get the actual job index from the sequence
                
                // Process the job through each machine
                for (uint m = 0; m < numMachines; m++) {
                    // Calculate the processing time index in the flattened array
                    uint procTimeIdx = job * numMachines + m;
                    float processingTime = processingTimes[procTimeIdx];
                    
                    if (m == 0) {
                        // First machine: just add to current completion time
                        machineCompletionTimes[m] += processingTime;
                    } else {
                        // Other machines: job can start when both previous machine 
                        // finishes this job AND this machine finishes previous job
                        machineCompletionTimes[m] = max(machineCompletionTimes[m], machineCompletionTimes[m-1]) + processingTime;
                    }
                }
                
                // Record the completion time for this job (when it exits the last machine)
                jobCompletionTimes[job] = machineCompletionTimes[numMachines - 1];
            }
            
            // Calculate objectives
            float makespan = machineCompletionTimes[numMachines - 1];  // Time when last job completes
            float totalWeightedTardiness = 0.0;
            
            // Calculate total weighted tardiness
            for (uint job = 0; job < numJobs; job++) {
                float completionTime = jobCompletionTimes[job];
                float deadline = deadlines[job];
                float tardiness = max(0.0f, completionTime - deadline);
                float priority = priorities[job];
                
                totalWeightedTardiness += tardiness * priority;
            }
            
            // Store the result
            objectives[id] = float2(makespan, totalWeightedTardiness);
        }
        """
        
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            
            // Create evaluation pipeline
            if let evaluationFunction = library.makeFunction(name: "evaluate_flowshop") {
                evaluationPipelineState = try device.makeComputePipelineState(function: evaluationFunction)
                print("Successfully created evaluation pipeline state")
            }
            
            // Create generate and evaluate pipeline
            if let generateAndEvaluateFunction = library.makeFunction(name: "generate_and_evaluate_flowshop") {
                generateAndEvaluatePipelineState = try device.makeComputePipelineState(function: generateAndEvaluateFunction)
                print("Successfully created generate and evaluate pipeline state")
            }
        } catch {
            print("Failed to create Metal library: \(error)")
        }
    }
    
    /// Load problem data to the GPU (called once)
    public func loadFlowshopProblemData(
        processingTimes: [[Float]],
        priorities: [Float],
        deadlines: [Float]
    ) -> Bool {
        guard isMetalAvailable,
              let device = device else {
            return false
        }
        
        // Validate input dimensions
        guard !processingTimes.isEmpty, 
              !priorities.isEmpty,
              priorities.count == processingTimes.count,
              deadlines.count == processingTimes.count else {
            print("Invalid problem data dimensions")
            return false
        }
        
        // Store problem dimensions
        numJobs = UInt32(processingTimes.count)
        numMachines = UInt32(processingTimes[0].count)
        
        print("Loading flowshop problem data: \(numJobs) jobs, \(numMachines) machines")
        
        // Flatten processing times matrix for GPU buffer
        var flatProcessingTimes = [Float]()
        flatProcessingTimes.reserveCapacity(Int(numJobs * numMachines))
        
        for jobTimes in processingTimes {
            flatProcessingTimes.append(contentsOf: jobTimes)
        }
        
        // Create GPU buffers with storageModeShared for optimal performance on Apple Silicon
        // (This enables zero-copy access between CPU and GPU)
        let flatProcessingTimesSize = MemoryLayout<Float>.size * flatProcessingTimes.count
        let prioritiesSize = MemoryLayout<Float>.size * priorities.count
        let deadlinesSize = MemoryLayout<Float>.size * deadlines.count
        
        // Create buffers with .storageModeShared for unified memory access
        processingTimesBuffer = device.makeBuffer(
            bytes: flatProcessingTimes,
            length: flatProcessingTimesSize,
            options: .storageModeShared
        )
        
        prioritiesBuffer = device.makeBuffer(
            bytes: priorities,
            length: prioritiesSize,
            options: .storageModeShared
        )
        
        deadlinesBuffer = device.makeBuffer(
            bytes: deadlines,
            length: deadlinesSize,
            options: .storageModeShared
        )
        
        return processingTimesBuffer != nil && prioritiesBuffer != nil && deadlinesBuffer != nil
    }
    
    /// Evaluate multiple flowshop solutions in parallel on GPU
    public func evaluateSolutions(
        jobSequences: [[UInt32]]
    ) -> [(makespan: Float, tardiness: Float)]? {
        guard isMetalAvailable,
              let device = device,
              let commandQueue = commandQueue,
              let evaluationPipelineState = evaluationPipelineState,
              let processingTimesBuffer = processingTimesBuffer,
              let prioritiesBuffer = prioritiesBuffer,
              let deadlinesBuffer = deadlinesBuffer else {
            return nil
        }
        
        // Ensure we have solutions to evaluate
        guard !jobSequences.isEmpty else {
            return []
        }
        
        // Ensure all solutions have the correct length
        for sequence in jobSequences {
            guard sequence.count == numJobs else {
                print("Invalid job sequence length: \(sequence.count), expected \(numJobs)")
                return nil
            }
        }
        
        let numSolutions = jobSequences.count
        
        // Flatten job sequences for the GPU buffer
        var flatJobSequences = [UInt32]()
        flatJobSequences.reserveCapacity(numSolutions * Int(numJobs))
        
        for sequence in jobSequences {
            flatJobSequences.append(contentsOf: sequence)
        }
        
        // Create GPU buffers
        let sequencesBufferSize = MemoryLayout<UInt32>.size * flatJobSequences.count
        let objectivesBufferSize = MemoryLayout<SIMD2<Float>>.size * numSolutions
        
        guard let sequencesBuffer = device.makeBuffer(
            bytes: flatJobSequences,
            length: sequencesBufferSize,
            options: .storageModeShared
        ),
        let objectivesBuffer = device.makeBuffer(
            length: objectivesBufferSize,
            options: .storageModeShared
        ) else {
            print("Failed to create Metal buffers")
            return nil
        }
        
        // Number of solutions as UInt32 for the kernel
        var numSolutionsUInt32 = UInt32(numSolutions)
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create command buffer or encoder")
            return nil
        }
        
        // Configure the compute encoder
        computeEncoder.setComputePipelineState(evaluationPipelineState)
        computeEncoder.setBuffer(sequencesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(processingTimesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(prioritiesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(deadlinesBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(objectivesBuffer, offset: 0, index: 4)
        computeEncoder.setBytes(&numJobs, length: MemoryLayout<UInt32>.size, index: 5)
        computeEncoder.setBytes(&numMachines, length: MemoryLayout<UInt32>.size, index: 6)
        computeEncoder.setBytes(&numSolutionsUInt32, length: MemoryLayout<UInt32>.size, index: 7)
        
        // Calculate grid and threadgroup sizes
        let threadExecutionWidth = evaluationPipelineState.threadExecutionWidth
        let maxThreadsPerThreadgroup = min(
            evaluationPipelineState.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth * 2
        )
        
        let threadsPerThreadgroup = MTLSize(
            width: maxThreadsPerThreadgroup,
            height: 1,
            depth: 1
        )
        
        let threadgroupsPerGrid = MTLSize(
            width: (numSolutions + maxThreadsPerThreadgroup - 1) / maxThreadsPerThreadgroup,
            height: 1,
            depth: 1
        )
        
        // Dispatch the compute kernel
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Execute and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read back results from GPU
        var results = [(makespan: Float, tardiness: Float)]()
        let objectivesPtr = objectivesBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: numSolutions)
        
        for i in 0..<numSolutions {
            let objective = objectivesPtr[i]
            results.append((makespan: objective.x, tardiness: objective.y))
        }
        
        return results
    }
    
    /// Generate and evaluate random flowshop solutions directly on GPU
    public func generateAndEvaluateRandomSolutions(
        count: Int,
        seed: UInt32 = UInt32.random(in: 1...UInt32.max)
    ) -> (sequences: [[UInt32]], objectives: [(makespan: Float, tardiness: Float)])? {
        guard isMetalAvailable,
              let device = device,
              let commandQueue = commandQueue,
              let generateAndEvaluatePipelineState = generateAndEvaluatePipelineState,
              let processingTimesBuffer = processingTimesBuffer,
              let prioritiesBuffer = prioritiesBuffer,
              let deadlinesBuffer = deadlinesBuffer else {
            return nil
        }
        
        // Ensure count is valid
        guard count > 0 else {
            return (sequences: [], objectives: [])
        }
        
        // Create GPU buffers
        let sequencesBufferSize = MemoryLayout<UInt32>.size * count * Int(numJobs)
        let objectivesBufferSize = MemoryLayout<SIMD2<Float>>.size * count
        
        guard let sequencesBuffer = device.makeBuffer(
            length: sequencesBufferSize,
            options: .storageModeShared
        ),
        let objectivesBuffer = device.makeBuffer(
            length: objectivesBufferSize,
            options: .storageModeShared
        ) else {
            print("Failed to create Metal buffers")
            return nil
        }
        
        // Number of solutions as UInt32 for the kernel
        var countUInt32 = UInt32(count)
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create command buffer or encoder")
            return nil
        }
        
        // Configure the compute encoder
        computeEncoder.setComputePipelineState(generateAndEvaluatePipelineState)
        computeEncoder.setBuffer(sequencesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(processingTimesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(prioritiesBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(deadlinesBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(objectivesBuffer, offset: 0, index: 4)
        computeEncoder.setBytes(&numJobs, length: MemoryLayout<UInt32>.size, index: 5)
        computeEncoder.setBytes(&numMachines, length: MemoryLayout<UInt32>.size, index: 6)
        computeEncoder.setBytes(&countUInt32, length: MemoryLayout<UInt32>.size, index: 7)
        var mutableSeed = seed
        computeEncoder.setBytes(&mutableSeed, length: MemoryLayout<UInt32>.size, index: 8)
        
        // Calculate grid and threadgroup sizes
        let threadExecutionWidth = generateAndEvaluatePipelineState.threadExecutionWidth
        let maxThreadsPerThreadgroup = min(
            generateAndEvaluatePipelineState.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth * 2
        )
        
        let threadsPerThreadgroup = MTLSize(
            width: maxThreadsPerThreadgroup,
            height: 1,
            depth: 1
        )
        
        let threadgroupsPerGrid = MTLSize(
            width: (count + maxThreadsPerThreadgroup - 1) / maxThreadsPerThreadgroup,
            height: 1,
            depth: 1
        )
        
        // Dispatch the compute kernel
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Execute and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read back sequences and objectives from GPU
        var sequences = [[UInt32]]()
        var objectives = [(makespan: Float, tardiness: Float)]()
        
        let sequencesPtr = sequencesBuffer.contents().bindMemory(to: UInt32.self, capacity: count * Int(numJobs))
        let objectivesPtr = objectivesBuffer.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
        
        // Extract each sequence and its objectives
        for i in 0..<count {
            var sequence = [UInt32]()
            let baseOffset = i * Int(numJobs)
            
            for j in 0..<Int(numJobs) {
                sequence.append(sequencesPtr[baseOffset + j])
            }
            
            let objective = objectivesPtr[i]
            
            sequences.append(sequence)
            objectives.append((makespan: objective.x, tardiness: objective.y))
        }
        
        return (sequences: sequences, objectives: objectives)
    }
    
    /// Check dominance relationships between multiple solutions
    public func checkDominanceRelationships(
        objectives: [(makespan: Float, tardiness: Float)]
    ) -> [[Int]]? {
        guard isMetalAvailable,
              let device = device,
              let commandQueue = commandQueue,
              let dominancePipelineState = dominancePipelineState else {
            return nil
        }
        
        // Ensure we have objectives to compare
        let numSolutions = objectives.count
        guard numSolutions > 0 else {
            return [[]]
        }
        
        // Prepare objectives data for GPU
        var flatObjectives = [SIMD2<Float>]()
        flatObjectives.reserveCapacity(numSolutions)
        
        for (makespan, tardiness) in objectives {
            flatObjectives.append(SIMD2<Float>(makespan, tardiness))
        }
        
        // Create GPU buffers
        let objectivesBufferSize = MemoryLayout<SIMD2<Float>>.size * numSolutions
        let dominanceMatrixSize = numSolutions * numSolutions * MemoryLayout<Int32>.size
        
        guard let objectivesBuffer = device.makeBuffer(
            bytes: flatObjectives,
            length: objectivesBufferSize,
            options: .storageModeShared
        ),
        let dominanceMatrixBuffer = device.makeBuffer(
            length: dominanceMatrixSize,
            options: .storageModeShared
        ) else {
            print("Failed to create Metal buffers")
            return nil
        }
        
        // Initialize dominance matrix to zeros
        let dominanceMatrixPtr = dominanceMatrixBuffer.contents()
        memset(dominanceMatrixPtr, 0, dominanceMatrixSize)
        
        // Number of solutions as UInt32 for the kernel
        var numSolutionsUInt32 = UInt32(numSolutions)
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create command buffer or encoder")
            return nil
        }
        
        // Configure the compute encoder
        computeEncoder.setComputePipelineState(dominancePipelineState)
        computeEncoder.setBuffer(objectivesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(dominanceMatrixBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&numSolutionsUInt32, length: MemoryLayout<UInt32>.size, index: 2)
        
        // For dominance checking, use a 2D grid
        let threadsPerGroup = min(16, dominancePipelineState.threadExecutionWidth)
        
        let threadsPerThreadgroup = MTLSize(
            width: threadsPerGroup,
            height: threadsPerGroup,
            depth: 1
        )
        
        let threadgroupsPerGrid = MTLSize(
            width: (numSolutions + threadsPerGroup - 1) / threadsPerGroup,
            height: (numSolutions + threadsPerGroup - 1) / threadsPerGroup,
            depth: 1
        )
        
        // Dispatch the compute kernel
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Execute and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read back dominance matrix from GPU
        var dominanceMatrix = [[Int]](repeating: [Int](repeating: 0, count: numSolutions), count: numSolutions)
        let matrixPtr = dominanceMatrixBuffer.contents().bindMemory(to: Int32.self, capacity: numSolutions * numSolutions)
        
        for i in 0..<numSolutions {
            for j in 0..<numSolutions {
                dominanceMatrix[i][j] = Int(matrixPtr[i * numSolutions + j])
            }
        }
        
        return dominanceMatrix
    }
}