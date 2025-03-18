import Foundation
import Metal
import OptimizationCore

/// Class that provides Metal-accelerated operations for optimization tasks
public class MetalAccelerator {
    // Singleton instance for easy access
    public static let shared = MetalAccelerator()
    
    // Metal device and command queue
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    
    // Metal compute pipeline state for dominance checking
    private var dominancePipelineState: MTLComputePipelineState?
    
    // Whether Metal is available and properly initialized
    public let isMetalAvailable: Bool
    
    // Initialize Metal device and command queue
    private init() {
        // Try to get the default Metal device
        if let device = MTLCreateSystemDefaultDevice() {
            self.device = device
            
            // Create a command queue
            if let commandQueue = device.makeCommandQueue() {
                self.commandQueue = commandQueue
                self.isMetalAvailable = true
                
                // Set up compute pipeline for dominance checking
                setupComputePipelines()
            } else {
                // Failed to create command queue
                self.commandQueue = nil
                self.isMetalAvailable = false
            }
        } else {
            // Metal not supported on this device
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
        
        // Create a Metal library with our shader functions
        do {
            let metalLibrary = try createMetalShaderLibrary()
            
            // Create compute pipeline state for dominance checking
            if let dominanceFunction = metalLibrary.makeFunction(name: "checkDominance") {
                do {
                    dominancePipelineState = try device.makeComputePipelineState(function: dominanceFunction)
                } catch {
                    print("Error creating compute pipeline state: \(error)")
                }
            }
        } catch {
            print("Error creating Metal shader library: \(error)")
        }
    }
    
    /// Create the Metal shader library programmatically
    private func createMetalShaderLibrary() throws -> MTLLibrary {
        guard let device = device else {
            throw NSError(domain: "MetalAccelerator", code: -1, userInfo: [NSLocalizedDescriptionKey: "Metal device not available"])
        }
        
        // Metal shading language code for our compute kernels
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct Solution {
            float values[16];
            int criteriaCount;
        };
        
        // Determine if solution at index i dominates solution at index j
        kernel void checkDominance(device const Solution* solutions [[buffer(0)]],
                                 device atomic_int* dominanceMatrix [[buffer(1)]],
                                 device const int* solutionCount [[buffer(2)]],
                                 device atomic_int* debugCounter [[buffer(3)]],
                                 uint2 position [[thread_position_in_grid]]) {
            // Get indices for the two solutions to compare
            uint i = position.x;
            uint j = position.y;
            
            // Skip diagonal elements (same solution) and out-of-bounds
            if (i >= *solutionCount || j >= *solutionCount || i == j) {
                return;
            }
            
            // Debug: increment counter to verify shader execution
            atomic_fetch_add_explicit(debugCounter, 1, memory_order_relaxed);
            
            // Get the solutions to compare
            Solution sol_i = solutions[i];
            Solution sol_j = solutions[j];
            
            // Ensure both solutions have the same number of criteria
            if (sol_i.criteriaCount != sol_j.criteriaCount) {
                return;
            }
            
            bool atLeastOneStrictlyBetter = false;
            bool allAtLeastAsGood = true;
            
            // Compare each criterion
            for (int k = 0; k < sol_i.criteriaCount; k++) {
                float value_i = sol_i.values[k];
                float value_j = sol_j.values[k];
                
                // Check if solution i is better in this criterion
                if (value_i > value_j) {
                    allAtLeastAsGood = false;
                    break;
                } else if (value_i < value_j) {
                    atLeastOneStrictlyBetter = true;
                }
            }
            
            // If solution i dominates solution j, mark it in the dominance matrix
            if (allAtLeastAsGood && atLeastOneStrictlyBetter) {
                atomic_store_explicit(dominanceMatrix + (i * (*solutionCount) + j), 1, memory_order_relaxed);
            }
        }
        """
        
        // Create options for the library
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        
        // Create the library
        do {
            return try device.makeLibrary(source: shaderSource, options: options)
        } catch {
            throw error
        }
    }
    
    /// Check dominance relationships between all pairs of solutions using GPU
    /// - Parameter solutions: Array of numeric solutions to compare
    /// - Returns: Dominance matrix where 1 means row dominates column
    public func checkDominanceRelationships<T: Solution>(solutions: [T]) -> [[Int]] {
        // Safety checks
        guard isMetalAvailable,
              let device = device,
              let commandQueue = commandQueue,
              let pipelineState = dominancePipelineState, 
              !solutions.isEmpty else {
            print("Metal acceleration unavailable, falling back to CPU")
            return []
        }
        
        // For large solution sets, use smaller batches
        if solutions.count > 1000 {
            print("Large solution set detected, processing in sub-batches...")
            let batchSize = 1000 // Reduced from 5000 to 1000
            var finalMatrix = Array(repeating: Array(repeating: 0, count: solutions.count), count: solutions.count)
            
            // Process in sub-batches
            for i in stride(from: 0, to: solutions.count, by: batchSize) {
                let batchEnd = min(i + batchSize, solutions.count)
                let batchSolutions = Array(solutions[i..<batchEnd])
                
                // Process this batch
                if let batchMatrix = processMetalBatch(batchSolutions, device: device, commandQueue: commandQueue, pipelineState: pipelineState) {
                    // Copy batch results to final matrix
                    for row in 0..<batchMatrix.count {
                        for col in 0..<batchMatrix[row].count {
                            finalMatrix[i + row][i + col] = batchMatrix[row][col]
                        }
                    }
                } else {
                    print("Failed to process batch \(i)-\(batchEnd), falling back to CPU")
                    return []
                }
            }
            
            return finalMatrix
        }
        
        return processMetalBatch(solutions, device: device, commandQueue: commandQueue, pipelineState: pipelineState) ?? []
    }
    
    /// Optimized version that processes solutions in a single large batch to maximize GPU throughput
    public func checkDominanceRelationshipsOptimized<T: Solution>(solutions: [T]) -> [[Int]] {
        // Safety checks
        guard isMetalAvailable,
              let device = device,
              let commandQueue = commandQueue,
              let pipelineState = dominancePipelineState,
              !solutions.isEmpty else {
            print("Metal acceleration unavailable, returning empty matrix")
            return []
        }
        
        print("Processing \(solutions.count) solutions in optimized GPU mode")
        
        // Prepare solution data for Metal all at once
        let prepStartTime = Date()
        var metalSolutions = [MetalSolution](repeating: MetalSolution(), count: solutions.count)
        
        for (i, solution) in solutions.enumerated() {
            let criteriaCount = min(solution.criteria.count, 16)
            metalSolutions[i].criteriaCount = Int32(criteriaCount)
            
            for j in 0..<criteriaCount {
                if let numericCriterion = solution.criteria[j] as? NumericCriterion {
                    let value = numericCriterion.lowerIsBetter ?
                        Float(numericCriterion.value) : -Float(numericCriterion.value)
                    metalSolutions[i].setValue(value, atIndex: j)
                }
            }
        }
        let prepTime = Date().timeIntervalSince(prepStartTime)
        print("Data preparation time: \(String(format: "%.4f", prepTime)) seconds")
        
        // Calculate buffer sizes
        let solutionCount = solutions.count
        let solutionsBufferSize = MemoryLayout<MetalSolution>.stride * solutionCount
        let matrixLength = solutionCount * solutionCount
        let matrixBufferSize = matrixLength * MemoryLayout<Int32>.stride
        
        print("Buffer sizes - Solutions: \(solutionsBufferSize) bytes, Matrix: \(matrixBufferSize) bytes")
        
        // Create Metal buffers with optimized storage mode
        let bufferStartTime = Date()
        guard let solutionsBuffer = device.makeBuffer(
            bytes: &metalSolutions,
            length: solutionsBufferSize,
            options: .storageModeShared // Use shared memory for faster transfers on Apple Silicon
        ),
        let dominanceMatrixBuffer = device.makeBuffer(
            length: matrixBufferSize,
            options: .storageModeShared
        ),
        let countBuffer = device.makeBuffer(
            bytes: [Int32(solutionCount)],
            length: MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ),
        let debugCounterBuffer = device.makeBuffer(
            length: MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ) else {
            print("Failed to create Metal buffers")
            return []
        }
        let bufferTime = Date().timeIntervalSince(bufferStartTime)
        print("Buffer creation time: \(String(format: "%.4f", bufferTime)) seconds")
        
        // Initialize buffers
        memset(dominanceMatrixBuffer.contents(), 0, matrixBufferSize)
        memset(debugCounterBuffer.contents(), 0, MemoryLayout<Int32>.stride)
        
        // Process in appropriately sized chunks for Metal
        var resultMatrix = Array(repeating: Array(repeating: 0, count: solutionCount), count: solutionCount)
        let maxGPUChunkSize = min(solutionCount, 2000) // Larger chunk size for better GPU utilization
        
        for chunkStart in stride(from: 0, to: solutionCount, by: maxGPUChunkSize) {
            let chunkEnd = min(chunkStart + maxGPUChunkSize, solutionCount)
            let chunkSize = chunkEnd - chunkStart
            
            print("Processing GPU chunk \(chunkStart)-\(chunkEnd)...")
            
            // Create command buffer and encoder
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                print("Failed to create command buffer or encoder")
                continue
            }
            
            // Configure compute encoder
            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(solutionsBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(dominanceMatrixBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(countBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(debugCounterBuffer, offset: 0, index: 3)
            
            // Calculate grid and threadgroup sizes for best performance
            let gridSize = MTLSize(width: chunkSize, height: solutionCount, depth: 1)
            
            // Optimize threadgroup size based on device capabilities
            let threadsPerGroup = 16 // Increased thread group size for better occupancy
            let threadgroupSize = MTLSize(
                width: threadsPerGroup,
                height: threadsPerGroup,
                depth: 1
            )
            
            print("Grid size: \(gridSize.width)x\(gridSize.height), Threadgroup size: \(threadgroupSize.width)x\(threadgroupSize.height)")
            
            // Dispatch compute kernel
            let gpuStartTime = Date()
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            computeEncoder.endEncoding()
            
            // Execute and wait for results
            var completed = false
            
            commandBuffer.addCompletedHandler { buffer in
                let gpuKernelTime = Date().timeIntervalSince(gpuStartTime)
                
                if buffer.status == .completed {
                    let debugCounterPtr = debugCounterBuffer.contents().bindMemory(to: Int32.self, capacity: 1)
                    let executedThreads = Int(debugCounterPtr.pointee)
                    print("GPU Kernel executed \(executedThreads) times in \(String(format: "%.4f", gpuKernelTime)) seconds")
                    print("Threads per second: \(String(format: "%.0f", Double(executedThreads) / gpuKernelTime))")
                } else {
                    print("Metal execution failed with status: \(buffer.status.rawValue)")
                }
                completed = true
            }
            
            commandBuffer.commit()
            
            // Wait for completion
            let chunkStartTime = Date()
            let timeout: TimeInterval = 30.0 // Increased timeout for larger workloads
            
            while !completed {
                if Date().timeIntervalSince(chunkStartTime) > timeout {
                    print("Metal execution timed out after \(timeout) seconds")
                    break
                }
                Thread.sleep(forTimeInterval: 0.01)
            }
        }
        
        // After all chunks processed, read final result from GPU memory
        let readStartTime = Date()
        let matrixPtr = dominanceMatrixBuffer.contents().bindMemory(to: Int32.self, capacity: matrixLength)
        var dominanceCount = 0
        
        for i in 0..<solutionCount {
            for j in 0..<solutionCount {
                let value = Int(matrixPtr[i * solutionCount + j])
                resultMatrix[i][j] = value
                if value == 1 {
                    dominanceCount += 1
                }
            }
        }
        let readTime = Date().timeIntervalSince(readStartTime)
        print("Result read time: \(String(format: "%.4f", readTime)) seconds")
        print("Found \(dominanceCount) dominance relationships")
        
        return resultMatrix
    }
    
    /// Process a single batch of solutions using Metal
    private func processMetalBatch<T: Solution>(_ solutions: [T], device: MTLDevice, commandQueue: MTLCommandQueue, pipelineState: MTLComputePipelineState) -> [[Int]]? {
        let solutionCount = solutions.count
        print("Processing batch of \(solutionCount) solutions...")
        
        // Prepare solution data for Metal
        let prepStartTime = Date()
        var metalSolutions = [MetalSolution](repeating: MetalSolution(), count: solutions.count)
        
        for (i, solution) in solutions.enumerated() {
            let criteriaCount = min(solution.criteria.count, 16)
            metalSolutions[i].criteriaCount = Int32(criteriaCount)
            
            for j in 0..<criteriaCount {
                if let numericCriterion = solution.criteria[j] as? NumericCriterion {
                    let value = numericCriterion.lowerIsBetter ? 
                        Float(numericCriterion.value) : -Float(numericCriterion.value)
                    metalSolutions[i].setValue(value, atIndex: j)
                }
            }
        }
        let prepTime = Date().timeIntervalSince(prepStartTime)
        print("Data preparation time: \(String(format: "%.4f", prepTime)) seconds")
        
        // Calculate buffer sizes
        let solutionsBufferSize = MemoryLayout<MetalSolution>.stride * solutions.count
        let matrixLength = solutionCount * solutionCount
        let matrixBufferSize = matrixLength * MemoryLayout<Int32>.stride
        
        print("Buffer sizes - Solutions: \(solutionsBufferSize) bytes, Matrix: \(matrixBufferSize) bytes")
        
        // Create Metal buffers
        let bufferStartTime = Date()
        guard let solutionsBuffer = device.makeBuffer(
            bytes: &metalSolutions,
            length: solutionsBufferSize,
            options: .storageModeShared
        ),
        let dominanceMatrixBuffer = device.makeBuffer(
            length: matrixBufferSize,
            options: .storageModeShared
        ),
        let countBuffer = device.makeBuffer(
            bytes: [Int32(solutionCount)],
            length: MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ),
        let debugCounterBuffer = device.makeBuffer(
            length: MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ) else {
            print("Failed to create Metal buffers")
            return nil
        }
        let bufferTime = Date().timeIntervalSince(bufferStartTime)
        print("Buffer creation time: \(String(format: "%.4f", bufferTime)) seconds")
        
        // Initialize buffers
        memset(dominanceMatrixBuffer.contents(), 0, matrixBufferSize)
        memset(debugCounterBuffer.contents(), 0, MemoryLayout<Int32>.stride)
        
        // Create and configure command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create command buffer or encoder")
            return nil
        }
        
        // Configure compute encoder
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setBuffer(solutionsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(dominanceMatrixBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(countBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(debugCounterBuffer, offset: 0, index: 3)
        
        // Calculate grid and threadgroup sizes
        let gridSize = MTLSize(width: solutionCount, height: solutionCount, depth: 1)
        let maxThreadsPerGroup = min(32, pipelineState.maxTotalThreadsPerThreadgroup / 32)
        let threadgroupSize = MTLSize(
            width: maxThreadsPerGroup,
            height: maxThreadsPerGroup,
            depth: 1
        )
        
        print("Grid size: \(gridSize.width)x\(gridSize.height), Threadgroup size: \(threadgroupSize.width)x\(threadgroupSize.height)")
        
        // Track GPU kernel execution time specifically
        var gpuKernelTime: TimeInterval = 0
        
        // Dispatch compute kernel
        let gpuStartTime = Date()
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        // Execute and wait for results
        var dominanceMatrix: [[Int]]?
        var completed = false
        
        commandBuffer.addCompletedHandler { buffer in
            gpuKernelTime = Date().timeIntervalSince(gpuStartTime)
            
            if buffer.status == .completed {
                let debugCounterPtr = debugCounterBuffer.contents().bindMemory(to: Int32.self, capacity: 1)
                let executedThreads = Int(debugCounterPtr.pointee)
                print("GPU Kernel executed \(executedThreads) times in \(String(format: "%.4f", gpuKernelTime)) seconds")
                print("Threads per second: \(String(format: "%.0f", Double(executedThreads) / gpuKernelTime))")
                
                let readStartTime = Date()
                let matrixPtr = dominanceMatrixBuffer.contents().bindMemory(to: Int32.self, capacity: matrixLength)
                var matrix = Array(repeating: Array(repeating: 0, count: solutionCount), count: solutionCount)
                var dominanceCount = 0
                
                for i in 0..<solutionCount {
                    for j in 0..<solutionCount {
                        let value = Int(matrixPtr[i * solutionCount + j])
                        matrix[i][j] = value
                        if value == 1 {
                            dominanceCount += 1
                        }
                    }
                }
                let readTime = Date().timeIntervalSince(readStartTime)
                
                print("Result read time: \(String(format: "%.4f", readTime)) seconds")
                print("Found \(dominanceCount) dominance relationships")
                dominanceMatrix = matrix
            } else {
                print("Metal execution failed with status: \(buffer.status.rawValue)")
            }
            completed = true
        }
        
        commandBuffer.commit()
        
        // Wait for completion with increased timeout
        let startTime = Date()
        let timeout: TimeInterval = 10.0
        
        while !completed {
            if Date().timeIntervalSince(startTime) > timeout {
                print("Metal execution timed out after \(timeout) seconds")
                return nil
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        
        // Performance breakdown
        print("\nPerformance breakdown:")
        print("Data preparation: \(String(format: "%.4f", prepTime)) seconds")
        print("Buffer creation: \(String(format: "%.4f", bufferTime)) seconds")
        print("GPU kernel execution: \(String(format: "%.4f", gpuKernelTime)) seconds")
        print("Total batch time: \(String(format: "%.4f", Date().timeIntervalSince(startTime))) seconds")
        
        return dominanceMatrix
    }
    
    /// Find non-dominated solutions using the GPU-computed dominance matrix
    /// - Parameters:
    ///   - solutions: Array of solutions
    ///   - dominanceMatrix: Matrix where 1 means row dominates column
    /// - Returns: Array of non-dominated solutions
    public func findNonDominatedSolutions<T: Solution>(solutions: [T], dominanceMatrix: [[Int]]) -> [T] {
        var nonDominated: [T] = []
        
        for (i, solution) in solutions.enumerated() {
            var isDominated = false
            
            // Check if this solution is dominated by any other
            for j in 0..<solutions.count {
                if i != j && j < dominanceMatrix.count && i < dominanceMatrix[j].count && dominanceMatrix[j][i] == 1 {
                    isDominated = true
                    break;
                }
            }
            
            if !isDominated {
                nonDominated.append(solution)
            }
        }
        
        return nonDominated
    }
}

/// Metal-compatible solution structure with array for values
struct MetalSolution {
    // Using fixed-size arrays for Metal compatibility
    var values: (Float, Float, Float, Float, Float, Float, Float, Float,
                 Float, Float, Float, Float, Float, Float, Float, Float)
    var criteriaCount: Int32
    
    init() {
        self.values = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        self.criteriaCount = 0
    }
    
    mutating func setValue(_ value: Float, atIndex index: Int) {
        switch index {
            case 0: values.0 = value
            case 1: values.1 = value
            case 2: values.2 = value
            case 3: values.3 = value
            case 4: values.4 = value
            case 5: values.5 = value
            case 6: values.6 = value
            case 7: values.7 = value
            case 8: values.8 = value
            case 9: values.9 = value
            case 10: values.10 = value
            case 11: values.11 = value
            case 12: values.12 = value
            case 13: values.13 = value
            case 14: values.14 = value
            case 15: values.15 = value
            default: break
        }
    }
}

/// Extension to ParetoFront to add Metal acceleration
extension ParetoFront {
    /// Add multiple solutions to the Pareto front using Metal-accelerated dominance checking
    /// - Parameter solutions: Array of solutions to evaluate
    /// - Returns: Array of solutions that were added to the front
    public func addBatchUsingMetal(_ solutions: [S]) -> [S] {
        // Skip Metal acceleration for small batches or if Metal is unavailable
        if solutions.count <= 50 || !MetalAccelerator.shared.isMetalAvailable {
            print("Using CPU implementation (small batch or Metal unavailable)")
            return addBatch(solutions)
        }
        
        // Try Metal acceleration with timeout handling
        print("Attempting Metal acceleration for Pareto front calculation...")
        
        let startTime = Date()
        let dominanceMatrix = MetalAccelerator.shared.checkDominanceRelationships(solutions: solutions)
        
        // Safety check - if dominance matrix is empty, fall back to CPU implementation
        if dominanceMatrix.isEmpty {
            print("Metal acceleration failed, falling back to CPU implementation")
            return addBatch(solutions)
        }
        
        let metalTime = Date().timeIntervalSince(startTime)
        print("Metal dominance calculation completed in \(String(format: "%.2f", metalTime)) seconds")
        
        // Find non-dominated solutions within the batch
        let nonDominatedInBatch = MetalAccelerator.shared.findNonDominatedSolutions(
            solutions: solutions, dominanceMatrix: dominanceMatrix)
        
        print("Found \(nonDominatedInBatch.count) non-dominated solutions using Metal")
        
        // Now add the filtered batch to the existing Pareto front
        var addedSolutions: [S] = []
        
        for solution in nonDominatedInBatch {
            if add(solution) {
                addedSolutions.append(solution)
            }
        }
        
        return addedSolutions
    }
}