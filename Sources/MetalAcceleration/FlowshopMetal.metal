#include <metal_stdlib>
using namespace metal;

// Helper function: Evaluate a single flowshop sequence
// Returns a float2 containing (makespan, tardiness)
float2 evaluate_single_flowshop(
    constant uint* sequence,                  // Job sequence (permutation)
    constant float* processingTimes,          // Processing times matrix (flattened)
    constant float* priorities,               // Job priorities
    constant float* deadlines,                // Job deadlines
    uint numJobs,                             // Number of jobs
    uint numMachines                          // Number of machines
) {
    // Allocate arrays for completion times on each machine
    float machineCompletionTimes[10];  // Assuming max 10 machines
    
    // Initialize machine completion times to 0
    for (uint m = 0; m < numMachines; m++) {
        machineCompletionTimes[m] = 0.0;
    }
    
    // Track job completion times (when each job exits the system)
    float jobCompletionTimes[200];    // Assuming max 200 jobs
    
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
    
    return float2(makespan, totalWeightedTardiness);
}

// Xorshift random number generator for GPU
uint xorshift_rng(thread uint* state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Generate a random permutation using the Fisher-Yates shuffle
void generate_permutation(
    thread uint* sequence,
    uint length,
    thread uint* rngState
) {
    // Initialize sequence with indices 0..length-1
    for (uint i = 0; i < length; i++) {
        sequence[i] = i;
    }
    
    // Apply Fisher-Yates shuffle
    for (uint i = length - 1; i > 0; i--) {
        uint j = xorshift_rng(rngState) % (i + 1);
        // Swap sequence[i] and sequence[j]
        uint temp = sequence[i];
        sequence[i] = sequence[j];
        sequence[j] = temp;
    }
}

// Kernel: Evaluate multiple flowshop solutions in parallel
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
    constant uint* sequence = jobSequences + (id * numJobs);
    
    // Evaluate the sequence
    float2 result = evaluate_single_flowshop(
        sequence, 
        processingTimes, 
        priorities, 
        deadlines, 
        numJobs, 
        numMachines
    );
    
    // Store the result
    objectives[id] = result;
}

// Kernel: Generate and evaluate random flowshop solutions in parallel
kernel void generate_and_evaluate_flowshop(
    device uint* outputSequences,             // Output: Generated job sequences
    device const float* processingTimes,      // Processing times matrix (flattened)
    device const float* priorities,           // Job priorities
    device const float* deadlines,            // Job deadlines
    device float2* objectives,                // Output: (makespan, tardiness) for each solution
    constant uint& numJobs,                   // Number of jobs
    constant uint& numMachines,               // Number of machines
    constant uint& numSolutions,              // Number of solutions to generate and evaluate
    constant uint& seed,                      // Random seed
    uint id [[thread_position_in_grid]]       // Thread ID
) {
    // Check if this thread should generate/evaluate a solution
    if (id >= numSolutions) {
        return;
    }
    
    // Initialize random state for this thread
    uint rngState = seed + id;  // Add thread ID to make each thread's sequence different
    
    // Get pointer to this thread's output sequence
    device uint* sequence = outputSequences + (id * numJobs);
    
    // Generate a random permutation (job sequence)
    uint localSequence[200];  // Assuming max 200 jobs
    generate_permutation(localSequence, numJobs, &rngState);
    
    // Copy the generated sequence to output buffer
    for (uint i = 0; i < numJobs; i++) {
        sequence[i] = localSequence[i];
    }
    
    // Evaluate the generated sequence
    float2 result = evaluate_single_flowshop(
        localSequence, 
        processingTimes, 
        priorities, 
        deadlines, 
        numJobs, 
        numMachines
    );
    
    // Store the result
    objectives[id] = result;
}

// Kernel: Check dominance relationships between solutions (for Pareto front)
kernel void check_dominance(
    device const float2* objectives,          // Array of objective values (makespan, tardiness)
    device int* dominanceMatrix,              // Output: dominance matrix (1 if row dominates column)
    constant uint& numSolutions,              // Number of solutions
    uint2 position [[thread_position_in_grid]] // 2D thread position
) {
    uint i = position.x;  // Row index
    uint j = position.y;  // Column index
    
    // Check bounds and skip diagonal (a solution can't dominate itself)
    if (i >= numSolutions || j >= numSolutions || i == j) {
        return;
    }
    
    float2 obj_i = objectives[i];
    float2 obj_j = objectives[j];
    
    // In our bi-objective problem, solution i dominates j if:
    // - It's better or equal in both objectives
    // - It's strictly better in at least one objective
    // Note: Lower values are better for both makespan and tardiness
    
    bool atLeastAsBothObjectives = (obj_i.x <= obj_j.x) && (obj_i.y <= obj_j.y);
    bool strictlyBetterInAtLeastOne = (obj_i.x < obj_j.x) || (obj_i.y < obj_j.y);
    
    // If i dominates j, set dominanceMatrix[i*numSolutions + j] to 1
    if (atLeastAsBothObjectives && strictlyBetterInAtLeastOne) {
        dominanceMatrix[i * numSolutions + j] = 1;
    }
}