#include <metal_stdlib>
using namespace metal;

// Intensive computation for each objective
float3 compute_objectives(const device float* values) {
    float x = values[0];
    float y = values[1];
    
    // Objective 1: Complex sinusoidal pattern
    float obj1 = sin(x * 3.14159) * cos(y * 2.71828) * exp(-((x*x + y*y)/100.0));
    
    // Objective 2: Waves and peaks
    float obj2 = cos(sqrt(x*x + y*y)) * exp(-(x*x + y*y)/200.0) + sin(x*0.5) * cos(y*0.5);
    
    // Objective 3: Complex landscape
    float obj3 = sin(x*y/10.0) * cos(x-y) * exp(-(x*x + y*y)/300.0);
    
    // Do intensive computation to stress the GPU
    for(int i = 0; i < 100; i++) {
        obj1 = sin(obj1) * cos(obj2) + obj3;
        obj2 = cos(obj2) * sin(obj3) + obj1;
        obj3 = sin(obj3) * cos(obj1) + obj2;
    }
    
    return float3(obj1, obj2, obj3);
}

kernel void dominance_check(const device float* solutions [[buffer(0)]],
                          device int* dominance_matrix [[buffer(1)]],
                          uint2 position [[thread_position_in_grid]],
                          uint2 grid_size [[threads_per_grid]]) {
    
    const int i = position.x;
    const int j = position.y;
    
    if (i >= grid_size.x || j >= grid_size.y || i == j) {
        return;
    }
    
    // Each solution has 10 values
    const device float* sol_i = solutions + (i * 10);
    const device float* sol_j = solutions + (j * 10);
    
    // Compute objectives for both solutions
    float3 objectives_i = compute_objectives(sol_i);
    float3 objectives_j = compute_objectives(sol_j);
    
    // Check dominance
    bool i_dominates_j = true;
    bool j_dominates_i = true;
    
    for (int k = 0; k < 3; k++) {
        if (objectives_i[k] > objectives_j[k]) {
            i_dominates_j = false;
        }
        if (objectives_j[k] > objectives_i[k]) {
            j_dominates_i = false;
        }
    }
    
    // Store result in dominance matrix
    if (i_dominates_j && !j_dominates_i) {
        dominance_matrix[i * grid_size.x + j] = 1;
    } else if (j_dominates_i && !i_dominates_j) {
        dominance_matrix[i * grid_size.x + j] = -1;
    } else {
        dominance_matrix[i * grid_size.x + j] = 0;
    }
}