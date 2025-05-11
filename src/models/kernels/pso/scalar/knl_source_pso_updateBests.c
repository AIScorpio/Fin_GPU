#define n_Dim %d
#define n_Fish %d

// each thread takes care of one fish
// 1. Compare pBest_costs vs Costs     scaler vs scaler
// 2. if Costs > pBest_costs           scaler
// 3. pBest_costs[i] = Costs[i] & pBest_pos[i] = position[i]              [nDim]
__kernel void update_pbest(
    __global float *costs,
    __global float *pbest_costs,
    __global float *position,
    __global float *pbest_pos
){
    int gid = get_global_id(0);
    int nParticle = get_global_size(0);

    // Update personal best
    if (costs[gid] > pbest_costs[gid]) {
        pbest_costs[gid] = costs[gid];
        
        // Copy all dimensions
        for (int i = 0; i < n_Dim; i++) {
            int idx = gid + i * nParticle;
            pbest_pos[idx] = position[idx];
        }
    }

}


// each thread handle one dimension
__kernel void update_gbest_pos(
    __global float *gbest_pos, 
    __global float *pbest_pos,
    const int gbest_id
){
    int gid = get_global_id(0);
    int pbest_pos_id = gbest_id + gid * n_Fish;

    gbest_pos[gid] = pbest_pos[pbest_pos_id];
}
