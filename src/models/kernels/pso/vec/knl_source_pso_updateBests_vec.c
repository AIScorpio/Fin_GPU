#define n_Dim %d
#define n_Fish %d

// each thread takes care of one fish
// 1. Compare pBest_costs vs Costs     scaler vs scaler
// 2. if Costs > pBest_costs           scaler
// 3. pBest_costs[i] = Costs[i] & pBest_pos[i] = position[i]              [nDim]

__kernel void update_pbest_f2f4(
    __global float *costs,
    __global float *pbest_costs,
    __global float *position,
    __global float *pbest_pos
){
    int gid = get_global_id(0);             // current fish ID
    int nParticle = get_global_size(0);

    if (costs[gid] > pbest_costs[gid]) {
        pbest_costs[gid] = costs[gid];

        // Copy all dimensions in steps of 4
        #pragma unroll 8
        for (int i = 0; i < n_Dim; i += 4) {
            int base_idx = i * nParticle + gid;

            float4 pos_vec = (float4)(
                position[base_idx + 0 * nParticle],
                position[base_idx + 1 * nParticle],
                position[base_idx + 2 * nParticle],
                position[base_idx + 3 * nParticle]
            );

            pbest_pos[base_idx + 0 * nParticle] = pos_vec.s0;
            pbest_pos[base_idx + 1 * nParticle] = pos_vec.s1;
            pbest_pos[base_idx + 2 * nParticle] = pos_vec.s2;
            pbest_pos[base_idx + 3 * nParticle] = pos_vec.s3;
        }
    }
}


// each thread handle one dimension - element wise dimension update
__kernel void update_gbest_pos_f2f4(
    __global float *gbest_pos, 
    __global float *pbest_pos,
    const int gbest_id
){
    int gid = get_global_id(0);     // one thread per dimension

    #pragma unroll 8
    for (int i = 0; i < n_Dim; i += 4) {
        int idx0 = (i + 0) * n_Fish + gbest_id;
        int idx1 = (i + 1) * n_Fish + gbest_id;
        int idx2 = (i + 2) * n_Fish + gbest_id;
        int idx3 = (i + 3) * n_Fish + gbest_id;

        float4 val = (float4)(
            pbest_pos[idx0],
            pbest_pos[idx1],
            pbest_pos[idx2],
            pbest_pos[idx3]
        );

        gbest_pos[i + 0] = val.s0;
        gbest_pos[i + 1] = val.s1;
        gbest_pos[i + 2] = val.s2;
        gbest_pos[i + 3] = val.s3;
    }
}