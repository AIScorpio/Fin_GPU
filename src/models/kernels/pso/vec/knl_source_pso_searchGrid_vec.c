#define n_Dim %d    // must be divisible by 4
#define n_Vec (n_Dim / 4)


__kernel void searchGrid_float4(       // change to float4, currently vec_size=4
    __global float4 *position,         // [nVec * nFish], dim-major (flattened column-wise)
    __global float4 *velocity,         // [nVec * nFish]
    __global const float4 *pbest_pos,  // [nVec * nFish]
    __global const float4 *gbest_pos,  // [nVec]
    __global const float4 *r1,         // [nVec * nFish]
    __global const float4 *r2,         // [nVec * nFish]
    const float w,
    const float c1,
    const float c2
){
    int gid = get_global_id(0);       // index of the fish
    int nFish = get_global_size(0);   // total number of fish

    #pragma unroll
    for (int i = 0; i < n_Vec; i++) {
        int idx = i * nFish + gid;    // [nVec, nFish] flatten

        float4 pos = position[idx];
        float4 vel = velocity[idx];
        float4 pbest = pbest_pos[idx];
        float4 r1_val = r1[idx];
        float4 r2_val = r2[idx];
        float4 gbest = gbest_pos[i];  // 每维 float4 的 gbest, Only depends on dimension

        vel = w * vel + c1 * r1_val * (pbest - pos) + c2 * r2_val * (gbest - pos);
        pos += vel;

        velocity[idx] = vel;
        position[idx] = pos;
    }

}


__kernel void searchGrid_f2f4(
    __global float *position,                // [nDim, nFish]
    __global float *velocity,                // [nDim, nFish]
    __global const float *pbest_pos,         // [nDim, nFish]
    __global const float *gbest_pos,         // [nDim]
    __global const float *r1,                // [nDim, nFish]
    __global const float *r2,                // [nDim, nFish]
    const float w, 
    const float c1, 
    const float c2
){
    int gid = get_global_id(0);          // index of the fish
    int nFish = get_global_size(0);      // nFish

    #pragma unroll 8
    for (int i = 0; i < n_Dim; i += 4) {
        // Vector index starts at (i * nFish + gid)
        int base_idx = i * nFish + gid;

        // Manually load float4 from 1D arrays
        float4 pos, vel, pbest, r1val, r2val, gbest;

        // Gather float4 from strided memory
        pos.s0 = position[base_idx + 0 * nFish];
        pos.s1 = position[base_idx + 1 * nFish];
        pos.s2 = position[base_idx + 2 * nFish];
        pos.s3 = position[base_idx + 3 * nFish];

        vel.s0 = velocity[base_idx + 0 * nFish];
        vel.s1 = velocity[base_idx + 1 * nFish];
        vel.s2 = velocity[base_idx + 2 * nFish];
        vel.s3 = velocity[base_idx + 3 * nFish];

        pbest.s0 = pbest_pos[base_idx + 0 * nFish];
        pbest.s1 = pbest_pos[base_idx + 1 * nFish];
        pbest.s2 = pbest_pos[base_idx + 2 * nFish];
        pbest.s3 = pbest_pos[base_idx + 3 * nFish];

        r1val.s0 = r1[base_idx + 0 * nFish];
        r1val.s1 = r1[base_idx + 1 * nFish];
        r1val.s2 = r1[base_idx + 2 * nFish];
        r1val.s3 = r1[base_idx + 3 * nFish];

        r2val.s0 = r2[base_idx + 0 * nFish];
        r2val.s1 = r2[base_idx + 1 * nFish];
        r2val.s2 = r2[base_idx + 2 * nFish];
        r2val.s3 = r2[base_idx + 3 * nFish];

        gbest.s0 = gbest_pos[i + 0];
        gbest.s1 = gbest_pos[i + 1];
        gbest.s2 = gbest_pos[i + 2];
        gbest.s3 = gbest_pos[i + 3];

        // PSO velocity and position update
        vel = w * vel + c1 * r1val * (pbest - pos) + c2 * r2val * (gbest - pos);
        pos += vel;

        // Scatter float4 back to strided memory
        position[base_idx + 0 * nFish] = pos.s0;
        position[base_idx + 1 * nFish] = pos.s1;
        position[base_idx + 2 * nFish] = pos.s2;
        position[base_idx + 3 * nFish] = pos.s3;

        velocity[base_idx + 0 * nFish] = vel.s0;
        velocity[base_idx + 1 * nFish] = vel.s1;
        velocity[base_idx + 2 * nFish] = vel.s2;
        velocity[base_idx + 3 * nFish] = vel.s3;
    }
}