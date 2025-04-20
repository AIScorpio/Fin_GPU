#define n_Dim %d    // must be divisible by 4
#define n_Vec (n_Dim / 4)


/* update nParticle positions & velocity, each thread handles on particle */
__kernel void searchGrid(
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

    for (int i = 0; i < n_Dim; i++) {
        int idx = i * nFish + gid;       // index into flattened (nDim, nFish)

        float pos = position[idx];
        float vel = velocity[idx];
        float pbest = pbest_pos[idx];
        float r1_val = r1[idx];
        float r2_val = r2[idx];
        float gbest = gbest_pos[i];          // Only depends on dimension

        vel = w * vel + c1 * r1_val * (pbest - pos) + c2 * r2_val * (gbest - pos);
        pos += vel;

        velocity[idx] = vel;
        position[idx] = pos;
    }
}



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