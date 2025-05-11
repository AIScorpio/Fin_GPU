#define n_Dim %d   

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


