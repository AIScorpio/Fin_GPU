#define n_Dim %d
#define n_PATH %d
#define n_PERIOD %d
#define n_Fish %d

/* update nParticle positions & velocity, each thread handles on particle */
__kernel void pso(
    // 1. searchGrid
    __global float *position,                // [nDim, nFish]
    __global float *velocity,                // [nDim, nFish]
    __global float *pbest_pos,         // [nDim, nFish]
    __global const float *gbest_pos,         // [nDim]
    __global const float *r1,                // [nDim, nFish]
    __global const float *r2,                // [nDim, nFish]
    const float w, 
    const float c1, 
    const float c2, 
    // 2. American option - fitness each fish
    __global const float *St, 
    __global float *costs, 
    __global int *boundary_idx, 
    __global float *exercise,
    const float r, 
    const float T, 
    const float K, 
    const char opt,
    // 3. update pbest
    __global float *pbest_costs
){
    int gid = get_global_id(0);          // index of the fish
    int nParticle = get_global_size(0);      // nFish

    /* 1. searchGrid */
    for (int i = 0; i < n_Dim; i++) {
        int idx = i * nParticle + gid;       // index into flattened (nDim, nFish)

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

    /* 2. fitness calculation - American option */
    float dt = T / n_PERIOD;
    float tmp_C = 0.0f;

    for (int path=0; path<n_PATH; path++){
        int bound_idx = n_PERIOD - 1;            // init to last period
        int St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;
        float early_excise = St[St_T_idx];       // init to St path_i last period price

        for (int prd=n_PERIOD-1; prd>-1; prd--){
            float cur_fish_val = position[gid + prd * nParticle];
            float cur_St_val = St[prd + path * n_PERIOD];

            // check early cross exhaust all periods
            bound_idx = select(bound_idx, prd, isgreaterequal(cur_fish_val, cur_St_val));               // a>b? a:b will be select(b, a, a>b), mind the sequence!!
            early_excise = select(early_excise, cur_St_val, isgreaterequal(cur_fish_val, cur_St_val));
        }

        // compute current path present value of simulated American option; then cumulate for average later
        tmp_C += exp(-r * (bound_idx+1) * dt) * max(0.0f, (K - early_excise)*opt); 
    }
    
    tmp_C = tmp_C / n_PATH;    // get average C_hat for current fish/thread investigation
    costs[gid] = tmp_C;   

    /* 3. update pbest */
    if (tmp_C > pbest_costs[gid]) {
        pbest_costs[gid] = tmp_C;
        
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