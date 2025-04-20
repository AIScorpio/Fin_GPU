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
    // barrier(CLK_GLOBAL_MEM_FENCE);

    /* 2. fitness calculation - American option */
    float dt = T / n_PERIOD;
    float tmp_C = 0.0f;

    for (int path=0; path<n_PATH; path++){
        int bound_idx = n_PERIOD - 1;            // init to last period
        int St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;
        float early_excise = St[St_T_idx];       // init to St path_i last period price

        // int path_boundary_id = gid + path * nParticle;  


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
    // int boundary_gid;                 // define shared global access id for boundary_idx & exercise
    // int St_T_idx;                     // St_T id for all paths
    // float cur_fish_val = 0.0f;        // current fish element value, pointer to loop thru current fish dimension, i.e. time t for St
    // float cur_St_val = 0.0f;          // current St element value, pointer to loop thru current path at time t of St

    // // Init intermediate buffer for next iteration
    // for (int path = 0; path < n_PATH; path++){         
    //     boundary_gid = gid + path * nParticle;        // calc shared global access id for boundary_idx & exercise
    //     St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;  // calc St_T id for all paths
    //     boundary_idx[boundary_gid] = n_PERIOD - 1;    // reset boundary index to time T
    //     exercise[boundary_gid] = St[St_T_idx];        // reset exercise to St_T
    // }
    
    // /* set intermediate arrays of index and exercise */
    // //outer loop thru periods (Note that fish dimension is equal to St periods), loop backwards in time to track early exercise point for each path
    // for (int prd= n_PERIOD - 1; prd > -1 ; prd--){
    //     cur_fish_val = position[gid + prd * nParticle];    // PSO global index & value

    //     //inner loop thru all St paths at current period
    //     //St global pointer from 0 to (nPath * nPeriod -1)
    //     for (int path= 0; path < n_PATH; path++){
    //         cur_St_val = St[prd + path * n_PERIOD];    // get St path value at same period
    //         boundary_gid = gid + path * nParticle;  
            
    //         // check if first cross: 1) pso > St; 2) corresponding boundary index not set from previous loops
    //         if (cur_fish_val > cur_St_val){  
    //             boundary_idx[boundary_gid] = prd;        //store current time step
    //             exercise[boundary_gid] = cur_St_val;     //store current price
    //         } 
    //     }
    // }
    
    // /* calc costs for current fish */
    // float tmp_C = 0.0f;
    // float dt = T / n_PERIOD;
    // // input parameter opt is the Put/Call flag, 1 for Put, -1 for Call
    // for (int path = 0; path < n_PATH; path++){         // sum all path costs
    //     boundary_gid = gid + path * nParticle;         // calc shared global access id for boundary_idx & exercise
    //     tmp_C += exp(-r * (boundary_idx[boundary_gid]+1) * dt) * max(0.0f, (K - exercise[boundary_gid]) * opt);   // boudnary_idx +1 to reflect actual time step, considering present is time zero
    // }
    
    // tmp_C = tmp_C / n_PATH;
    // costs[gid] = tmp_C;     // get average costs for current fish/thread investigation
    // // barrier(CLK_GLOBAL_MEM_FENCE);

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