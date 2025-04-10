#define n_PATH %d
#define n_PERIOD %d

/* 
        # # udpated on 6 Apr. 2025 
        # 1. for unified Z, St shape as [nPath by nPeriod], synced and shared by PSO and Longstaff
        # 2. No concatenation of spot price
        # 3. handle index of time period, spot price at time zero (present), St from time 1 to T
*/
__kernel void psoAmerOption_gb(
    __global const float *St, 
    __global const float *pso, 
    __global float *C_hat, 
    __global int *boundary_idx, 
    __global float *exercise,
    const float r, 
    const float T, 
    const float K, 
    const char opt
){
    
    //global variables, one fish per thread (work-item), check for early aross, for all paths
    int gid = get_global_id(0);            //thread id, per fish
    int nParticle = get_global_size(0);    //number of fishes
    
    int boundary_gid;                 // define shared global access id for boundary_idx & exercise
    int St_T_idx;                     // St_T id for all paths
    float cur_fish_val = 0.0f;        // current fish element value, pointer to loop thru current fish dimension, i.e. time t for St
    float cur_St_val = 0.0f;          // current St element value, pointer to loop thru current path at time t of St

    // Init intermediate buffer for next iteration
    for (int path = 0; path < n_PATH; path++){         
        boundary_gid = gid + path * nParticle;        // calc shared global access id for boundary_idx & exercise
        St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;  // calc St_T id for all paths
        boundary_idx[boundary_gid] = n_PERIOD - 1;    // reset boundary index to time T
        exercise[boundary_gid] = St[St_T_idx];        // reset exercise to St_T
    }
    
    /* set intermediate arrays of index and exercise */
    //outer loop thru periods (Note that fish dimension is equal to St periods), loop backwards in time to track early exercise point for each path
    for (int prd= n_PERIOD - 1; prd > -1 ; prd--){
        //e.g. total 5 fishes, nParticle = 5; total 3 time steps, nPeriod = 3
        //gid=0: prd:[2, 1, 0] --> 0 + [2 1 0] * nParticle =  PSO global index [10, 5, 0] for fish 0
        //gid=1: prd:[2, 1, 0] --> 1 + [2 1 0] * nParticle =  PSO global index [11, 6, 1] for fish 1
        float cur_fish_val = pso[gid + prd * nParticle];    // PSO global index & value

        //inner loop thru all St paths at current period
        //St global pointer from 0 to (nPath * nPeriod -1)
        for (int path= 0; path < n_PATH; path++){
            //e.g. total 3 periods, nPeriod = 3
            //prd: 2  path:[0, 1, 2, 3] --> 2 + [0, 1, 2, 3] * nPeriod = St global index [2, 5, 8, 11] for period 2
            //prd: 1  path:[0, 1, 2, 3] --> 1 + [0, 1, 2, 3] * nPeriod = St global index [1, 4, 7, 10] for period 2
            float cur_St_val = St[prd + path * n_PERIOD];    // get St path value at same period
            
            // each fish access to corresponding column of boundary_idx and exercise matrix, both nPath by nFish/nParticle
            // calc shared global access id for boundary_idx & exercise
            // e.g. total 5 fishes, nParticle=5, total 10 paths, nPath=10
            // gid=0: --> 0 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * nParticle = boundary_idx/exercise global index [0, 5, 10, 15, 20..45]
            // gid=1: --> 1 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * nParticle = boundary_idx/exercise global index [1, 6, 11, 16, 21..46]
            boundary_gid = gid + path * nParticle;  
            
            // check if first cross: 1) pso > St; 2) corresponding boundary index not set from previous loops
            if (cur_fish_val > cur_St_val){  
                boundary_idx[boundary_gid] = prd;        //store current time step
                exercise[boundary_gid] = cur_St_val;     //store current price
            } 
        }
    }
    
    /* calc C_hat for current fish */
    float tmp_C = 0.0f;
    float dt = T / n_PERIOD;
    // input parameter opt is the Put/Call flag, 1 for Put, -1 for Call
    for (int path = 0; path < n_PATH; path++){         // sum all path C_hat
        boundary_gid = gid + path * nParticle;         // calc shared global access id for boundary_idx & exercise
        tmp_C += exp(-r * (boundary_idx[boundary_gid]+1) * dt) * max(0.0f, (K - exercise[boundary_gid]) * opt);   // boudnary_idx +1 to reflect actual time step, considering present is time zero
    }
    
    C_hat[gid] = tmp_C / n_PATH;     // get average C_hat for current fish/thread investigation
}

