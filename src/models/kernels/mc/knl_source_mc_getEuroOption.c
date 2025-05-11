#define n_PATH %d
#define n_PERIOD %d

/*
getEuroOption_optimized is the WINNER
16384 paths : getEuroOption   7.457971572875977 ms; getEuroOption_optimized 2.4869441986083984 ms; getEuroOption_optimized_sum  4.420995712280273 ms
32768 paths : getEuroOption  37.24312782287598 ms;  getEuroOption_optimized 3.8788318634033203 ms; getEuroOption_optimized_sum 18.48912239074707 ms
65536 paths : getEuroOption  51.766157150268555 ms; getEuroOption_optimized 3.778696060180664 ms;  getEuroOption_optimized_sum 19.116878509521484 ms
131072 paths: getEuroOption  93.9640998840332 ms;   getEuroOption_optimized 4.794836044311523 ms;  getEuroOption_optimized_sum 21.71611785888672 ms
262144 paths: getEuroOption 117.06280708312988 ms;  getEuroOption_optimized 7.85517692565918 ms;   getEuroOption_optimized_sum 25.4669189453125 ms
*/


__kernel void getEuroOption(__global float *Z, float S0, float K, float r, float sigma, float T, 
    char opt, __global float *payoffs){
    
    /* pre calc dt, nudt, volsdt, lnS0*/
    private float dt = T / n_PERIOD;
    private float nudt = (r - 0.5* sigma*sigma) * dt;
    private float volsdt = sigma * sqrt(dt);
    private float lnS0 = log(S0);
    
    /* one thread per path, gid is current working thread/path */
    int path_id = get_global_id(0);
    
    float St = S0;
    float last_tmp = S0;
    float deltaSt = 0.0f;
    
    // simulate log normal price
    /* loop thru n_PERIOD for each path to get maturity price */
    for(int cur_t = 0; cur_t < n_PERIOD; cur_t++){
        last_tmp = deltaSt;      // set St to lastSt for next period calc
        /* get corresponding z */
        //float z = Z[cur_t * n_PATH + path_id]; 
        float z = Z[cur_t + n_PERIOD * path_id];    // updated on 6 Apr. 2025 for unified Z, St shape as nPath by nPeriod
        deltaSt = nudt + volsdt * z;
        deltaSt += last_tmp;
    }
    St = exp(deltaSt + lnS0); 
    
    /* return C_hat of Euro option */
    float path_C_hat = exp(-r*T) * fmax(0.0f, (K - St)*opt);
    payoffs[path_id] = path_C_hat; 
}


__kernel void getEuroOption_optimized(
    __global const float *Z,
    const float lnS0,
    const float K,
    const float r,
    const float sigma,
    const float T,
    const char opt,
    const float exp_neg_rT,
    __global float *payoffs
){
    const int path_id = get_global_id(0);
    
    // Pre-calculate constants at host level and pass them as arguments
    const float dt = T / n_PERIOD;
    const float nudt = (r - 0.5f * sigma * sigma) * dt;
    const float volsdt = sigma * sqrt(dt);
    
    // Simulate path
    float deltaSt = 0.0f;
    const int z_offset = path_id * n_PERIOD;
    
    for(int cur_t = 0; cur_t < n_PERIOD; cur_t++){
        deltaSt += nudt + volsdt * Z[z_offset + cur_t];
    }

    // Single exp calculation at the end
    const float St = exp(deltaSt + lnS0);
    payoffs[path_id] = exp_neg_rT * fmax(0.0f, (K - St) * opt);   // Use fmax() instead of max() for better performance
}


/*
   Only works no exceeding 2**16 paths
*/
// Sum reduction - Two-step Reduction
// Group-wise reduction, writes partial sums (C_hats[group_id])
__kernel void getEuroOption_optimized_sum1(
    __global const float *Z,
    const float lnS0,
    const float K,
    const float r,
    const float sigma,
    const float T,
    const char opt,
    const float exp_neg_rT,
    // __global float *payoffs,
    __local float *local_sums,   // For reduction
    __global float *C_hats       // C_hat groups output
){
    const int path_id = get_global_id(0);
    const int lid = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);

    // Pre-calculate constants at host level and pass them as arguments
    const float dt = T / n_PERIOD;
    const float nudt = (r - 0.5f * sigma * sigma) * dt;
    const float volsdt = sigma * sqrt(dt);

    // Simulate path
    float deltaSt = 0.0f;
    const int z_offset = path_id * n_PERIOD;
    
    for(int cur_t = 0; cur_t < n_PERIOD; cur_t++) {
        deltaSt += nudt + volsdt * Z[z_offset + cur_t];
    }

    // Path wise exp calculation at the end, and copy data to local memory
    const float St = exp(deltaSt + lnS0);
    // payoffs[path_id] = exp_neg_rT * fmax(0.0f, (K - St) * opt);   // Use fmax() instead of max() for better performance

    // copy data to local memory
    // local_sums[lid] = payoffs[path_id];
    local_sums[lid] = exp_neg_rT * fmax(0.0f, (K - St) * opt);   // Use fmax() instead of max() for better performance;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction in local memory
    for(int stride = local_size >> 1; stride > 0; stride >>= 1){
        if(lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

   // Write group sum to global memory
    if(lid == 0) {
        C_hats[group_id] = local_sums[0]; 
    }
}

// Final reduction of C_hats → single float
__kernel void getEuroOption_optimized_sum2(
    const int n_groups,
    __global float* partial_sums,
    __local float* local_sums,
    __global float* final_result
){
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int local_size = get_local_size(0);
    
    // Load into local memory
    local_sums[lid] = (gid < n_groups) ? partial_sums[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce
    for(int stride = local_size >> 1; stride > 0; stride >>= 1){
        if(lid < stride){
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        final_result[0] = local_sums[0];  // compute the final sumation
    }
}

