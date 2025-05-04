#define n_PATH %d
#define n_PERIOD %d

// 编译时通过-DVEC_SIZE=4传入向量宽度, 默认float4
#ifndef VEC_SIZE
#define VEC_SIZE 4
#endif

// 动态定义向量类型和转换函数
#if VEC_SIZE == 1
    typedef float float_vec;
    typedef int int_vec;
    #define convert_float_vec(x) ((float)(x))
    #define SUM_VEC(v) (v)
#elif VEC_SIZE == 2
    typedef float2 float_vec;
    typedef int2 int_vec;
    #define convert_float_vec convert_float2
    #define SUM_VEC(v) (v.s0 + v.s1)
#elif VEC_SIZE == 4
    typedef float4 float_vec;
    typedef int4 int_vec;
    #define convert_float_vec convert_float4
    #define SUM_VEC(v) (v.s0 + v.s1 + v.s2 + v.s3)
#elif VEC_SIZE == 8
    typedef float8 float_vec;
    typedef int8 int_vec;
    #define convert_float_vec convert_float8
    #define SUM_VEC(v) (v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7)
#elif VEC_SIZE == 16
    typedef float16 float_vec;
    typedef int16 int_vec;
    #define convert_float_vec convert_float16
    #define SUM_VEC(v) (v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7 + \
                       v.s8 + v.s9 + v.sA + v.sB + v.sC + v.sD + v.sE + v.sF)
#else
    #error "Unsupported VEC_SIZE"
#endif

#define n_VecPath ((n_PATH + VEC_SIZE - 1) / VEC_SIZE)

/* 
        # # udpated on 6 Apr. 2025 
        # 1. for unified Z, St shape as [nPath by nPeriod], synced and shared by PSO and Longstaff
        # 2. No concatenation of spot price
        # 3. handle index of time period, spot price at time zero (present), St from time 1 to T
*/
__kernel void psoAmerOption_gb(
    __global const float *St, 
    __global const float *pso,        // the updated position matrix, [nDim by nFish] 
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

    /* move this initalization to host, each thread only need to init once */
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
        cur_fish_val = pso[gid + prd * nParticle];    // PSO global index & value

        //inner loop thru all St paths at current period
        //St global pointer from 0 to (nPath * nPeriod -1)
        for (int path= 0; path < n_PATH; path++){
            //e.g. total 3 periods, nPeriod = 3
            //prd: 2  path:[0, 1, 2, 3] --> 2 + [0, 1, 2, 3] * nPeriod = St global index [2, 5, 8, 11] for period 2
            //prd: 1  path:[0, 1, 2, 3] --> 1 + [0, 1, 2, 3] * nPeriod = St global index [1, 4, 7, 10] for period 2
            cur_St_val = St[prd + path * n_PERIOD];    // get St path value at same period
            
            // each fish access to corresponding column of boundary_idx and exercise matrix, both nPath by nFish/nParticle
            // calc shared global access id for boundary_idx & exercise
            // e.g. total 5 fishes, nParticle=5, total 10 paths, nPath=10
            // gid=0: --> 0 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * nParticle = boundary_idx/exercise global index [0, 5, 10, 15, 20..45]
            // gid=1: --> 1 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * nParticle = boundary_idx/exercise global index [1, 6, 11, 16, 21..46]
            boundary_gid = gid + path * nParticle;  
            
            /* use private float for comparison interim; consider make PATH outer loop */
            // check if first cross: 1) pso > St; 2) corresponding boundary index not set from previous loops
            if (cur_fish_val >= cur_St_val){  
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
    
    tmp_C = tmp_C / n_PATH;
    C_hat[gid] = tmp_C;     // get average C_hat for current fish/thread investigation
}


__kernel void psoAmerOption_gb2(
    __global const float *St, 
    __global const float *pso,        // the updated position matrix, [nDim by nFish] 
    __global float *C_hat, 
    const float r, 
    const float T, 
    const float K, 
    const char opt
){
    
    //global variables, one fish per thread (work-item), check for early aross, for all paths
    int gid = get_global_id(0);            //thread id, per fish
    int nParticle = get_global_size(0);    //number of fishes
    
    float dt = T / n_PERIOD;
    float tmp_cost = 0.0f; 

    for (int path=0; path<n_PATH; path++){
        int bound_idx = n_PERIOD - 1;            // init to last period
        int St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;
        float early_excise = St[St_T_idx];       // init to St path_i last period price

        // int path_boundary_id = gid + path * nParticle; 

        for (int prd=0; prd<n_PERIOD; prd++){
            float cur_fish_val = pso[gid + prd * nParticle];
            float cur_St_val = St[prd + path * n_PERIOD];

            // if first cross then break
            // bound_idx = cur_fish_val >= cur_St_val ? prd : bound_idx;
            // early_excise =  cur_fish_val >= cur_St_val ? cur_St_val : early_excise;
            if (cur_fish_val >= cur_St_val){
                bound_idx = prd;
                early_excise = cur_St_val;
                break;
            }
        }

        // compute current path present value of simulated American option; then cumulate for average later
        tmp_cost += exp(-r * (bound_idx+1) * dt) * max(0.0f, (K - early_excise)*opt); 
    }
    
    tmp_cost = tmp_cost / n_PATH;    // get average C_hat for current fish/thread investigation
    C_hat[gid] = tmp_cost;     
}


__kernel void psoAmerOption_gb3(
    __global const float *St, 
    __global const float *pso,        // the updated position matrix, [nDim by nFish] 
    __global float *C_hat, 
    const float r, 
    const float T, 
    const float K, 
    const char opt
){
    
    //global variables, one fish per thread (work-item), check for early aross, for all paths
    int gid = get_global_id(0);            //thread id, per fish
    int nParticle = get_global_size(0);    //number of fishes
    
    float dt = T / n_PERIOD;
    float tmp_cost = 0.0f;

    for (int path=0; path<n_PATH; path++){
        int bound_idx = n_PERIOD - 1;            // init to last period
        int St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;
        float early_excise = St[St_T_idx];       // init to St path_i last period price

        for (int prd=n_PERIOD-1; prd>-1; prd--){
            float cur_fish_val = pso[gid + prd * nParticle];
            float cur_St_val = St[prd + path * n_PERIOD];

            // check early cross exhaust all periods
            bound_idx = select(bound_idx, prd, isgreaterequal(cur_fish_val, cur_St_val));               // a>b? a:b will be select(b, a, a>b), mind the sequence!!
            early_excise = select(early_excise, cur_St_val, isgreaterequal(cur_fish_val, cur_St_val));

        }

        // compute current path present value of simulated American option; then cumulate for average later
        tmp_cost += exp(-r * (bound_idx+1) * dt) * max(0.0f, (K - early_excise)*opt); 
    }
    
    tmp_cost = tmp_cost / n_PATH;    // get average C_hat for current fish/thread investigation
    C_hat[gid] = tmp_cost;     
}


__kernel void psoAmerOption_gb3_vec(
    __global const float_vec *St,        // 向量化后的St，布局为[nPeriod, nVecPath]
    __global const float *pso,        // 位置矩阵 [nDim, nFish] 
    __global float *C_hat,            // 输出成本 [nFish]
    const float r, 
    const float T, 
    const float K, 
    const char opt
){
    
    //global variables, one fish per thread (work-item), check for early aross, for all paths
    int gid = get_global_id(0);            //thread id, per fish
    int nParticle = get_global_size(0);    //number of fishes
    
    float dt = T / n_PERIOD;
    float tmp_cost = 0.0f;

    // 每个线程处理VEC_SIZE个路径
    for (int vec_path=0; vec_path<n_VecPath; vec_path++){
        int_vec bound_idx = (int_vec)(n_PERIOD - 1);            // init to last period
        int St_T_idx = (n_PERIOD-1) * n_VecPath + vec_path;
        float_vec early_excise = St[St_T_idx];               // init to St path_i last period price

        #pragma unroll 4
        for (int prd=n_PERIOD-1; prd>-1; prd--){
            float cur_fish_val = pso[gid + prd * nParticle];
            float_vec cur_St_val = St[vec_path + prd * n_VecPath];

            // 向量化比较更新
            int_vec cmp_mask = isgreaterequal((float_vec)cur_fish_val, cur_St_val);
            bound_idx = select(bound_idx, (int_vec)prd, cmp_mask);               // a>b? a:b will be select(b, a, a>b), mind the sequence!!
            early_excise = select(early_excise, cur_St_val, cmp_mask);

        }

        // compute current path present value of simulated American option; then cumulate for average later
        // 计算当前向量路径组的期权价值
        float_vec payoffs = max((float_vec)(0.0f), (K - early_excise) * opt);
        float_vec discounts = exp(-r * (convert_float_vec(bound_idx) + 1) * dt);
        float_vec payoff_discount = payoffs * discounts;
        
        // 手动累加4个路径的值到标量tmp_cost
        tmp_cost += SUM_VEC(payoff_discount);
    }
    
    C_hat[gid] = tmp_cost / n_PATH;  // 平均化处理
}


