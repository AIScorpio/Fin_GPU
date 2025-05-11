#define n_PATH %d
#define n_PERIOD %d

#ifndef STRIDE
#define STRIDE 3
#endif


// Optimized matrix inversion - branchless version
void mat3x3_inverse_optimized(const float XTX[9], float XTX_inv[9]) {
    float det = XTX[0]*(XTX[4]*XTX[8] - XTX[5]*XTX[7]) 
              - XTX[1]*(XTX[3]*XTX[8] - XTX[5]*XTX[6]) 
              + XTX[2]*(XTX[3]*XTX[7] - XTX[4]*XTX[6]);

    float inv_det = 1.0f / (det + 1e-15f); // Avoid division by zero
    
    XTX_inv[0] = (XTX[4]*XTX[8] - XTX[7]*XTX[5]) * inv_det;
    XTX_inv[1] = (XTX[2]*XTX[7] - XTX[1]*XTX[8]) * inv_det;
    XTX_inv[2] = (XTX[1]*XTX[5] - XTX[2]*XTX[4]) * inv_det;
    XTX_inv[3] = (XTX[5]*XTX[6] - XTX[3]*XTX[8]) * inv_det;
    XTX_inv[4] = (XTX[0]*XTX[8] - XTX[2]*XTX[6]) * inv_det;
    XTX_inv[5] = (XTX[2]*XTX[3] - XTX[0]*XTX[5]) * inv_det;
    XTX_inv[6] = (XTX[3]*XTX[7] - XTX[6]*XTX[4]) * inv_det;
    XTX_inv[7] = (XTX[1]*XTX[6] - XTX[0]*XTX[7]) * inv_det;
    XTX_inv[8] = (XTX[0]*XTX[4] - XTX[1]*XTX[3]) * inv_det;
}

__kernel void preCalcAll_Optimized(
    __global const float* St,
    const float K,
    const char opt,
    // const int n_PATH,
    // const int n_PERIOD,
    __global float* X_big_T,
    __global float* Xdagger_big)
{
    int gid = get_global_id(0);
    if (gid >= n_PERIOD) return;

    const int X_offset = gid * STRIDE * n_PATH;
    int itm = 0;

    // Phase 1: Process ITM paths with early exit
    for (int i = 0; i < n_PATH && itm < n_PATH; i++) {
        float payoff = fmax(0.0f, (K - St[gid + i*n_PERIOD]) * opt);
        int is_itm = (payoff > 1e-6f); // Branchless comparison
        
        X_big_T[X_offset + itm + 0*n_PATH] = is_itm * 1.0f;
        X_big_T[X_offset + itm + 1*n_PATH] = is_itm * payoff;
        X_big_T[X_offset + itm + 2*n_PATH] = is_itm * payoff * payoff;
        
        itm += is_itm;
    }

    // Phase 2: Compute XTX matrix - optimized partial unrolling
    float XTX[9] = {0.0f};
    for (int i = 0; i < itm; i++) {
        float X0 = X_big_T[X_offset + i + 0*n_PATH];
        float X1 = X_big_T[X_offset + i + 1*n_PATH];
        float X2 = X_big_T[X_offset + i + 2*n_PATH];
        
        // Manually unrolled computations
        XTX[0] = mad(X0, X0, XTX[0]);
        XTX[1] = mad(X0, X1, XTX[1]);
        XTX[2] = mad(X0, X2, XTX[2]);
        XTX[4] = mad(X1, X1, XTX[4]);
        XTX[5] = mad(X1, X2, XTX[5]);
        XTX[8] = mad(X2, X2, XTX[8]);
    }
    XTX[3] = XTX[1]; 
    XTX[6] = XTX[2]; 
    XTX[7] = XTX[5];

    // Phase 3: Matrix inversion
    float XTX_inv[9];
    mat3x3_inverse_optimized(XTX, XTX_inv);

    // Phase 4: Compute Xdagger with loop unrolling
    for (int i = 0; i < itm; i++) {
        float X[3] = {
            X_big_T[X_offset + i + 0*n_PATH],
            X_big_T[X_offset + i + 1*n_PATH],
            X_big_T[X_offset + i + 2*n_PATH]
        };
        
        Xdagger_big[X_offset + i + 0*n_PATH] = 
            mad(XTX_inv[0], X[0], 
            mad(XTX_inv[1], X[1], 
            XTX_inv[2] * X[2]));
            
        Xdagger_big[X_offset + i + 1*n_PATH] = 
            mad(XTX_inv[3], X[0], 
            mad(XTX_inv[4], X[1], 
            XTX_inv[5] * X[2]));
            
        Xdagger_big[X_offset + i + 2*n_PATH] = 
            mad(XTX_inv[6], X[0], 
            mad(XTX_inv[7], X[1], 
            XTX_inv[8] * X[2]));
    }
}