
#define n_PATH %d
#define n_PERIOD %d

#ifndef STRIDE
#define STRIDE 3
#endif


__kernel void preCalcAll_ClassicAdjoint(global float* St, float K, char opt, global float* X_big_T, global float* Xdagger_big){

    int gid = get_global_id(0); //period id
    
    if (gid >= n_PERIOD)
        return;
    
    /* 1. generate X_big & 2. generate X_big_T */
    // calc start
    int XbigT_start_idx = gid * STRIDE * n_PATH; 
    
    int itm = 0;
    
    float X0 = 0.0f;
    float X1 = 0.0f;
    float X2 = 0.0f;
    
    for (int i=0; i<n_PATH; i++){
        
        /* 0. generate X */        
        //X0 = 1.0f;
        //X1 = (K - St[gid + i * n_PERIOD]) * opt;
        //X2 = X1 * X1;
        
        //check in-the-money paths
        float payoff = max(0.0f, (K - St[gid + i * n_PERIOD]) * opt);
        
        if (payoff > 0){  
            
            /* 1. generate X_big */
            // compute on-the-fly, no need for storage
            
            /* 2. generate X_big_T */
                        //use positive payoff as the basis functions:::Normalization to avoid nearly singular matrix
            X_big_T[XbigT_start_idx + itm + 0 * n_PATH] = 1.0f;
            X_big_T[XbigT_start_idx + itm + 1 * n_PATH] = payoff;
            X_big_T[XbigT_start_idx + itm + 2 * n_PATH] = payoff * payoff;
            
            itm++;
        }
    }
    
    /* 3. each thread calc XTX of per period in private memory */
    
    float XTX[STRIDE * STRIDE] = {0.0f};
    // XTX is diognal, only need to calc upper triangle - 6 values: 0, 1, 2, 4, 5, 8
    
    // calc XTX[0, 1, 2, 4, 5, 8]
    for (int i=0; i<itm; i++){
        
        X0 = X_big_T[XbigT_start_idx + i + 0 * n_PATH];
        X1 = X_big_T[XbigT_start_idx + i + 1 * n_PATH];
        X2 = X_big_T[XbigT_start_idx + i + 2 * n_PATH];
        
        XTX[0] += X0 * X0; 
        XTX[1] += X0 * X1; 
        XTX[2] += X0 * X2; 
        XTX[4] += X1 * X1; 
        XTX[5] += X1 * X2; 
        XTX[8] += X2 * X2; 
    }
    
    XTX[3] = XTX[1];
    XTX[6] = XTX[2];
    XTX[7] = XTX[5];
    
    /* 4. calc XTX inverse */

    /** Classic Adjoint inverse 3X3 **/ 
    float XTX_inv[STRIDE * STRIDE] = {0.0f};
  
    // a. Calculate the determinant of the matrix 
    float det = XTX[0] * (XTX[4] * XTX[8] - XTX[5] * XTX[7])
              - XTX[1] * (XTX[3] * XTX[8] - XTX[5] * XTX[6])
              + XTX[2] * (XTX[3] * XTX[7] - XTX[4] * XTX[6]);

    // b. Check if the determinant is non-zero
    if (det != 0.0f) {       // Calculate the inverse using cofactors and adjugate
        XTX_inv[0] = (XTX[4] * XTX[8] - XTX[5] * XTX[7]) * 1.0 / det;
        XTX_inv[1] = (XTX[2] * XTX[7] - XTX[1] * XTX[8]) * 1.0 / det;
        XTX_inv[2] = (XTX[1] * XTX[5] - XTX[2] * XTX[4]) * 1.0 / det;
        XTX_inv[3] = (XTX[5] * XTX[6] - XTX[3] * XTX[8]) * 1.0 / det;
        XTX_inv[4] = (XTX[0] * XTX[8] - XTX[2] * XTX[6]) * 1.0 / det;
        XTX_inv[5] = (XTX[2] * XTX[3] - XTX[0] * XTX[5]) * 1.0 / det;
        XTX_inv[6] = (XTX[3] * XTX[7] - XTX[4] * XTX[6]) * 1.0 / det;
        XTX_inv[7] = (XTX[1] * XTX[6] - XTX[0] * XTX[7]) * 1.0 / det;
        XTX_inv[8] = (XTX[0] * XTX[4] - XTX[1] * XTX[3]) * 1.0 / det;

    }
    else {        // Matrix is singular, set the inverse to identity matrix
        XTX_inv[0] = 1.0f;
        XTX_inv[1] = 0.0f;
        XTX_inv[2] = 0.0f;
        XTX_inv[3] = 0.0f;
        XTX_inv[4] = 1.0f;
        XTX_inv[5] = 0.0f;
        XTX_inv[6] = 0.0f;
        XTX_inv[7] = 0.0f;
        XTX_inv[8] = 1.0f;
    } 
   
    /* 5 write to XTX_big & XTXinv_big */
    // compute on-the-fly, no need for storage
    
    /* 6. calc Xdagger_big */
    // calc starting index
    int Xdagger_big_start_idx = gid * STRIDE * n_PATH; 
    
    for (int i=0; i<itm; i++){  // loop over paths
    
        X0 = 0.0f;
        X1 = 0.0f;
        X2 = 0.0f;
        
        for (int k=0; k<STRIDE; k++){  //  loop over STRIDE
            //k=0: XTX_inv[0,1,2]; k=1: XTX_inv[3,4,5]; k=2: XTX_inv[6,7,8]            
            X0 += XTX_inv[0 * STRIDE + k] * X_big_T[XbigT_start_idx + i + k * n_PATH];
            X1 += XTX_inv[1 * STRIDE + k] * X_big_T[XbigT_start_idx + i + k * n_PATH];
            X2 += XTX_inv[2 * STRIDE + k] * X_big_T[XbigT_start_idx + i + k * n_PATH];            
        }
        
        Xdagger_big[Xdagger_big_start_idx + i + 0 * n_PATH] = X0;
        Xdagger_big[Xdagger_big_start_idx + i + 1 * n_PATH] = X1;
        Xdagger_big[Xdagger_big_start_idx + i + 2 * n_PATH] = X2;

    }
}
