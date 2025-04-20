import numpy as np
import numpy.linalg as la
import pyopencl as cl
from .mc import MonteCarloBase
from .utils import openCLEnv

import time

def checkError(a, b):
    err = a - b
    err_norm = la.norm(err)
    return err_norm

# Classic Adjoint
def inverse_3X3_matrix(A):
    I_Q_list = np.where(isinstance(A, list), A, A.tolist())
    I_Q_list = A
    
    det_ = I_Q_list[0][0] * (
            (I_Q_list[1][1] * I_Q_list[2][2]) - (I_Q_list[1][2] * I_Q_list[2][1])) - \
           I_Q_list[0][1] * (
                   (I_Q_list[1][0] * I_Q_list[2][2]) - (I_Q_list[1][2] * I_Q_list[2][0])) + \
           I_Q_list[0][2] * (
                   (I_Q_list[1][0] * I_Q_list[2][1]) - (I_Q_list[1][1] * I_Q_list[2][0]))
    
    if det_ == 0.0:
        return det_, np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(np.float32)
    
    else:
    
        co_fctr_1 = [(I_Q_list[1][1] * I_Q_list[2][2]) - (I_Q_list[1][2] * I_Q_list[2][1]),
                     -((I_Q_list[1][0] * I_Q_list[2][2]) - (I_Q_list[1][2] * I_Q_list[2][0])),
                     (I_Q_list[1][0] * I_Q_list[2][1]) - (I_Q_list[1][1] * I_Q_list[2][0])]

        co_fctr_2 = [-((I_Q_list[0][1] * I_Q_list[2][2]) - (I_Q_list[0][2] * I_Q_list[2][1])),
                     (I_Q_list[0][0] * I_Q_list[2][2]) - (I_Q_list[0][2] * I_Q_list[2][0]),
                     -((I_Q_list[0][0] * I_Q_list[2][1]) - (I_Q_list[0][1] * I_Q_list[2][0]))]

        co_fctr_3 = [(I_Q_list[0][1] * I_Q_list[1][2]) - (I_Q_list[0][2] * I_Q_list[1][1]),
                     -((I_Q_list[0][0] * I_Q_list[1][2]) - (I_Q_list[0][2] * I_Q_list[1][0])),
                     (I_Q_list[0][0] * I_Q_list[1][1]) - (I_Q_list[0][1] * I_Q_list[1][0])]

        inv_list = [[1 / det_ * (co_fctr_1[0]), 1 / det_ * (co_fctr_2[0]), 1 / det_ * (co_fctr_3[0])],
                    [1 / det_ * (co_fctr_1[1]), 1 / det_ * (co_fctr_2[1]), 1 / det_ * (co_fctr_3[1])],
                    [1 / det_ * (co_fctr_1[2]), 1 / det_ * (co_fctr_2[2]), 1 / det_ * (co_fctr_3[2])]]

        return det_.astype(np.float32), np.array(inv_list).astype(np.float32),# np.array([co_fctr_1, co_fctr_2, co_fctr_3]).astype(np.float32)

# Gauss-Jordan Elimination - row reduction
def GJ_Elimination_inverse_3X3(A):
    B = np.zeros((3,6), dtype=np.float32)
    
    # right joint identity matrix
    for i in range(3):
        for j in range(3):
          B[i][j] = A[i][j]
    
    B[0][3] = 1
    B[1][4] = 1
    B[2][5] = 1
    
    # partial pivoting
    for i in range(2, 0, -1):
        if (B[i - 1][1] < B[i][1]):
            for j in range(6):
                d = B[i][j]
                B[i][j] = B[i - 1][j]
                B[i - 1][j] = d
                
    # reducing to diagonal  matrix 
    for i in range(3):
        for j in range(3):
            if (j != i and B[j][i] != 0):
                d = B[j][i] / B[i][i]
                for k in range(6):
                    B[j][k] -= B[i][k] * d
                    
    # reducing to unit matrix 
    C = np.zeros((3,3), dtype=np.float32)

    C[0][0] = B[0][3] / B[0][0]
    C[0][1] = B[0][4] / B[0][0]
    C[0][2] = B[0][5] / B[0][0]
    C[1][0] = B[1][3] / B[1][1]
    C[1][1] = B[1][4] / B[1][1]
    C[1][2] = B[1][5] / B[1][1]
    C[2][0] = B[2][3] / B[2][2]
    C[2][1] = B[2][4] / B[2][2]
    C[2][2] = B[2][5] / B[2][2]
    
    return C

class LongStaffBase:
    def __init__(self, mc: MonteCarloBase):
        """
        Parameters
        ----------
        mc : MonteCarloBase instance which includes the following:

            mc.S0 : float
                stock spot price
            mc.r : float 
                risk free rate
            mc.sigma : float
                volatility
            mc.T : float
                duration
            mc.nPath : int
                The number of simulation paths
            mc.nPeriod : int
                The number of simulation time steps
            mc.K : float
                The strick price
            mc.opttype : char
                'P' for put; 'C' for call
            mc.opt : int8
                option operator flag, Indicates Put or Call options
            mc.St : 2D array [float]
                from Monte-Carlo simulation
        """
        self.mc = mc

class LSMC_Numpy(LongStaffBase):
    def __init__(self, mc: MonteCarloBase, inverseType='benchmark_pinv', toggleCV='OFF', log=None):
        """
            inverseType : str, optional ["benchmark_pinv", "benchmark_lstsq", "SVD", "CA", "GJ"]
                    The GPU matrix inversion method. Default value is 'GJ' using Gauss-Jordan Elimination. Can be set to
                    'CA' for Classic Adjoint
        """
        super().__init__(mc)
        self.inverseType = inverseType
        self.toggleCV = toggleCV
        self.log = log

    # calculate conditional expectation of Continuation, follow matrix linear algebra form
    def __continuation_value(self, x, Y):
        inverseType_cpu = ["benchmark_pinv", "benchmark_lstsq", "SVD", "CA", "GJ"]
        if self.inverseType not in inverseType_cpu:
            raise Exception(f'Wrong value for inverseType, can ONLY be one of {inverseType_cpu}')
            
        # poly = 2        # polynomial to generate X matrix
        # X matraix contains constant, x...x^poly   
        X = np.c_[np.ones(len(x)), x, np.square(x)].astype(np.float32)
        
        # conditional expectation function E[Y|X] = coef_[0] + coef_[1]*x + ... + coef_[poly]*x^poly
        match self.inverseType:
            case "benchmark_pinv":  
                Xdagger = np.linalg.pinv(X)     # standard Numpy function for pseudo-inverse
                coef_ = Xdagger @ Y
            case "benchmark_lstsq":
                coef_ = np.linalg.lstsq(X, Y, rcond=None)[0]
            case "SVD":
                U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
                Xdagger = VT.T @ np.linalg.inv(np.diag(Sigma)) @ U.T
                coef_ = Xdagger @ Y
            case "CA":
                Xdagger = inverse_3X3_matrix(X.T @ X)[1] @ X.T
                coef_ = Xdagger @ Y
            case "GJ":
                Xdagger = GJ_Elimination_inverse_3X3(X.T @ X) @ X.T
                coef_ = Xdagger @ Y

        if (self.log=='INFO'):
            print('X:\n',X)
            print('Y:', Y.flatten())
            print('Xdagger:\n', Xdagger)

        cont_value = X @ coef_
        return cont_value, coef_

    # lease-square to calc continuation value as approaximation of "actual discounted cashflow" 
    def longstaff_schwartz_itm_path_fast(self):
        start = time.perf_counter()
        # setup computing components
        dt = self.mc.T / self.mc.nPeriod
        df = np.exp(- self.mc.r * dt)
        
        # track discounted cashflow per time step
        dc_cashflow_t = []
        coef_t = []
        
        # immediate exercise payoffs for each path and time step
        payoffs = self.mc.getPayoffs()
        itm_allPeriod = (payoffs!=0).sum(0)
        
        # initiate cashflow as of time maturity
        dc_cashflow = payoffs[:, -1]
        
        for t in range(self.mc.nPeriod-2, -1, -1):
            if (self.log=='INFO'):
                print('\ntime t:', t+1)
            
            # 1. discount cashflow of time t+1 to time t
            dc_cashflow = dc_cashflow * df
            
            # 2. in-the-money paths & numbers
            itm = payoffs[:, t].nonzero()
            num_itm = itm_allPeriod[t] 
            
            # 3. construct x and Y
            # x = St[itm, t]           # St at time t for basis function to generate X
            x = payoffs[itm,t].reshape(num_itm, 1)          ### (K-St) at time t for basis function to generate X::Normalization to avoid nearly singular matrix
            Y = dc_cashflow[itm].reshape(num_itm, 1)   # discounted cashflow as dependent variable
            
            # 4. calc continuation values
            cont_value, coef_ = self.__continuation_value(x, Y)
            coef_t.append(coef_.flatten())
            
            if (self.log=='INFO'):
                print('coef:', coef_.flatten())
                print('pre ds cf itm :', dc_cashflow[itm])
                print('cont val      :', cont_value.flatten())
                print('exer val      :', payoffs[itm, t])
            
            # 5. update cashflow for time t
            match self.toggleCV:
                case 'ON':
                    # control variate with Black-Scholes for time step t in-the-money paths
                    BS_itm = self.mc.BS[itm, t].flatten()        
                    max_CvBS = np.maximum(cont_value.flatten(), BS_itm)
                    dc_cashflow[itm] = np.where(payoffs[itm, t] > max_CvBS, payoffs[itm, t], dc_cashflow[itm])  
                case 'OFF':
                    # follow the original Longstaff-Schwartz LSMC method
                    dc_cashflow[itm] = np.where(payoffs[itm, t] > cont_value.flatten(), payoffs[itm, t], dc_cashflow[itm])
                
            dc_cashflow_t.append(dc_cashflow)  # insert current time step discounted cashflow
            
            if (self.log=='INFO'):
                print('post ds cf itm:', dc_cashflow)     

        # discount cashflow to time zero for option pricing
        C_hat = np.sum(dc_cashflow) * df / self.mc.nPath

        elapse = (time.perf_counter() - start) * 1e3
        print(f'Longstaff numpy price: {C_hat} - {elapse} ms')
        return C_hat, elapse, #np.array(coef_t), np.array(dc_cashflow_t)

class LSMC_OpenCL(LongStaffBase):
    def __init__(self, mc: MonteCarloBase, preCalc=None, inverseType='GJ', toggleCV='OFF', log=None):
        """
            inverseType : str, optional ["CA", "GJ"]
                    The GPU matrix inversion method. Default value is 'GJ' using Gauss-Jordan Elimination. Can be set to
                    'CA' for Classic Adjoint
        """
        super().__init__(mc)
        self.preCalc = preCalc
        self.inverseType = inverseType
        self.toggleCV = toggleCV
        self.log = log

        # polynomial to the power 2
        self.stride = 3
    
    # calculate to generate Xdagger, pseudo-inverse of all Xs (each X per period)
    def __preCalc_gpu(self):     # St in shape of nPath by nPeriod    
        inverseType_gpu = ["CA", "GJ"]
        if self.inverseType not in inverseType_gpu:
            raise Exception(f'Wrong value for inverseType, can ONLY be one of {inverseType_gpu}')
        
        # set up buffer on device
        St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mc.St.flatten())

        X_big_T = np.zeros((self.stride * self.mc.nPeriod, self.mc.nPath), dtype=np.float32)
        X_big_T_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE, size=X_big_T.nbytes)

        Xdagger_big = np.zeros((self.stride * self.mc.nPeriod, self.mc.nPath), dtype=np.float32)
        Xdagger_big_d = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=Xdagger_big.nbytes)
        
        match self.inverseType:
            case "GJ": 
                prog = cl.Program(openCLEnv.context, open("./models/kernels/knl_src_pre_calc_GaussJordan.c").read()%(self.mc.nPath, self.mc.nPeriod)).build()
                knl_preCalcAll = cl.Kernel(prog, 'preCalcAll_GaussJordan')
            case "CA":
                prog = cl.Program(openCLEnv.context, open('./models/kernels/knl_src_pre_calc_ClassicAdjoint.c').read()%(self.mc.nPath, self.mc.nPeriod)).build()
                knl_preCalcAll = cl.Kernel(prog, 'preCalcAll_ClassicAdjoint')

        # kernel run
        knl_preCalcAll.set_args(St_d, np.float32(self.mc.K), np.int8(self.mc.opt), X_big_T_d, Xdagger_big_d)
        
        global_size = (self.mc.nPeriod, )
        local_size = None
        evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, knl_preCalcAll, global_size, local_size)
        evt.wait()

        cl.enqueue_copy(openCLEnv.queue, X_big_T, X_big_T_d)
        cl.enqueue_copy(openCLEnv.queue, Xdagger_big, Xdagger_big_d)
        
        openCLEnv.queue.finish()
            
        # release memory
        St_d.release()
        X_big_T_d.release()
        Xdagger_big_d.release()

        return Xdagger_big , X_big_T

    def __preCalc_gpu_optimized(self):
        inverseType_gpu = ["CA", "GJ"]
        if self.inverseType not in inverseType_gpu:
            raise Exception(f'Wrong value for inverseType, can ONLY be one of {inverseType_gpu}')
        
        # Use pinned memory for faster transfers
        St_host = np.ascontiguousarray(self.mc.St.flatten(), dtype=np.float32)
        X_big_T = np.zeros((self.stride * self.mc.nPeriod, self.mc.nPath), dtype=np.float32)
        Xdagger_big = np.zeros_like(X_big_T)

        # Create buffers with optimal flags
        St_dev = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=St_host)
        X_big_T_dev = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE, size=X_big_T.nbytes)
        Xdagger_big_dev = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=Xdagger_big.nbytes)

        # Build kernel with optimization flags
        kernel_src = open(f"./models/kernels/knl_src_pre_calc_optimized.c").read()
        # build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", f"-Dn_PATH={self.mc.nPath}", f"-Dn_PERIOD={self.mc.nPeriod}"]
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros"]
        prog = cl.Program(openCLEnv.context, kernel_src %(self.mc.nPath, self.mc.nPeriod)).build(options=build_options)
        # prog = cl.Program(openCLEnv.context, kernel_src ).build(options=build_options)
        
        knl_preCalcAll = cl.Kernel(prog, 'preCalcAll_Optimized')
        knl_preCalcAll.set_args(St_dev, np.float32(self.mc.K), np.int8(self.mc.opt), 
                                # np.int32(self.mc.nPath), np.int32(self.mc.nPeriod),
                                X_big_T_dev, Xdagger_big_dev)

        # Execute with optimal workgroup size
        max_wg_size = openCLEnv.device.max_work_group_size
        wg_size = min(256, max_wg_size)  # Optimal for matrix operations
        global_size = (self.mc.nPeriod,)
        local_size = (wg_size,) if wg_size <= self.mc.nPeriod else None
        # local_size = None
        # print(local_size)

        # Execute kernel with profiling
        evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, knl_preCalcAll, global_size, local_size)
        
        # Overlap copy with computation
        copy_evt1 = cl.enqueue_copy(openCLEnv.queue, X_big_T, X_big_T_dev, wait_for=[evt])
        copy_evt2 = cl.enqueue_copy(openCLEnv.queue, Xdagger_big, Xdagger_big_dev, wait_for=[evt])
        copy_evt2.wait()

        # Release resources
        for buf in [St_dev, X_big_T_dev, Xdagger_big_dev]:
            buf.release()

        return Xdagger_big, X_big_T

    def longstaff_schwartz_itm_path_fast_hybrid(self):
        start = time.perf_counter()
        # track discounted cashflow per time step
        dc_cashflow_t = []
        coef_t = []
 
        # setup computing components
        dt = self.mc.T / self.mc.nPeriod
        df = np.exp(- self.mc.r * dt)
        
        # immediate exercise payoffs for each path and time step
        payoffs = self.mc.getPayoffs()
        itm_allPeriod = (payoffs!=0).sum(0)
        
        # initiate cashflow as of time maturity
        dc_cashflow = payoffs[:, -1]
        
        # Pre-calc
        if self.preCalc == None:
            Xdagger_big, X_big_T = self.__preCalc_gpu()
        elif self.preCalc == 'optimized':
            Xdagger_big, X_big_T = self.__preCalc_gpu_optimized()
        
        for t in range(self.mc.nPeriod-2, -1, -1):
            if (self.log=='INFO'):
                print('\ntime t:', t+1)
            
            # 1. discount cashflow of time t+1 to time t
            dc_cashflow = dc_cashflow * df
            
            # 2. in-the-money paths & numbers
            itm = payoffs[:, t].nonzero()
            num_itm = itm_allPeriod[t]
            
            # 3. construct X and Y
            X = X_big_T[t*self.stride : t*self.stride+self.stride, :num_itm].T #.reshape(num_itm, 3)

            Xdagger = Xdagger_big[t*self.stride : t*self.stride+self.stride, :num_itm] #.reshape(3, num_itm) 
            
            Y = dc_cashflow[itm].reshape(num_itm, 1)   # discounted cashflow as dependent variable
            coef_ = Xdagger @ Y
            coef_t.append(coef_.flatten())
            
            # 4. calc continuation values
            cont_value = X @ coef_
            
            if (self.log=='INFO'):
                print('X:\n',X)
                print('Y:', Y.flatten())
                print('Xdagger:\n', Xdagger)
                print('coef:', coef_.flatten())
                print('pre ds cf itm :', dc_cashflow[itm])
                print('cont val      :', cont_value.flatten())
                print('exer val      :', payoffs[:, t][itm])
        
            # 5. update cashflow for time t
            if self.toggleCV=='ON':
                # control variate with Black-Scholes for time step t in-the-money paths
                BS_itm = self.mc.BS[itm, t].flatten()        
                max_CvBS = np.maximum(cont_value.flatten(), BS_itm)
                dc_cashflow[itm] = np.where(payoffs[:, t][itm] > max_CvBS, payoffs[:,t][itm], dc_cashflow[itm])  
            elif self.toggleCV=='OFF':
                # follow the original Longstaff-Schwartz LSMC method
                dc_cashflow[itm] = np.where(payoffs[:, t][itm] > cont_value.flatten(), payoffs[:,t][itm], dc_cashflow[itm])  
                
            dc_cashflow_t.append(dc_cashflow)  # insert current time step discounted cashflow
            
            if (self.log=='INFO'):
                print('post ds cf itm:', dc_cashflow)     

        # discount cashflow to time zero for option pricing
        C_hat = np.sum(dc_cashflow) * df / self.mc.nPath

        elapse = (time.perf_counter() - start) * 1e3
        print(f'Longstaff {openCLEnv.deviceName} price: {C_hat} - {elapse} ms')
        return C_hat, elapse, # np.array(coef_t), np.array(dc_cashflow_t)


def main():
    print("longstaff.py")

if __name__ == "__main__":
    main()