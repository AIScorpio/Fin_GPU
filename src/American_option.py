from models.mc import MonteCarloBase, hybridMonteCarlo
from models.longstaff import LSMC_Numpy, LSMC_OpenCL
import models.benchmarks as bm
from models.pso import PSO_Numpy, PSO_OpenCL_hybrid, PSO_OpenCL_scalar, PSO_OpenCL_scalar_fusion, PSO_OpenCL_vec, PSO_OpenCL_vec_fusion
from models.utils import checkOpenCL
    

if __name__ == "__main__":
    checkOpenCL()

    # S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 20000, 2**8, 100.0, 'P', 2**10
    S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 2**14, 2**8, 110.0, 'P', 2**10
    # S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 32000, 200, 122.0, 'P', 4
    mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
    # print(f'{nPath} paths, {nPeriod} periods, {nFish} particles.\n')

    # print(f'Initial St in shape {mc.St.shape}\n{mc.St}')

    # benchmarks
    print("Benchmark ====================")
    bm.blackScholes(S0, K, r, sigma, T, opttype)
    bm.binomialEuroOption(S0, K, r, sigma, nPeriod, T, opttype)
    mc.getEuroOption_np()
    mc.getEuroOption_cl_optimized()
    bm.binomialAmericanOption(S0, K, r, sigma, nPeriod, T, opttype)

    # longstaff
    print("\nLSMC ====================")
    lsmc_np = LSMC_Numpy(mc)
    lsmc_np.longstaff_schwartz_itm_path_fast()

    lsmc_cl = LSMC_OpenCL(mc, preCalc="optimized", inverseType='CA')
    lsmc_cl.longstaff_schwartz_itm_path_fast_hybrid()

    lsmc_cl = LSMC_OpenCL(mc, preCalc=None, inverseType='CA')
    lsmc_cl.longstaff_schwartz_itm_path_fast_hybrid()

    lsmc_cl = LSMC_OpenCL(mc, preCalc=None, inverseType='GJ')
    lsmc_cl.longstaff_schwartz_itm_path_fast_hybrid()

    # pso
    print("\nPSO ====================")
    pso_np = PSO_Numpy(mc, nFish)
    pso_np.solvePsoAmerOption_np()

    pso_cl_hybrid = PSO_OpenCL_hybrid(mc, nFish)
    pso_cl_hybrid.solvePsoAmerOption_cl()

    pso_cl = PSO_OpenCL_scalar(mc, nFish, direction='forward')
    pso_cl.solvePsoAmerOption_cl()

    pso_cl = PSO_OpenCL_scalar(mc, nFish, direction='backward')
    pso_cl.solvePsoAmerOption_cl()

    pso_cl2 = PSO_OpenCL_scalar_fusion(mc, nFish)
    pso_cl2.solvePsoAmerOption_cl()

    pso_cl_vec = PSO_OpenCL_vec(mc, nFish, vec_size=1)
    pso_cl_vec.solvePsoAmerOption_cl()

    pso_cl_vec = PSO_OpenCL_vec(mc, nFish, vec_size=2)
    pso_cl_vec.solvePsoAmerOption_cl()

    pso_cl_vec = PSO_OpenCL_vec(mc, nFish, vec_size=4)
    pso_cl_vec.solvePsoAmerOption_cl()

    pso_cl_vec = PSO_OpenCL_vec(mc, nFish, vec_size=8)
    pso_cl_vec.solvePsoAmerOption_cl()

    pso_cl_vec = PSO_OpenCL_vec(mc, nFish, vec_size=16)
    pso_cl_vec.solvePsoAmerOption_cl()

    pso_cl_vec = PSO_OpenCL_vec_fusion(mc, nFish)
    pso_cl_vec.solvePsoAmerOption_cl()

    # clear up memory
    mc.cleanUp()
    
    