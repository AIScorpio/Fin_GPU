import numpy as np
import pyopencl as cl
from .mc import MonteCarloBase
import matplotlib.pyplot as plt
from .utils import openCLEnv

import time 

class PSOBase:
    # const
    _w = 0.5
    _c1 = 0.5
    _c2 = 0.5
    _criteria = 1e-6
    def __init__(self, mc: MonteCarloBase, nFish):
        self.mc = mc
        self.nDim = self.mc.nPeriod
        self.nFish = nFish
        self.dt = self.mc.T / self.mc.nPeriod


class PSO_Numpy(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        # self.fitFunc_vectorized = np.vectorize(fitFunc, signature='(n)->()')
        self.fitFunc_vectorized = np.vectorize(self._costPsoAmerOption_np, signature='(n)->()')

        # init swarm particles positions & velocity    (nDim, nFish)
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2

        # init particles costs          (nFish,)
        self.costs = np.zeros((nFish, ), dtype=np.float32)
        
        # init personal best (position & cost)
        self.pbest_costs = self.costs.copy()     # (nFish,) 
        self.pbest_pos = self.position.copy()    # (nDim, nFish) each particle has its persional best pos by dimension
        
        # init global best (position & cost)       
        gid = np.argmax(self.pbest_costs)         # find index for global optimal 
        self.gbest_cost = self.pbest_costs[gid]   # np.float32
        self.gbest_pos = self.pbest_pos[:,gid]#.reshape(self.nDim, 1)   # (nDim, 1) reshape to col vector
        
        # create array best global cost storage for each iteration
        self.BestCosts = np.array([])

    def _searchGrid(self):
        # update velocity  
        self.velocity = self._w * self.velocity + self._c1*self.r1*(self.pbest_pos - self.position) + \
                    self._c2*self.r2*(self.gbest_pos.reshape(self.nDim, 1) - self.position)
        # out_v = out_v.clip(vMin, vMax)       # bound velocity
        # update position
        self.position += self.velocity 
        # out_p = out_p.clip(pMin, pMax)       # bound position
        # print(self.position)
        return 

    # Fitness function: Monte Carlo pso American option - CPU: take one particle each time and loop thru PSO
    def _costPsoAmerOption_np(self, in_particle):
        # # udpated on 6 Apr. 2025 
        # 1. for unified Z, St shape as [nPath, nPeriod], synced and shared by PSO and Longstaff
        # 2. No concatenation of spot price
        # 3. handle index of time period, spot price at time zero (present), St from time 1 to T

        # get the boundary index where early cross (particle period > St period), as if an early exercise judgement by this fish/particle
        boundaryIdx = np.argmax(self.mc.St < in_particle[None, :], axis=1)   # [0, 1] as of true or false of early cross

        # if no, set boundary index to last time period, meaning no early exercise suggested for that path
        boundaryIdx[boundaryIdx==0] = self.mc.nPeriod - 1    # to handle time T index for boundary index to match St time wise dimension (i.e. indexing from zero)

        # determine exercise prices by getting the early cross St_ij on path i and period j
        exerciseSt = self.mc.St[np.arange(len(boundaryIdx)), boundaryIdx]    # len of boundaryIdx is nPath
        # print(f'pos numpy: boundary & exercise: {boundaryIdx}, {exerciseSt}\n')
        # discounted back to time zero, hence boundaryIdx+1
        searchCost = (np.exp(-self.mc.r * (boundaryIdx+1) * self.dt) * np.maximum(0, (self.mc.K - exerciseSt)*self.mc.opt) ).sum() / self.mc.nPath
        
        return searchCost

    def solvePsoAmerOption_np(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move             
            t = time.perf_counter()             
            self._searchGrid()
            search.append((time.perf_counter()- t) * 1e3)

            # 2. recalculate fitness/cost 
            t = time.perf_counter()
            self.costs = self.fitFunc_vectorized( np.transpose(self.position) ).astype(np.float32)
            fit.append((time.perf_counter()- t) * 1e3)

            # 3. update pbest
            t = time.perf_counter()
            mask = np.greater(self.costs, self.pbest_costs)    # numpy vectorized comparison
            self.pbest_costs[mask] = self.costs[mask]
            self.pbest_pos[:,mask] = self.position[:,mask]
            
            # 4. update gbest        
            gid = np.argmax(self.pbest_costs)
            if self.pbest_costs[gid] > self.gbest_cost:  # compare with global best
                self.gbest_cost = self.pbest_costs[gid]
                self.gbest_pos = self.pbest_pos[:,gid]#.reshape(self.nDim, 1)   # (nDim, 1) reshape to col vector
                
            # 5. record global best cost for current iteration
            self.BestCosts = np.concatenate( (self.BestCosts, [self.gbest_cost]) )
            rest.append((time.perf_counter()- t) * 1e3)
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.perf_counter()- start) * 1e3
        print(f'Pso numpy price    : {C_hat} - {elapse} ms')
        return C_hat, elapse, search, fit, rest


class PSO_OpenCL_hybrid(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax

        # init swarm particles positions & velocity    (nDim, nFish)
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.position)
        self.vel_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.velocity)

        # init r1, r2 on device
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2
        self.r1_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r1)
        self.r2_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r2)

        # init fitness buffer
        self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mc.St)
        self.boundary_idx = np.empty(shape=(self.mc.nPath, nFish), dtype=np.int32) #+ nPeriod
        self.exercise = np.empty(shape=(self.mc.nPath, nFish), dtype=np.float32) #+ self.St[:, -1].reshape(nPath, 1)
        self.boundary_idx_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.boundary_idx)
        self.exercise_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.exercise)

        # init particles costs          (nFish,)
        self.costs = np.zeros((self.nFish,), dtype=np.float32)
        self.costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.costs.nbytes)
        
        # init personal best (costs & position)
        self.pbest_costs = self.costs.copy()     # (nFish,)      
        self.pbest_pos = self.position.copy()    # (nDim, nFish) each particle has its persional best pos by dimension
        self.pbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.pbest_pos)  
        
        # init global best (costs & position)      
        gid = np.argmax(self.pbest_costs)         # find index for global optimal 
        self.gbest_cost = self.pbest_costs[gid]   # np.float32
        self.gbest_pos = self.pbest_pos[:, gid].copy()#.reshape(self.nDim, 1)   # (nDim, ) reshape to col vector
        self.gbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.gbest_pos)
        
        # create array best global cost storage for each iteration
        self.BestCosts = np.array([])

        # prepare kernels
        prog = cl.Program(openCLEnv.context, open("./models/kernels/pso/scalar/knl_source_pso_searchGrid.c").read()%(self.nDim)).build()
        self.knl_searchGrid = cl.Kernel(prog, 'searchGrid')
        # fitness function
        prog_AmerOpt = cl.Program(openCLEnv.context, open("./models/kernels/pso/scalar/knl_source_pso_getAmerOption.c").read()%(self.mc.nPath, self.mc.nPeriod)).build()
        self.knl_psoAmerOption_gb = cl.Kernel(prog_AmerOpt, 'psoAmerOption_gb')
    
    # use GPU to update moves
    def _searchGrid(self):
        # set kernel arguments
        self.knl_searchGrid.set_args(self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d, 
                                     self.r1_d, self.r2_d, 
                                     np.float32(self._w), np.float32(self._c1), np.float32(self._c2))
        # run kernel
        global_size = (self.nFish, )
        local_size = None
        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_searchGrid, global_size, local_size).wait()
        openCLEnv.queue.finish()         

        # cl.enqueue_copy(openCLEnv.queue, self.position, self.pos_d)
        # print(self.position)
        return 

    # Fitness function: pso American option - GPU: take the whole PSO and process once
    def _costPsoAmerOption_cl(self):
        
        self.knl_psoAmerOption_gb.set_args(self.St_d, self.pos_d, self.costs_d, 
                                           self.boundary_idx_d, self.exercise_d, 
                                           np.float32(self.mc.r), np.float32(self.mc.T), np.float32(self.mc.K), np.int8(self.mc.opt))

        # execute kernel
        global_size = (self.nFish, )
        local_size = None
        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_psoAmerOption_gb, global_size, local_size).wait()
        openCLEnv.queue.finish()    # <------- sychrnozation

        # # sanity check
        # cl.enqueue_copy(openCLEnv.queue, self.boundary_idx, self.boundary_idx_d).wait()
        # cl.enqueue_copy(openCLEnv.queue, self.exercise, self.exercise_d).wait()
        # print(f'pos hybrid: boundary & exercise: {self.boundary_idx}, {self.exercise}\n')
        return 

    def solvePsoAmerOption_cl(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move       
            t = time.perf_counter()     
            self._searchGrid()
            cl.enqueue_copy(openCLEnv.queue, self.position, self.pos_d).wait()   # read back new position
            openCLEnv.queue.finish()    # <------- sychrnozation
            search.append((time.perf_counter()- t) * 1e3)

            # 2. recalculate fitness/cost - to be implemented on GPU
            t = time.perf_counter()
            self._costPsoAmerOption_cl()
            cl.enqueue_copy(openCLEnv.queue, self.costs, self.costs_d).wait()   # read back new costs
            openCLEnv.queue.finish()    # <------- sychrnozation
            fit.append((time.perf_counter()- t) * 1e3)

            # 3. update pbest
            t = time.perf_counter()
            mask = np.greater(self.costs, self.pbest_costs)    # numpy vectorized comparison
            self.pbest_costs[mask] = self.costs[mask]
            self.pbest_pos[:,mask] = self.position[:,mask]
            cl.enqueue_copy(openCLEnv.queue, self.pbest_pos_d, self.pbest_pos).wait()   # write to device new pbest_pos
            openCLEnv.queue.finish()    # <------- sychrnozation
            
            # 4. update gbest        
            gid = np.argmax(self.pbest_costs)            
            if self.pbest_costs[gid] > self.gbest_cost:  # compare with global best
                self.gbest_cost = self.pbest_costs[gid]
                self.gbest_pos = self.pbest_pos[:,gid].copy() #.reshape(self.nDim, 1)   # (nDim, 1) reshape to col vector
                cl.enqueue_copy(openCLEnv.queue, self.gbest_pos_d, self.gbest_pos).wait()   # write to device new gbest_pos
                openCLEnv.queue.finish()    # <------- sychrnozation
    
            # 5. record global best cost for current iteration
            self.BestCosts = np.concatenate( (self.BestCosts, [self.gbest_cost]) )
            rest.append((time.perf_counter()- t) * 1e3)
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.perf_counter()- start) * 1e3
        print(f'Pso cl_hybrid price: {C_hat} - {elapse} ms')

        self.cleanUp()
        
        return C_hat, elapse, search, fit, rest
    
    def cleanUp(self):
        self.St_d.release()
        self.boundary_idx_d.release()
        self.exercise_d.release()
        self.pos_d.release()
        self.vel_d.release()
        self.r1_d.release()
        self.r2_d.release()
        self.costs_d.release()
        self.pbest_pos_d.release()
        self.gbest_pos_d.release()
        return

#################################

class PSO_OpenCL_scalar(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, direction='backward', iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        # self.fitFunc = fitFunc

        # init swarm particles positions & velocity    (nDim, nFish)
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.position)
        self.vel_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.velocity)

        # init r1, r2 on device
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2
        self.r1_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r1)
        self.r2_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r2)

        # init fitness buffer
        self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mc.St)

        # init particles costs          (nFish,)
        self.costs = np.zeros((self.nFish,), dtype=np.float32)
        self.costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.costs.nbytes)
        
        # init personal best (costs & position)
        self.pbest_costs = self.costs.copy()     # (nFish,)      
        self.pbest_costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.pbest_costs.nbytes)
        self.pbest_pos = self.position.copy()    # (nDim, nFish) each particle has its persional best pos by dimension
        self.pbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.pbest_pos)  
        
        # init global best (costs & position)      
        gid = np.argmax(self.pbest_costs)         # find index for global optimal 
        self.gbest_cost = self.pbest_costs[gid]   # np.float32 >>>> change into 1 element array
        self.gbest_cost_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.gbest_cost.nbytes) 
        self.gbest_pos = self.pbest_pos[:, gid].copy()#.reshape(self.nDim, 1)   # (nDim, ) reshape to col vector
        self.gbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.gbest_pos)
        
        # create array best global cost storage for each iteration
        self.BestCosts = np.array([])

        # prepare kernels
        # searchGrid
        prog_sg = cl.Program(openCLEnv.context, open("./models/kernels/pso/scalar/knl_source_pso_searchGrid.c").read()%(self.nDim)).build()
        self.knl_searchGrid = cl.Kernel(prog_sg, 'searchGrid')
        # fitness function
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros",]
        prog_AmerOpt = cl.Program(openCLEnv.context, open("./models/kernels/pso/scalar/knl_source_pso_getAmerOption.c").read()%(self.mc.nPath, self.mc.nPeriod)).build(options=build_options)
        # prog_AmerOpt = cl.Program(openCLEnv.context, open("./models/kernels/knl_source_pso_getAmerOption.c").read()%(self.mc.nPath, self.mc.nPeriod)).build()
        if direction=='forward':
            self.knl_psoAmerOption_gb = cl.Kernel(prog_AmerOpt, 'psoAmerOption_gb2')
        elif direction=='backward':
            self.knl_psoAmerOption_gb = cl.Kernel(prog_AmerOpt, 'psoAmerOption_gb3')
        # update bests
        prog_ub = cl.Program(openCLEnv.context, open("./models/kernels/pso/scalar/knl_source_pso_updateBests.c").read()%(self.nDim, self.nFish)).build()
        self.knl_update_pbest = cl.Kernel(prog_ub, 'update_pbest')
        self.knl_update_gbest_pos = cl.Kernel(prog_ub, 'update_gbest_pos')
    
    # use GPU to update moves
    def _searchGrid(self):
        # set kernel arguments
        self.knl_searchGrid.set_args(self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d, 
                                     self.r1_d, self.r2_d, 
                                     np.float32(self._w), np.float32(self._c1), np.float32(self._c2))
        # run kernel
        global_size = (self.nFish, )
        local_size = None
        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_searchGrid, global_size, local_size).wait()
        openCLEnv.queue.finish()         
        return 
    
    # Fitness function: pso American option - GPU: take the whole PSO and process once
    def _costPsoAmerOption_cl(self):      
        self.knl_psoAmerOption_gb.set_args(self.St_d, self.pos_d, self.costs_d, 
                                           np.float32(self.mc.r), np.float32(self.mc.T), np.float32(self.mc.K), np.int8(self.mc.opt))

        # execute kernel
        global_size = (self.nFish, )
        local_size = None
        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_psoAmerOption_gb, global_size, local_size).wait()
        openCLEnv.queue.finish()    # <------- sychrnozation

        return 

    def solvePsoAmerOption_cl(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move            
            t = time.perf_counter()
            self._searchGrid()
            search.append((time.perf_counter()- t) * 1e3)

            # 2. recalculate fitness/cost - to be implemented on GPU
            t = time.perf_counter()
            self._costPsoAmerOption_cl()
            fit.append((time.perf_counter()- t) * 1e3)

            # 3. update pbest
            t = time.perf_counter()
            self.knl_update_pbest.set_args(self.costs_d, self.pbest_costs_d, 
                                      self.pos_d, self.pbest_pos_d, 
                                    #   self.gbest_cost_d, self.gbest_pos_d
                                      )
            # run kernel
            global_size = (self.nFish, )
            local_size = None
            evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_update_pbest, global_size, local_size)
            cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d, wait_for=[evt]).wait()   # write to host new pbest_costs
            openCLEnv.queue.finish()    # <------- sychrnozation
            
            # 4. update gbest        
            gid = np.argmax(self.pbest_costs)            
            if self.pbest_costs[gid] > self.gbest_cost:  # compare with global best
                self.gbest_cost = self.pbest_costs[gid]
                self.knl_update_gbest_pos.set_args(self.gbest_pos_d, self.pbest_pos_d, 
                                      np.int32(gid))
                # run kernel
                global_size = (self.nDim, )
                local_size = None
                cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_update_gbest_pos, global_size, local_size)
                openCLEnv.queue.finish()    # <------- sychrnozation
            
    
            # 5. record global best cost for current iteration
            self.BestCosts = np.concatenate( (self.BestCosts, [self.gbest_cost]) )
            rest.append((time.perf_counter()- t) * 1e3)
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.perf_counter()- start) * 1e3
        print(f'Pso cl_scalar price: {C_hat} - {elapse} ms')

        self.cleanUp()
        return C_hat, elapse, search, fit, rest
    
    def cleanUp(self):
        self.St_d.release()
        self.pos_d.release()
        self.vel_d.release()
        self.r1_d.release()
        self.r2_d.release()
        self.costs_d.release()
        self.pbest_costs_d.release()
        self.pbest_pos_d.release()
        self.gbest_cost_d.release()
        self.gbest_pos_d.release()
        return
#################################

class PSO_OpenCL_scalar_fusion(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        # self.fitFunc = fitFunc

        # init swarm particles positions & velocity    (nDim, nFish)
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.position)
        self.vel_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.velocity)

        # init r1, r2 on device
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2
        self.r1_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r1)
        self.r2_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r2)

        # init fitness buffer
        self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mc.St)

        ## compute on the fly on GPU - init particles costs          (nFish,)
        # self.costs = np.zeros((self.nFish,), dtype=np.float32)
        # self.costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.costs.nbytes)
        
        # init personal best (costs & position)
        # self.pbest_costs = self.costs.copy()     # (nFish,)      
        self.pbest_costs = np.zeros((self.nFish,), dtype=np.float32)
        self.pbest_costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.pbest_costs.nbytes)
        self.pbest_pos = self.position.copy()    # (nDim, nFish) each particle has its persional best pos by dimension
        self.pbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.pbest_pos)  
        
        # init global best (costs & position)      
        gid = np.argmax(self.pbest_costs)         # find index for global optimal 
        self.gbest_cost = self.pbest_costs[gid]   # np.float32 >>>> change into 1 element array
        self.gbest_cost_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.gbest_cost.nbytes) 
        self.gbest_pos = self.pbest_pos[:, gid].copy()#.reshape(self.nDim, 1)   # (nDim, ) reshape to col vector
        self.gbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.gbest_pos)
        
        # create array best global cost storage for each iteration
        self.BestCosts = np.array([])

        # prepare kernels
        # searchGrid, fitness function, update pbest
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros",]
        prog = cl.Program(openCLEnv.context, open("./models/kernels/pso/scalar/knl_source_pso_fusion.c").read()%(self.nDim, self.mc.nPath, self.mc.nPeriod, self.nFish)).build(options=build_options)
        self.knl_pso = cl.Kernel(prog, 'pso')
        # update bests
        self.knl_update_gbest_pos = cl.Kernel(prog, 'update_gbest_pos')
    
    def _runPso(self):
        # set kernel arguments
        self.knl_pso.set_args(
            # searchGrid
            self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d, 
            self.r1_d, self.r2_d, 
            np.float32(self._w), np.float32(self._c1), np.float32(self._c2),
            # fitness function
            self.St_d, #self.costs_d, 
            np.float32(self.mc.r), np.float32(self.mc.T), np.float32(self.mc.K), np.int8(self.mc.opt),
            # update pbest
            self.pbest_costs_d
        )

        # run kernel
        global_size = (self.nFish, )
        local_size = None

        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_pso, global_size, local_size).wait()
        # # sanity check
        # cl.enqueue_copy(openCLEnv.queue, self.position, self.pos_d).wait()   # write to host new position
        # cl.enqueue_copy(openCLEnv.queue, self.costs, self.costs_d).wait()   # write to host new costs
        # cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d).wait()   # write to host new pbest_costs
        # print(f'current pos:\n {self.position}')
        # print(f'current costs:\n {self.costs}')
        # print(f'current pbest_costs:\n {self.pbest_costs}')

        openCLEnv.queue.finish()         

        return

    def solvePsoAmerOption_cl(self):
        search_fit, rest = [], []
        start = time.perf_counter()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move            
            # 2. recalculate fitness/cost - to be implemented on GPU
            # 3. update pbest
            t = time.perf_counter()
            self._runPso()
            cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d).wait()   # write to host new pbest_costs
            openCLEnv.queue.finish()    # <------- sychrnozation
            search_fit.append((time.perf_counter()- t) * 1e3)
            
            # 4. update gbest        
            t = time.perf_counter()
            gid = np.argmax(self.pbest_costs)            
            if self.pbest_costs[gid] > self.gbest_cost:  # compare with global best
                self.gbest_cost = self.pbest_costs[gid]
                self.knl_update_gbest_pos.set_args(self.gbest_pos_d, self.pbest_pos_d, 
                                      np.int32(gid))
                # run kernel
                global_size = (self.nDim, )
                local_size = None
                cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_update_gbest_pos, global_size, local_size)
                openCLEnv.queue.finish()    # <------- sychrnozation
                
                # self.gbest_pos = self.pbest_pos[:,gid].copy() #.reshape(self.nDim, 1)   # (nDim, 1) reshape to col vector
                # cl.enqueue_copy(openCLEnv.queue, self.gbest_pos_d, self.gbest_pos).wait()   # write to device new gbest_pos
                # openCLEnv.queue.finish()    # <------- sychrnozation
            
    
            # 5. record global best cost for current iteration
            self.BestCosts = np.concatenate( (self.BestCosts, [self.gbest_cost]) )
            rest.append((time.perf_counter()- t) * 1e3)
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.perf_counter() - start) * 1e3
        print(f'Pso cl_scalar_fusion price: {C_hat} - {elapse} ms')

        self.cleanUp()
        return C_hat, elapse, search_fit, rest
    
    def cleanUp(self):
        self.St_d.release()
        self.pos_d.release()
        self.vel_d.release()
        self.r1_d.release()
        self.r2_d.release()
        # self.costs_d.release()
        self.pbest_costs_d.release()
        self.pbest_pos_d.release()
        self.gbest_cost_d.release()
        self.gbest_pos_d.release()
        return



#################################

class PSO_OpenCL_vec(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, vec_size=4, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        # self.fitFunc = fitFunc

        # set SIMD vectorization parameters
        self.vec_size = vec_size
        try:
            # assert (self.nDim % self.vec_size) == 0
            assert (self.mc.nPath % self.vec_size) == 0
        except Exception as e:
            print(f"nDim {self.nDim} or nPath {self.mc.nPath} not divisable by vec_size {self.vec_size}")

        self.nVec_nPath = self.mc.nPath // self.vec_size   # for boundary_idx, exercise

        # init swarm particles positions & velocity    (nDim, nFish)
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.position)
        self.vel_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.velocity)

        # init r1, r2 on device
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2
        self.r1_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r1)
        self.r2_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r2)

        # init fitness buffer
        # mc.St is in shape [nPath, nPeriod], to vectorize, need to transpose to [nPeriod, nPath]
        # St_vec in shapge [nPeriod, vec_size, nVec_nPath], 将 nPath 折叠，一个 period 的迭代，可以同时处理 vec_size个 path
        self.St_vec = self.mc.St.copy().T.reshape(self.mc.nPeriod, self.nVec_nPath, self.vec_size).transpose(0, 1, 2).copy().reshape(-1, self.vec_size)
        self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.St_vec.ravel())

        # init particles costs          (nFish,)
        self.costs = np.zeros((self.nFish,), dtype=np.float32)
        self.costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.costs.nbytes)
        
        # init personal best (costs & position)
        self.pbest_costs = self.costs.copy()     # (nFish,)      
        self.pbest_costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.pbest_costs.nbytes)
        self.pbest_pos = self.position.copy()    # (nDim, nFish) each particle has its persional best pos by dimension
        self.pbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.pbest_pos)  
        
        # init global best (costs & position)      
        gid = np.argmax(self.pbest_costs)         # find index for global optimal 
        self.gbest_cost = self.pbest_costs[gid]   # np.float32 
        self.gbest_cost_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.gbest_cost.nbytes) 
        self.gbest_pos = self.pbest_pos[:, gid].copy()#.reshape(self.nDim, 1)   # (nDim, ) reshape to col vector
        self.gbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.gbest_pos)
        
        # create array best global cost storage for each iteration
        self.BestCosts = np.array([])

        # prepare kernels
        # searchGrid
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros",]
        prog_sg = cl.Program(openCLEnv.context, open("./models/kernels/pso/vec/knl_source_pso_searchGrid_vec.c").read()%(self.nDim)).build(options=build_options)
        # self.knl_searchGrid = cl.Kernel(prog_sg, 'searchGrid')
        self.knl_searchGrid = cl.Kernel(prog_sg, 'searchGrid_f2f4')
        # fitness function
        # build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros", f"-DVEC_SIZE={self.vec_size}"]
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros", 
                         f"-DVEC_SIZE={self.vec_size}",
                         f"-Dn_PATH={self.mc.nPath}",
                         f"-Dn_PERIOD={self.mc.nPeriod}",
                         ]
        prog_AmerOpt = cl.Program(openCLEnv.context, open("./models/kernels/pso/vec/knl_source_pso_getAmerOption_vec.c").read() ).build(options=build_options)
        # prog_AmerOpt = cl.Program(openCLEnv.context, open("./models/kernels/knl_source_pso_getAmerOption.c").read()%(self.mc.nPath, self.mc.nPeriod)).build(options=build_options)
        self.knl_psoAmerOption_gb = cl.Kernel(prog_AmerOpt, 'psoAmerOption_gb3_vec')
        # update bests
        prog_ub = cl.Program(openCLEnv.context, open("./models/kernels/pso/vec/knl_source_pso_updateBests_vec.c").read()%(self.nDim, self.nFish)).build()
        # self.knl_update_pbest = cl.Kernel(prog_ub, 'update_pbest')
        self.knl_update_pbest = cl.Kernel(prog_ub, 'update_pbest_f2f4')
        # self.knl_update_gbest_pos = cl.Kernel(prog_ub, 'update_gbest_pos')
        self.knl_update_gbest_pos = cl.Kernel(prog_ub, 'update_gbest_pos_f2f4')
    
    # use GPU to update moves
    def _searchGrid(self):
        # set kernel arguments
        self.knl_searchGrid.set_args(self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d, 
                                     self.r1_d, self.r2_d, 
                                     np.float32(self._w), np.float32(self._c1), np.float32(self._c2))
        # run kernel
        global_size = (self.nFish, )
        local_size = None
        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_searchGrid, global_size, local_size).wait()
        openCLEnv.queue.finish()         

        # cl.enqueue_copy(openCLEnv.queue, self.pos_vec, self.pos_d)
        # print(self.pos_vec.reshape(self.nVec_nDim, self.nFish, self.vec_size).transpose(0, 2, 1).reshape(self.nDim, self.nFish))
        return 
    
    # Fitness function: pso American option - GPU: take the whole PSO and process once
    def _costPsoAmerOption_cl(self):
        self.knl_psoAmerOption_gb.set_args(self.St_d, self.pos_d, self.costs_d, 
                                           np.float32(self.mc.r), np.float32(self.mc.T), np.float32(self.mc.K), np.int8(self.mc.opt))

        # execute kernel
        global_size = (self.nFish, )
        local_size = None
        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_psoAmerOption_gb, global_size, local_size).wait()
        openCLEnv.queue.finish()    # <------- sychrnozation

        return 

    def solvePsoAmerOption_cl(self):
        search, fit, rest = [], [], []
        start = time.perf_counter()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move     
            t = time.perf_counter()       
            self._searchGrid()
            search.append((time.perf_counter()- t) * 1e3)

            # 2. recalculate fitness/cost - to be implemented on GPU
            t = time.perf_counter()
            # self.fitFunc(self.pos_d, self.costs_d)
            self._costPsoAmerOption_cl()
            fit.append((time.perf_counter()- t) * 1e3)

            # 3. update pbest
            t = time.perf_counter()
            self.knl_update_pbest.set_args(self.costs_d, self.pbest_costs_d, 
                                      self.pos_d, self.pbest_pos_d, 
                                    #   self.gbest_cost_d, self.gbest_pos_d
                                      )
            # run kernel
            global_size = (self.nFish, )
            local_size = None
            evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_update_pbest, global_size, local_size)
            cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d, wait_for=[evt]).wait()   # write to host new pbest_costs
            openCLEnv.queue.finish()    # <------- sychrnozation
            
            # 4. update gbest        
            gid = np.argmax(self.pbest_costs)            
            if self.pbest_costs[gid] > self.gbest_cost:  # compare with global best
                self.gbest_cost = self.pbest_costs[gid]
                self.knl_update_gbest_pos.set_args(self.gbest_pos_d, self.pbest_pos_d, 
                                      np.int32(gid))
                # run kernel
                global_size = (self.nDim, )
                local_size = None
                cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_update_gbest_pos, global_size, local_size)
                openCLEnv.queue.finish()    # <------- sychrnozation
            
    
            # 5. record global best cost for current iteration
            self.BestCosts = np.concatenate( (self.BestCosts, [self.gbest_cost]) )
            rest.append((time.perf_counter()- t) * 1e3)
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.perf_counter()- start) * 1e3
        print(f'Pso cl_vec(float{self.vec_size}) price: {C_hat} - {elapse} ms')

        self.cleanUp()
        return C_hat, elapse, search, fit, rest
    
    def cleanUp(self):
        self.St_d.release()
        self.pos_d.release()
        self.vel_d.release()
        self.r1_d.release()
        self.r2_d.release()
        self.costs_d.release()
        self.pbest_costs_d.release()
        self.pbest_pos_d.release()
        self.gbest_cost_d.release()
        self.gbest_pos_d.release()
        return
#################################

class PSO_OpenCL_vec_fusion(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, vec_size=4, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        # self.fitFunc = fitFunc

        # set SIMD vectorization parameters
        self.vec_size = vec_size
        try:
            # assert (self.nDim % self.vec_size) == 0
            assert (self.mc.nPath % self.vec_size) == 0
        except Exception as e:
            print(f"nDim {self.nDim} or nPath {self.mc.nPath} not divisable by vec_size {self.vec_size}")

        self.nVec_nPath = self.mc.nPath // self.vec_size   # for boundary_idx, exercise

        # init swarm particles positions & velocity    (nDim, nFish)
        self.position = self.mc.pos_init.copy()
        self.velocity = self.mc.vel_init.copy()
        self.pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.position)
        self.vel_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.velocity)

        # init r1, r2 on device
        self.r1 = self.mc.r1
        self.r2 = self.mc.r2
        self.r1_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r1)
        self.r2_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.r2)

        # init fitness buffer
        # mc.St is in shape [nPath, nPeriod], to vectorize, need to transpose to [nPeriod, nPath]
        # St_vec in shapge [nPeriod, vec_size, nVec_nPath], 将 nPath 折叠，一个 period 的迭代，可以同时处理 vec_size个 path
        self.St_vec = self.mc.St.copy().T.reshape(self.mc.nPeriod, self.nVec_nPath, self.vec_size).transpose(0, 1, 2).copy().reshape(-1, self.vec_size)
        self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=self.St_vec.ravel())

        ## compute on the fly on GPU - init particles costs          (nFish,)
        # self.costs = np.zeros((self.nFish,), dtype=np.float32)
        # self.costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.costs.nbytes)
        
        # init personal best (costs & position)
        # self.pbest_costs = self.costs.copy()     # (nFish,)      
        self.pbest_costs = np.zeros((self.nFish,), dtype=np.float32)   
        self.pbest_costs_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.pbest_costs.nbytes)
        self.pbest_pos = self.position.copy()    # (nDim, nFish) each particle has its persional best pos by dimension
        self.pbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.pbest_pos)  
        
        # init global best (costs & position)      
        gid = np.argmax(self.pbest_costs)         # find index for global optimal 
        self.gbest_cost = self.pbest_costs[gid]   # np.float32 >>>> change into 1 element array
        self.gbest_cost_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=self.gbest_cost.nbytes) 
        self.gbest_pos = self.pbest_pos[:, gid].copy()#.reshape(self.nDim, 1)   # (nDim, ) reshape to col vector
        self.gbest_pos_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.gbest_pos)
        
        # create array best global cost storage for each iteration
        self.BestCosts = np.array([])

        # prepare kernels
        # searchGrid, fitness function, update pbest
        build_options = ["-cl-fast-relaxed-math", "-cl-mad-enable", "-cl-no-signed-zeros", 
                         f"-DVEC_SIZE={self.vec_size}",
                         f"-Dn_Dim={self.nDim}",
                         f"-Dn_PATH={self.mc.nPath}",
                         f"-Dn_PERIOD={self.mc.nPeriod}",
                         f"-Dn_Fish={self.nFish}",
                         ]
        prog = cl.Program(openCLEnv.context, open("./models/kernels/pso/vec/knl_source_pso_fusion_vec.c").read() ).build(options=build_options)
        # prog = cl.Program(openCLEnv.context, open("./models/kernels/knl_source_pso_oneKernel_vec.c").read()%(self.nDim, self.mc.nPath, self.mc.nPeriod, self.nFish)).build()
        self.knl_pso = cl.Kernel(prog, 'pso_vec')
        # update bests
        self.knl_update_gbest_pos = cl.Kernel(prog, 'update_gbest_pos_vec')
    
    def _runPso(self):
        # set kernel arguments
        self.knl_pso.set_args(
            # searchGrid
            self.pos_d, self.vel_d, self.pbest_pos_d, self.gbest_pos_d, 
            self.r1_d, self.r2_d, 
            np.float32(self._w), np.float32(self._c1), np.float32(self._c2),
            # fitness function
            self.St_d, #self.costs_d, 
            np.float32(self.mc.r), np.float32(self.mc.T), np.float32(self.mc.K), np.int8(self.mc.opt),
            # update pbest
            self.pbest_costs_d
        )

        # run kernel
        global_size = (self.nFish, )
        local_size = None

        cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_pso, global_size, local_size).wait()
        # # sanity check
        # cl.enqueue_copy(openCLEnv.queue, self.position, self.pos_d).wait()   # write to host new position
        # cl.enqueue_copy(openCLEnv.queue, self.costs, self.costs_d).wait()   # write to host new costs
        # cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d).wait()   # write to host new pbest_costs
        # print(f'current pos:\n {self.position}')
        # print(f'current costs:\n {self.costs}')
        # print(f'current pbest_costs:\n {self.pbest_costs}')

        openCLEnv.queue.finish()         

        return

    def solvePsoAmerOption_cl(self):
        search_fit, rest = [], []
        start = time.perf_counter()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move            
            # 2. recalculate fitness/cost - to be implemented on GPU
            # 3. update pbest
            t = time.perf_counter()
            self._runPso()
            cl.enqueue_copy(openCLEnv.queue, self.pbest_costs, self.pbest_costs_d).wait()   # write to host new pbest_costs
            openCLEnv.queue.finish()    # <------- sychrnozation
            search_fit.append((time.perf_counter()- t) * 1e3)
            
            # 4. update gbest        
            t = time.perf_counter()
            gid = np.argmax(self.pbest_costs)            
            if self.pbest_costs[gid] > self.gbest_cost:  # compare with global best
                self.gbest_cost = self.pbest_costs[gid]
                self.knl_update_gbest_pos.set_args(self.gbest_pos_d, self.pbest_pos_d, 
                                      np.int32(gid))
                # run kernel
                global_size = (self.nDim, )
                local_size = None
                cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_update_gbest_pos, global_size, local_size)
                openCLEnv.queue.finish()    # <------- sychrnozation
                
                # self.gbest_pos = self.pbest_pos[:,gid].copy() #.reshape(self.nDim, 1)   # (nDim, 1) reshape to col vector
                # cl.enqueue_copy(openCLEnv.queue, self.gbest_pos_d, self.gbest_pos).wait()   # write to device new gbest_pos
                # openCLEnv.queue.finish()    # <------- sychrnozation
            
    
            # 5. record global best cost for current iteration
            self.BestCosts = np.concatenate( (self.BestCosts, [self.gbest_cost]) )
            rest.append((time.perf_counter()- t) * 1e3)
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.perf_counter() - start) * 1e3
        print(f'Pso cl_vec(float{self.vec_size})_fusion price: {C_hat} - {elapse} ms')

        self.cleanUp()
        return C_hat, elapse, search_fit, rest
    
    def cleanUp(self):
        self.St_d.release()
        self.pos_d.release()
        self.vel_d.release()
        self.r1_d.release()
        self.r2_d.release()
        # self.costs_d.release()
        self.pbest_costs_d.release()
        self.pbest_pos_d.release()
        self.gbest_cost_d.release()
        self.gbest_pos_d.release()
        return



def main():
    print("pso.py")

if __name__ == "__main__":
    main()
