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
    def __init__(self, mc: MonteCarloBase, nFish, fitFunc, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        self.fitFunc_vectorized = np.vectorize(fitFunc, signature='(n)->()')

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

    def searchGrid(self):
        # update velocity  
        self.velocity = self._w * self.velocity + self._c1*self.r1*(self.pbest_pos - self.position) + \
                    self._c2*self.r2*(self.gbest_pos.reshape(self.nDim, 1) - self.position)
        # out_v = out_v.clip(vMin, vMax)       # bound velocity
        # update position
        self.position += self.velocity 
        # out_p = out_p.clip(pMin, pMax)       # bound position
        return 

    def solvePsoAmerOption_np(self):
        start = time.time()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move                          
            self.searchGrid()

            # 2. recalculate fitness/cost 
            self.costs = self.fitFunc_vectorized( np.transpose(self.position) ).astype(np.float32)

            # 3. update pbest
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
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.time() - start) * 1e3
        print(f'Pso numpy price: {C_hat} - {elapse} ms')
        return C_hat


class PSO_OpenCL(PSOBase):
    def __init__(self, mc: MonteCarloBase, nFish, fitFunc, iterMax=30):
        super().__init__(mc, nFish)
        self.iterMax = iterMax
        self.fitFunc = fitFunc

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
        prog = cl.Program(openCLEnv.context, open("./models/kernels/knl_source_pso_searchGrid.c").read()%(self.nDim)).build()
        self.knl_searchGrid = cl.Kernel(prog, 'searchGrid')

     # use GPU to update moves
    def searchGrid(self):
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

    def solvePsoAmerOption_cl(self):
        start = time.time()
        # print('  Solver job started')
        for i in range(self.iterMax):         # loop of iterations
        # while True:
            # 1. particle move            
            self.searchGrid()
            cl.enqueue_copy(openCLEnv.queue, self.position, self.pos_d).wait()   # read back new position
            openCLEnv.queue.finish()    # <------- sychrnozation

            # 2. recalculate fitness/cost - to be implemented on GPU
            self.fitFunc(self.pos_d, self.costs_d)
            cl.enqueue_copy(openCLEnv.queue, self.costs, self.costs_d).wait()   # read back new costs
            openCLEnv.queue.finish()    # <------- sychrnozation

            # 3. update pbest
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
    
            # 6. The computation stops when the improvement of the value is less than criteria
            if len(self.BestCosts) > 2 and abs(self.BestCosts[-1] - self.BestCosts[-2]) < self._criteria:
                break
        
        C_hat = self.gbest_cost

        elapse = (time.time() - start) * 1e3
        print(f'Pso {openCLEnv.deviceName} price: {C_hat} - {elapse} ms')
        return C_hat 
    
    def cleanUp(self):
        self.pos_d.release()
        self.vel_d.release()
        self.r1_d.release()
        self.r2_d.release()
        self.costs_d.release()
        self.pbest_pos_d.release()
        self.gbest_pos_d.release()
        return


def main():
    print("pso.py")

if __name__ == "__main__":
    main()
