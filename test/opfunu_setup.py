#%%
from DEEM.DEEM import DEEM
import numpy as np
import opfunu

def test_func_DEEM(func, name, NDim, NRUNS)-> list:
    """
    func: function that is to be evaluated
    name: name of the function based on the opfunu module
    NDim: dimension of the functionspace 
    NRUNS: number of runs
    """
    
    nparticles_max = 10*NDim
    nparticles_min = 10*NDim
    nswarm_max = 10
    nswarm_min = 4
    maxiter = 1000
    maxfev = NDim*10000
    nworker = 1

    lb = np.array([-100]*NDim) ; ub = np.array([100]*NDim)

    method_subswarm_reduction = 'linear'
    method_subswarm_creation = 'fitness-focused'
    method_boundary = 'damping-periodic'
    sampling = 'Random-Uniform'

    f_i = func
    f_values = []

    for i in range(0,NRUNS):
        optimizer = DEEM(
            function = f_i.evaluate, 
            nparticles_max=nparticles_max, 
            nparticles_min=nparticles_min, 
            nswarm_max=nswarm_max,
            nswarm_min=nswarm_min,
            lower_bound=lb, 
            upper_bound=ub, 
            maxiter=maxiter, 
            maxfev=maxfev,
            sampling_method = sampling,
            nworkers=nworker, 
            tolerance=1e-4, 
            maxiter_below_tolerance=maxiter,
            method_subswarm_reduction=method_subswarm_reduction,
            method_subswarm_creation=method_subswarm_creation,
            method_boundary=method_boundary)
        optimizer.update()
        f_values.append(optimizer.FBEST)

    return f_values


#%%

# Run ONE functions of the selected CEC event

if __name__ == "__main__":

    all_funcs = opfunu.get_functions_based_classname("2022")
    func = all_funcs[0]
    NDim = 10
    NRUNS = 1
    
    name = func.name.split(":")[0]
    
    f_values = test_func_DEEM(all_funcs[4](ndim = NDim), name, NDim, NRUNS)

    with open(f'DEEM_CEC2022_{name}_Ndim={NDim}_Nruns={NRUNS}.txt', 'w') as f:
            f.write(", ".join(str(item) for item in f_values))
            f.write("\n")

#%%

# Run ALL functions of the selected CEC event

if __name__ == "__main__":

    all_funcs = opfunu.get_functions_based_classname("2022")
    NDim = 10
    NRUNS = 1

    for func in all_funcs:
        name = func.name.split(":")[0]
        f_values = test_func_DEEM(func(ndim = NDim), name, NDim, NRUNS)

        with open(f'DEEM_CEC2022_{name}_Ndim={NDim}_Nruns={NRUNS}.txt', 'w') as f:
            f.write(", ".join(str(item) for item in f_values))
            f.write("\n")
