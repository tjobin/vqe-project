from qiskit_algorithms.optimizers.spsa import SPSA
import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import SPSA

def get_vqe_results_v2(
        state,              
        hamiltonian,        
        optimizer=SPSA(maxiter=100), 
        estimator=EstimatorV2(),
        filename='vqe_v2_log'
    ):
    
    coupling_map = estimator._backend.coupling_map
    target_basis = estimator._backend._basis_gates()
    isa_ansatz = transpile(state, coupling_map=coupling_map, basis_gates=target_basis, optimization_level=3, seed_transpiler=0)
    isa_hamiltonian = hamiltonian

    iters = []
    energies = []
    
    fout = open(f'out/{filename}.out', 'w')
    fout.write(f"VQE Simulation (EstimatorV2)\n")
    fout.write(f"{'Iter':>10} {'Energy (Hartree)':>20}\n")
    
    def cost_func(params):
        pub = (isa_ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        result = job.result()[0] 
        current_energy = result.data.evs
        
        iter_count = len(energies)
        iters.append(iter_count)
        energies.append(current_energy)
        
        fout.write(f"{iter_count:>10} {current_energy:>20.6f}\n")
        return current_energy

    # --- Run Optimizer ---
    num_params = isa_ansatz.num_parameters
    x0 = np.random.uniform(-np.pi, np.pi, num_params)
    result = optimizer.minimize(fun=cost_func, x0=x0) 
    
    pub_final = (isa_ansatz, isa_hamiltonian, result.x)
    job_final = estimator.run([pub_final])
    _ = job_final.result()[0]

    fout.close()
    
    # Return variance as the 4th value
    return iters, energies