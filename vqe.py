import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import SPSA

def get_vqe_results_v2(
        state,              
        hamiltonian,        
        optimizer=SPSA(maxiter=100), 
        estimator=EstimatorV2(),
        filename='default_filename'
    ):

    """
    Returns the lists of iteration numbers and corresponding energy values obtained during the VQE optimization process
    using EstimatorV2. The results are also saved in a .out file in the out/ folder.
        Args:
            - state: qiskit.circuit.QuantumCircuit, the ansatz circuit for the VQE simulation
            - hamiltonian: qiskit.SparsePauliOp, the Hamiltonian of the system for which we want to find the ground state
            energy
            - optimizer: qiskit_algorithms.optimizers.optimizer instance, the optimizer to be used in the VQE simulation
            (default is SPSA with maxiter=100)
            - estimator: qiskit_aer.primitives.EstimatorV2 instance, the estimator to be used in the VQE simulation
            - filename: str, name of the .out file to be saved in the out/ folder (without extension)
        Returns:
            - iters: list, of N_iters numbers, contains the iteration numbers during the optimization process
            - energies: list, of N_iters numbers, contains the energy values corresponding to each iteration during the
            optimization process
    """
    
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
    
    return iters, energies