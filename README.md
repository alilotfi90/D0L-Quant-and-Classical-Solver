# D0L-System Inference via a Characteristic Graph (classical + QAOA)

Code and benchmarks accompanying the paper on reducing deterministic L-system (D0L)
inference to Maximum Independent Set via a characteristic graph, with an exact classical
decision procedure and a small-instance QAOA proof-of-concept.

## Requirements
- Python 3.9+
- `numpy`, `scipy` (see `requirements.txt`)

No Qiskit, IBM Quantum account, or quantum hardware is required: the QAOA results are
produced by a local statevector simulation written in NumPy and run entirely on the CPU.

```bash
pip install -r requirements.txt
```

## Contents
- `merged_d0l_experiments.py` — characteristic-graph construction, the exact classical
  decision procedure, negative controls, graph statistics, and the gate-count/depth table.
- `run_qaoa_cases.py` — local statevector QAOA runner for the small diagnostic instances.
- `lsystems/` — 22 JSON D0L benchmark systems, collected by Ian McQuillan and Jason Bernard,
  used to generate the string sequences.
- `results/` — CSV/JSON/LaTeX outputs reproduced by the scripts.

## Benchmark systems
The 22 D0L-systems in `lsystems/` were collected by Ian McQuillan and Jason Bernard as part
of their work on L-system inference, and are included here with attribution. Please credit
them when reusing these benchmark systems. See:

> Jason Bernard and Ian McQuillan. *Techniques for inferring context-free Lindenmayer systems
> with genetic algorithm.* Swarm and Evolutionary Computation, 64:100893, 2021.

## Reproducing the classical experiments
```bash
python merged_d0l_experiments.py --lsystems-dir lsystems --out-dir results
```
Expected headline results: **22/22** positive traces validated and **8/8** incompatible
controls rejected, plus the gate-count-versus-depth table.

## Reproducing the QAOA diagnostics
```bash
python run_qaoa_cases.py --lsystems-dir lsystems --out-dir results
```
If the combined run is slow on a given machine, run the four cases independently:
```bash
python run_qaoa_cases.py --lsystems-dir lsystems --out-dir results --case 0 --case-out results/qaoa_case_0.json
python run_qaoa_cases.py --lsystems-dir lsystems --out-dir results --case 1 --case-out results/qaoa_case_1.json
python run_qaoa_cases.py --lsystems-dir lsystems --out-dir results --case 2 --case-out results/qaoa_case_2.json
python run_qaoa_cases.py --lsystems-dir lsystems --out-dir results --case 3 --case-out results/qaoa_case_3.json
```
The parameter search is a seeded multi-start local search and is a diagnostic protocol, not
a scalable training method. The QAOA table is a small noiseless sanity check; it is **not**
evidence of quantum advantage over exact classical MIS/SAT/CP solvers.

## Citation
Please cite the accompanying paper once published. When reusing the benchmark systems in
`lsystems/`, please also cite Bernard and McQuillan (2021) as noted under **Benchmark systems**.
