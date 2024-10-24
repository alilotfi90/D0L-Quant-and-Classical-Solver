# D0L-Quant-and-Classical-Solver

This project implements both quantum and classical approaches to solve the D0L-system inference problem using characteristic graphs.
The quantum implementation utilizes QAOA (Quantum Approximate Optimization Algorithm) while the classical approach uses a Maximum Independent Set (MIS) solver.

Overview
D0L-systems are a type of L-system where the production rules are context-free and deterministic. The inference problem involves finding the production rules given a sequence of strings. 
This implementation:

Constructs characteristic graphs from input sequences
Solves the inference problem using:

1- Quantum approach (QAOA)
2- Classical approach (MIS-based solver)

Requirements: 
a) Qiskit
b) Numpy
c) Matplotlib
d) Networkx

bash: pip install qiskit qiskit-ibm-runtime qiskit-algorithms qiskit-optimization numpy matplotlib networkx

Authors: Ali Lotfi, Ian McQuillan, Steven Rayan
