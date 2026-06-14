#!/usr/bin/env python3
"""
Merged second-revision experiments for the D0L characteristic-graph paper.

Features combined from the two draft packages:
  * fixed-identity-symbol pruning using known_symbols from the benchmark files;
  * exact classical inference/validation for 22 generated D0L traces;
  * negative controls demonstrating exact rejection of incompatible traces;
  * explicit graph statistics and edge-colouring depth estimates for small graphs;
  * QAOA statevector simulation on small pruned characteristic graphs, with no Qiskit
    or hardware credentials required.

The implementation is intentionally dependency-light: numpy/scipy/networkx only.
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set, Optional, Any
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

try:
    import networkx as nx
except Exception:
    nx = None

sys.setrecursionlimit(1_000_000)

Vertex = Tuple[int, int, int, int]  # (transition i, predecessor position j, start, end), Python slices

# ---------------------------------------------------------------------------
# Loading and generation
# ---------------------------------------------------------------------------

def load_instances(base: Path) -> List[Tuple[str, Dict[str, Any]]]:
    return [(p.name, json.load(open(p, encoding='utf-8'))) for p in sorted(base.glob('*.json'))]


def complete_rules(data: Dict[str, Any]) -> Dict[str, str]:
    rules = dict(data.get('production_rules', {}))
    for a in data.get('alphabet', []):
        rules.setdefault(a, a)
    return rules


def apply_rules(rules: Dict[str, str], s: str) -> str:
    return ''.join(rules.get(ch, ch) for ch in s)


def generate(data: Dict[str, Any], steps: int) -> List[str]:
    rules = complete_rules(data)
    seq = [data['axiom']]
    w = data['axiom']
    for _ in range(steps):
        w = apply_rules(rules, w)
        seq.append(w)
    return seq


def minimal_steps_all_rule_predecessors(data: Dict[str, Any], max_steps: int = 100) -> Tuple[int, List[str]]:
    """Smallest m such that every explicit production-rule symbol occurs in some
    predecessor string w_0,...,w_{m-1}.  This makes the trace principled: every
    rule to be inferred has at least one observed left-hand-side occurrence."""
    rule_symbols = set(data.get('production_rules', {}).keys())
    rules = complete_rules(data)
    seq = [data['axiom']]
    seen: Set[str] = set()
    for m in range(1, max_steps + 1):
        # At transition m-1, the predecessor is seq[-1].
        seen.update(ch for ch in seq[-1] if ch in rule_symbols)
        seq.append(apply_rules(rules, seq[-1]))
        if seen >= rule_symbols:
            return m, seq
    return len(seq) - 1, seq


def observed_constants(seq: List[str], data: Dict[str, Any]) -> Set[str]:
    """Known fixed-identity symbols for the experiments.

    We combine the benchmark's known_symbols field with observed symbols that
    have no explicit production rule in the benchmark file.  Such symbols are
    treated as drawing/control constants and fixed to c -> c.
    """
    observed = set(''.join(seq))
    explicit = set(data.get('production_rules', {}).keys())
    return set(data.get('known_symbols', [])) | (observed - explicit)

# ---------------------------------------------------------------------------
# Characteristic graph with optional fixed-identity pruning
# ---------------------------------------------------------------------------

def raw_ranges(prev_len: int, next_len: int, j: int, allow_erasing: bool = True) -> List[Tuple[int, int]]:
    if prev_len == 1:
        ranges = [(0, next_len)]
    elif j == 0:
        ranges = [(0, e) for e in range(0, next_len + 1)]
    elif j == prev_len - 1:
        ranges = [(s, next_len) for s in range(0, next_len + 1)]
    else:
        ranges = [(s, e) for s in range(0, next_len + 1) for e in range(s, next_len + 1)]
    if not allow_erasing:
        ranges = [(s, e) for s, e in ranges if e > s]
    return ranges


def count_possible(ch: str, nextstr: str, prev_len: int, j: int,
                   constants: Optional[Set[str]] = None,
                   allow_erasing: bool = True) -> int:
    constants = constants or set()
    L = len(nextstr)
    if ch not in constants:
        if prev_len == 1:
            return 1 if (allow_erasing or L > 0) else 0
        if j == 0 or j == prev_len - 1:
            return L + 1 if allow_erasing else L
        return (L + 1) * (L + 2) // 2 if allow_erasing else L * (L + 1) // 2
    # fixed identity c -> c: only intervals whose substring equals c remain
    if prev_len == 1:
        return 1 if nextstr == ch else 0
    if j == 0:
        return 1 if L >= 1 and nextstr[0] == ch else 0
    if j == prev_len - 1:
        return 1 if L >= 1 and nextstr[-1] == ch else 0
    # Interior fixed-identity symbols can only use length-one intervals equal to ch.
    # Counting occurrences avoids enumerating O(|w_{i+1}|^2) intervals.
    return nextstr.count(ch)


def graph_stats_formula(seq: List[str], constants: Optional[Set[str]] = None,
                        allow_erasing: bool = True) -> Dict[str, int]:
    constants = constants or set()
    n = k = max_clique = empty_domains = 0
    for i in range(len(seq) - 1):
        prev, nxt = seq[i], seq[i + 1]
        p = len(prev)
        for j, ch in enumerate(prev):
            c = count_possible(ch, nxt, p, j, constants, allow_erasing)
            n += c
            k += 1
            max_clique = max(max_clique, c)
            if c == 0:
                empty_domains += 1
    return dict(vertices=n, k=k, max_clique=max_clique, empty_domains=empty_domains)


def build_graph(seq: List[str], constants: Optional[Set[str]] = None,
                allow_erasing: bool = True, max_vertices: int = 200_000) -> Tuple[List[Vertex], List[List[int]], List[Set[int]]]:
    """Build a pruned characteristic graph explicitly for small instances."""
    constants = constants or set()
    vertices: List[Vertex] = []
    cliques: List[List[int]] = []
    for i in range(len(seq) - 1):
        prev, nxt = seq[i], seq[i + 1]
        p, L = len(prev), len(nxt)
        for j, ch in enumerate(prev):
            cv: List[int] = []
            for s, e in raw_ranges(p, L, j, allow_erasing):
                if ch in constants and nxt[s:e] != ch:
                    continue
                idx = len(vertices)
                vertices.append((i, j, s, e))
                cv.append(idx)
                if len(vertices) > max_vertices:
                    raise MemoryError(f"too many vertices ({len(vertices)}>{max_vertices})")
            cliques.append(cv)
    n = len(vertices)
    adj: List[Set[int]] = [set() for _ in range(n)]
    # intra-clique edges
    for cv in cliques:
        for a_i in range(len(cv)):
            a = cv[a_i]
            for b in cv[a_i + 1:]:
                adj[a].add(b); adj[b].add(a)
    # inter-clique C1/C2 edges
    for a in range(n):
        i1, j1, s1, e1 = vertices[a]
        ch1 = seq[i1][j1]
        sub1 = seq[i1 + 1][s1:e1]
        for b in range(a + 1, n):
            i2, j2, s2, e2 = vertices[b]
            if (i1, j1) == (i2, j2):
                continue
            ch2 = seq[i2][j2]
            sub2 = seq[i2 + 1][s2:e2]
            edge = False
            if ch1 == ch2 and sub1 != sub2:
                edge = True
            elif i1 == i2 and j2 == j1 + 1 and e1 != s2:
                edge = True
            elif i1 == i2 and j1 == j2 + 1 and e2 != s1:
                edge = True
            if edge:
                adj[a].add(b); adj[b].add(a)
    return vertices, cliques, adj


def graph_stats_explicit(vertices: List[Vertex], cliques: List[List[int]], adj: List[Set[int]]) -> Dict[str, int]:
    return dict(n=len(vertices), edges=sum(len(s) for s in adj)//2,
                max_clique=max((len(c) for c in cliques), default=0),
                max_degree=max((len(s) for s in adj), default=0),
                k=len(cliques))


def greedy_edge_coloring_depth(adj: List[Set[int]]) -> int:
    """Greedy proper edge colouring: color count is an achievable all-to-all
    two-qubit depth for one cost layer.  Processes high-conflict edges first."""
    edges = [(u, v) for u in range(len(adj)) for v in adj[u] if u < v]
    edges.sort(key=lambda e: max(len(adj[e[0]]), len(adj[e[1]])), reverse=True)
    used: List[Set[int]] = [set() for _ in range(len(adj))]
    max_color = -1
    for u, v in edges:
        c = 0
        uu, vv = used[u], used[v]
        while c in uu or c in vv:
            c += 1
        uu.add(c); vv.add(c)
        if c > max_color:
            max_color = c
    return max_color + 1

# ---------------------------------------------------------------------------
# Exact classical inference with fixed constants
# ---------------------------------------------------------------------------

class SearchLimit(Exception):
    pass


def infer_global(seq: List[str], production_keys: Iterable[str], known_constants: Set[str],
                 allow_erasing: bool = True, time_limit: float = 30.0,
                 max_nodes: int = 5_000_000, count_cap: Optional[int] = None) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
    """Exact depth-first search over the same substring choices encoded by the
    characteristic graph, using fixed-identity constants.  If count_cap is not
    None, count distinct compatible morphisms up to the cap."""
    start = time.perf_counter()
    nodes = 0
    observed = set(''.join(seq))
    prod_keys = set(production_keys)
    constants = set(known_constants) | (observed - prod_keys)
    base_prod = {c: c for c in constants}
    solutions: List[Tuple[Tuple[str, str], ...]] = []
    first_solution: Optional[Dict[str, str]] = None

    def tick():
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or (time.perf_counter() - start) > time_limit:
            raise SearchLimit

    def possible_ends(y: str, pos: int, rest: str, prod: Dict[str, str]) -> List[int]:
        L = len(y)
        min_rest = 0
        for s in rest:
            if s in prod:
                min_rest += len(prod[s])
            else:
                min_rest += 0 if allow_erasing else 1
        lo = pos + (0 if allow_erasing else 1)
        hi = L - min_rest
        if hi < lo:
            return []
        # Try ends aligned with the next assigned symbol first, but keep all ends.
        preferred: List[int] = []
        if rest:
            ns = rest[0]
            img = prod.get(ns)
            if img:
                k = lo
                while True:
                    idx = y.find(img, k)
                    if idx == -1 or idx > hi:
                        break
                    preferred.append(idx)
                    k = idx + 1
        seen: Set[int] = set()
        ordered: List[int] = []
        for e in preferred + list(range(lo, hi + 1)):
            if e not in seen:
                ordered.append(e); seen.add(e)
        return ordered

    def gen_match(w: str, y: str, prod_in: Dict[str, str]):
        Lw, Ly = len(w), len(y)
        def dfs(j: int, pos: int, prod: Dict[str, str]):
            tick()
            if j == Lw:
                if pos == Ly:
                    yield prod
                return
            if pos > Ly:
                return
            s = w[j]
            if s in prod:
                img = prod[s]
                if y.startswith(img, pos):
                    yield from dfs(j + 1, pos + len(img), prod)
                return
            rest = w[j+1:]
            for e in possible_ends(y, pos, rest, prod):
                img = y[pos:e]
                if (not allow_erasing) and img == '':
                    continue
                prod2 = dict(prod)
                prod2[s] = img
                yield from dfs(j + 1, e, prod2)
        yield from dfs(0, 0, dict(prod_in))

    def solve_transition(i: int, prod: Dict[str, str]):
        nonlocal first_solution
        tick()
        if i == len(seq) - 1:
            completed = dict(prod)
            for a in observed | prod_keys:
                completed.setdefault(a, a)
            if all(apply_rules(completed, seq[t]) == seq[t+1] for t in range(len(seq)-1)):
                if first_solution is None:
                    first_solution = completed
                if count_cap is not None:
                    key = tuple(sorted(completed.items()))
                    if key not in solutions:
                        solutions.append(key)
                        if len(solutions) >= count_cap:
                            raise StopIteration
                return completed
            return None
        for prod2 in gen_match(seq[i], seq[i+1], prod):
            res = solve_transition(i + 1, prod2)
            if count_cap is None and res is not None:
                return res
        return None

    status = 'unknown'
    capped = False
    try:
        prod = solve_transition(0, base_prod)
        status = 'success' if first_solution is not None else 'no-solution'
    except StopIteration:
        prod = first_solution
        status = 'success'
        capped = True
    except SearchLimit:
        prod = first_solution
        status = 'limit' if first_solution is None else 'success-limit'
    elapsed = time.perf_counter() - start
    valid = False
    if first_solution is not None:
        valid = all(apply_rules(first_solution, seq[i]) == seq[i+1] for i in range(len(seq)-1))
    return first_solution, dict(status=status, nodes=nodes, time_s=elapsed, valid=valid,
                                constants=len(constants), count=(len(solutions) if count_cap is not None else None),
                                count_capped=capped)

# ---------------------------------------------------------------------------
# QAOA statevector simulation, no qiskit
# ---------------------------------------------------------------------------

def qaoa_cost_array(n: int, edges: List[Tuple[int, int]], penalty: float = 2.0) -> np.ndarray:
    N = 1 << n
    idx = np.arange(N, dtype=np.uint64)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.float64)
    costs = -bits.sum(axis=1)
    for u, v in edges:
        costs += penalty * bits[:, u] * bits[:, v]
    return costs


def apply_mixer_inplace(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply exp(-i beta sum_j X_j) in place.

    This vectorized implementation is mathematically identical to the earlier
    block-by-block loop, but is substantially faster for the small statevector
    simulations reported in the paper.
    """
    c = math.cos(beta); s = -1j * math.sin(beta)
    for q in range(n):
        step = 1 << q
        period = step << 1
        arr = state.reshape(-1, period)
        a = arr[:, :step].copy()
        b = arr[:, step:period].copy()
        arr[:, :step] = c*a + s*b
        arr[:, step:period] = s*a + c*b
    return state


def qaoa_probs(params: np.ndarray, costs: np.ndarray, n: int, p: int) -> np.ndarray:
    N = len(costs)
    state = np.ones(N, dtype=np.complex128) / math.sqrt(N)
    gammas, betas = params[:p], params[p:]
    for layer in range(p):
        state *= np.exp(-1j * gammas[layer] * costs)
        apply_mixer_inplace(state, betas[layer], n)
    probs = np.abs(state)**2
    probs /= probs.sum()
    return probs


def qaoa_expectation(params: np.ndarray, costs: np.ndarray, n: int, p: int) -> float:
    return float(np.dot(qaoa_probs(params, costs, n, p), costs))


def optimize_qaoa(n: int, edges: List[Tuple[int, int]], p_list=(1,2,3), seed: int = 7) -> Dict[int, Dict[str, Any]]:
    """Deterministic multi-start statevector optimization for small QAOA cases.

    The optimizer is intentionally modest and reproducible.  It uses the
    standard expectation value of the corrected MIS Hamiltonian as the primary
    objective.  For depth p>1 it includes an exact embedding of the previous
    depth's parameters with a zero final layer, so the search space explicitly
    contains the shallower circuit.  A small number of seeded Nelder--Mead
    restarts are then used.  This is not a scalability claim; it is only the
    local simulator used for the small implementation checks in the paper.
    """
    costs = qaoa_cost_array(n, edges)
    c_min, c_max = float(costs.min()), float(costs.max())
    opt_mask = np.isclose(costs, c_min)
    n_opt = int(opt_mask.sum())
    uniform = n_opt / (1 << n)
    out: Dict[int, Dict[str, Any]] = {}
    prev: Optional[np.ndarray] = None

    for p in p_list:
        rng = np.random.default_rng(seed + 1009 * p)
        candidates: List[np.ndarray] = []

        if p > 1 and prev is not None:
            # The p-layer ansatz contains the (p-1)-layer ansatz: append a
            # zero gamma/beta layer.  This protects against artificial
            # non-monotonicity caused solely by a poor starting point.
            candidates.append(np.concatenate([prev[:p-1], [0.0], prev[p-1:], [0.0]]))

        # Seeded random starts in the standard angle ranges used for these
        # simulations.  The objective itself is periodic, so normalization is
        # only cosmetic and not needed for correctness.
        starts = 14
        for _ in range(starts):
            candidates.append(np.array(
                [rng.uniform(0, math.pi) for _ in range(p)] +
                [rng.uniform(0, math.pi/2) for _ in range(p)],
                dtype=float
            ))

        def objective(x: np.ndarray) -> float:
            return qaoa_expectation(np.asarray(x, dtype=float), costs, n, p)

        best: Optional[Tuple[float, float, np.ndarray]] = None
        for x0 in candidates:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 250, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            x = np.asarray(res.x, dtype=float)
            exp_val = objective(x)
            probs = qaoa_probs(x, costs, n, p)
            p_opt = float(probs[opt_mask].sum())
            # Primary key is expected cost; p_opt is only a numerical tie-breaker.
            key = (exp_val, -p_opt)
            if best is None or key < (best[0], -best[1]):
                best = (exp_val, p_opt, x)

        assert best is not None
        exp_val, p_opt, best_x = best
        prev = best_x
        appr = float((c_max - exp_val) / (c_max - c_min)) if c_max > c_min else 1.0
        out[p] = dict(expected_cost=float(exp_val), p_optimal=float(p_opt),
                      approximation_ratio=appr,
                      concentration=(p_opt / uniform if uniform else float('inf')),
                      n_opt=n_opt, uniform=uniform, ground_cost=c_min,
                      params=best_x.tolist())
    return out

def reconstruct_from_indices(indices: Iterable[int], seq: List[str], vertices: List[Vertex]) -> Dict[str, str]:
    prod = {}
    for idx in indices:
        i, j, s, e = vertices[idx]
        prod[seq[i][j]] = seq[i+1][s:e]
    for ch in set(''.join(seq)):
        prod.setdefault(ch, ch)
    return prod


def is_independent(indices: Iterable[int], adj: List[Set[int]]) -> bool:
    chosen = list(indices); chosen_set = set(chosen)
    return all((v not in adj[u]) for pos, u in enumerate(chosen) for v in chosen[pos+1:])


def validate_repair_sample(bit_int: int, seq: List[str], vertices: List[Vertex], cliques: List[List[int]], adj: List[Set[int]]) -> bool:
    selected = {i for i in range(len(vertices)) if (bit_int >> i) & 1}
    chosen: List[int] = []
    chosen_set: Set[int] = set()
    def ok(v: int) -> bool:
        return all(c not in adj[v] for c in chosen_set)
    for cv in cliques:
        # Prefer sampled vertices that keep independence, then any vertex that keeps independence.
        pick = None
        for v in cv:
            if v in selected and ok(v):
                pick = v; break
        if pick is None:
            # minimum degree is a deterministic, low-conflict fallback
            for v in sorted(cv, key=lambda x: (len(adj[x]), x)):
                if ok(v):
                    pick = v; break
        if pick is None:
            return False
        chosen.append(pick); chosen_set.add(pick)
    if not is_independent(chosen, adj):
        return False
    prod = reconstruct_from_indices(chosen, seq, vertices)
    return all(apply_rules(prod, seq[i]) == seq[i+1] for i in range(len(seq)-1))


def qaoa_instance(label: str, seq: List[str], constants: Optional[Set[str]] = None,
                  p_list=(1,2,3), shots: int = 4000, seed: int = 7) -> Dict[str, Any]:
    vertices, cliques, adj = build_graph(seq, constants=constants, max_vertices=20_000)
    edges = [(u, v) for u in range(len(vertices)) for v in adj[u] if u < v]
    res = optimize_qaoa(len(vertices), edges, p_list=p_list, seed=seed)
    rng = np.random.default_rng(seed + 987)
    costs = qaoa_cost_array(len(vertices), edges)
    for p, dd in res.items():
        probs = qaoa_probs(np.asarray(dd['params']), costs, len(vertices), p)
        samples = rng.choice(1 << len(vertices), size=shots, p=probs)
        good = sum(validate_repair_sample(int(b), seq, vertices, cliques, adj) for b in samples)
        dd['repair_to_d0l'] = good / shots
    stats = graph_stats_explicit(vertices, cliques, adj)
    return dict(label=label, lengths='-'.join(str(len(s)) for s in seq), n=stats['n'], edges=stats['edges'], k=stats['k'], depths=res)



def qaoa_worker(args: Tuple[str, List[str], Set[str], int, int]) -> Dict[str, Any]:
    """Top-level worker so QAOA cases can be evaluated in fresh processes."""
    label, seq, constants, seed, shots = args
    return qaoa_instance(label, seq, constants, p_list=(1,2,3), shots=shots, seed=seed)

# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def fmt_int(n: int) -> str:
    return f"{n:,}"


def corrupt_last_symbol(seq: List[str]) -> List[str]:
    seq2 = list(seq)
    last = seq2[-1]
    if not last:
        seq2[-1] = 'X'
    else:
        # use a fresh symbol to make the corruption maximally visible
        replacement = '#'
        if last[-1] == replacement:
            replacement = '$'
        seq2[-1] = last[:-1] + replacement
    return seq2


def run_all(lsystems_dir: Path, out_dir: Path, run_qaoa: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = load_instances(lsystems_dir)
    # Existence / classical validation
    classical_rows: List[Dict[str, Any]] = []
    for fname, data in instances:
        m, seq = minimal_steps_all_rule_predecessors(data)
        seq = seq[:m+1]
        constants = observed_constants(seq, data)
        stats_un = graph_stats_formula(seq, constants=set())
        stats_pr = graph_stats_formula(seq, constants=constants)
        prod, info = infer_global(seq, data.get('production_rules', {}).keys(), constants,
                                  time_limit=30.0, max_nodes=5_000_000, count_cap=None)
        exact = 0; mismatches = []
        if prod:
            for a, img in data.get('production_rules', {}).items():
                if prod.get(a) == img:
                    exact += 1
                else:
                    mismatches.append(a)
        classical_rows.append(dict(
            instance=fname.replace('.json','').replace('_problem',''),
            alphabet_observed=len(set(''.join(seq))),
            fixed_identity_symbols=len(constants),
            rules_to_infer=len(data.get('production_rules', {})),
            m=m,
            lengths='-'.join(str(len(s)) for s in seq),
            k=stats_pr['k'],
            vertices_unpruned=stats_un['vertices'],
            vertices_pruned=stats_pr['vertices'],
            max_clique_pruned=stats_pr['max_clique'],
            empty_domains=stats_pr['empty_domains'],
            status=info['status'],
            valid=info['valid'],
            time_ms=info['time_s']*1000.0,
            search_nodes=info['nodes'],
            compatible_count=('>=65' if info.get('count_capped') else (info.get('count') if info.get('count') is not None else '')),
            generating_rules_recovered=f"{exact}/{len(data.get('production_rules', {}))}",
            alternative_rule_symbols=''.join(mismatches)
        ))

    # Negative controls
    neg_specs: List[Tuple[str, List[str], Set[str], Set[str]]] = [
        ("hand: (ab, ab, ba)", ['ab','ab','ba'], set('ab'), set()),
        ("hand: (abc, abc, cba)", ['abc','abc','cba'], set('abc'), set()),
        ("hand: (aa, ab)", ['aa','ab'], set('ab'), set()),
        ("hand: (ab, abab, ab)", ['ab','abab','ab'], set('ab'), set()),
        ("hand: (abc, abcabc, abc)", ['abc','abcabc','abc'], set('abc'), set()),
    ]
    data_by_prefix = {fname[:3]: data for fname, data in instances}
    for prefix, label, steps in [('010','algae corrupted',3), ('011','Cantor dust corrupted',2), ('013','dragon curve corrupted',2)]:
        data = data_by_prefix[prefix]
        seq = corrupt_last_symbol(generate(data, steps))
        constants = observed_constants(seq, data)
        neg_specs.append((label, seq, set(data.get('production_rules', {}).keys()), constants))
    negative_rows = []
    for label, seq, prod_keys, constants in neg_specs:
        prod, info = infer_global(seq, prod_keys, constants, time_limit=20.0, max_nodes=2_000_000, count_cap=None)
        stats = graph_stats_formula(seq, constants=constants)
        negative_rows.append(dict(label=label, lengths='-'.join(str(len(s)) for s in seq), k=stats['k'], vertices=stats['vertices'], status=info['status'], valid=info['valid'], decision=('no compatible D0L' if info['status']=='no-solution' else 'CHECK')))

    # Ambiguity trajectories (constant-aware, capped)
    traj_cases = [('010','algae'), ('011','Cantor dust'), ('013','dragon curve'), ('027','Dipterosiphonia II'), ('031','Herposiphonia')]
    trajectory_rows = []
    for prefix, label in traj_cases:
        data = data_by_prefix[prefix]
        for m in range(1, 6):
            seq = generate(data, m)
            constants = observed_constants(seq, data)
            # only count if graph size is moderate enough for the recursive search
            _, info = infer_global(seq, data.get('production_rules', {}).keys(), constants,
                                   time_limit=5.0, max_nodes=600_000, count_cap=65)
            count = 'limit' if info['status'] == 'limit' else ('>=65' if info.get('count_capped') else info.get('count'))
            trajectory_rows.append(dict(system=label, m=m, lengths='-'.join(str(len(s)) for s in seq), compatible_count=count, status=info['status']))

    # Gate-count vs depth for explicit small graphs, using fixed-identity pruning when constants are known.
    gate_specs: List[Tuple[str, List[str], Set[str]]] = []
    gate_specs.append(("worked example", ['ab', 'aab'], set()))
    # Use generated small traces.
    for prefix, label, steps in [('030','Herpopteros',2), ('010','algae',3), ('011','Cantor dust',2), ('015','Pythagoras tree',2), ('013','dragon curve',2), ('016','Sierpinski triangle',2)]:
        data = data_by_prefix[prefix]
        seq = generate(data, steps)
        gate_specs.append((f"{label}, m={steps}", seq, observed_constants(seq, data)))
    gate_rows = []
    for label, seq, constants in gate_specs:
        vertices, cliques, adj = build_graph(seq, constants=constants, max_vertices=10_000)
        st = graph_stats_explicit(vertices, cliques, adj)
        depth = greedy_edge_coloring_depth(adj)
        gate_rows.append(dict(label=label, n=st['n'], edges=st['edges'], max_clique=st['max_clique'], max_degree=st['max_degree'], depth=depth, ratio=(st['edges']/depth if depth else 0)))

    # QAOA statevector small cases, all fixed-pruned when applicable.
    qaoa_cases: List[Tuple[str, List[str], Set[str]]] = [
        ("worked example", ['ab','aab'], set()),
        ("second example", ['ab','abab'], set()),
    ]
    algae = data_by_prefix['010']; seq = generate(algae, 2)
    qaoa_cases.append(("algae (A,AB,ABA)", seq, observed_constants(seq, algae)))
    dragon = data_by_prefix['013']; seq = generate(dragon, 1)
    qaoa_cases.append(("dragon curve, m=1 pruned", seq, observed_constants(seq, dragon)))
    qaoa_results = []
    if run_qaoa:
        # For a more robust no-Qiskit QAOA run, use run_qaoa_cases.py, which runs
        # each case in a fresh Python process.  The inline path is kept for small
        # local checks.
        qaoa_results = [qaoa_instance(label, seq, constants, p_list=(1,2,3), shots=1000, seed=7+i)
                        for i, (label, seq, constants) in enumerate(qaoa_cases)]

    # Write structured results
    with open(out_dir/'classical_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(classical_rows[0].keys()))
        w.writeheader(); w.writerows(classical_rows)
    with open(out_dir/'negative_controls.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(negative_rows[0].keys()))
        w.writeheader(); w.writerows(negative_rows)
    with open(out_dir/'ambiguity_trajectories.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        w.writeheader(); w.writerows(trajectory_rows)
    with open(out_dir/'gate_depth_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        w.writeheader(); w.writerows(gate_rows)
    if qaoa_results:
        with open(out_dir/'qaoa_results.json', 'w', encoding='utf-8') as f:
            json.dump(qaoa_results, f, indent=2)

    # LaTeX snippets
    def esc(x: Any) -> str:
        return str(x).replace('_','\\_').replace('#','\\#')
    with open(out_dir/'table_classical.tex','w',encoding='utf-8') as f:
        f.write('''\\begin{tabular}{lrrrrrrr}\n\\toprule\nInstance & $m$ & lengths & $k$ & $|V(G_{\\theta,C})|$ & status & valid & recovered \\\\\n\\midrule\n''')
        for r in classical_rows:
            f.write(f"{esc(r['instance'])} & {r['m']} & {r['lengths']} & {r['k']} & {fmt_int(r['vertices_pruned'])} & {r['status']} & {r['valid']} & {r['generating_rules_recovered']} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    with open(out_dir/'table_negative.tex','w',encoding='utf-8') as f:
        f.write('''\\begin{tabular}{lrrl}\n\\toprule\nInput & $k$ & $|V(G_{\\theta,C})|$ & Exact decision \\\\\n\\midrule\n''')
        for r in negative_rows:
            f.write(f"{esc(r['label'])} & {r['k']} & {fmt_int(r['vertices'])} & {r['decision']} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    with open(out_dir/'table_gate_depth.tex','w',encoding='utf-8') as f:
        f.write('''\\begin{tabular}{lrrrrr}\n\\toprule\nInstance & $n$ & $|E|$ & $\\omega$ & $\\Delta$ & depth \\\\\n\\midrule\n''')
        for r in gate_rows:
            f.write(f"{esc(r['label'])} & {fmt_int(r['n'])} & {fmt_int(r['edges'])} & {fmt_int(r['max_clique'])} & {fmt_int(r['max_degree'])} & {fmt_int(r['depth'])} \\\\\n")
        f.write('\\bottomrule\n\\end{tabular}\n')
    if qaoa_results:
        with open(out_dir/'table_qaoa.tex','w',encoding='utf-8') as f:
            f.write('''\\begin{tabular}{lrrrcccc}\n\\toprule\nInstance & $n$ & $|E|$ & $k$ & $p$ & appr. ratio & $P(\\mathrm{MIS})$ & repair$\\to$D0L \\\\\n\\midrule\n''')
            for ri, r in enumerate(qaoa_results):
                depths = r['depths']
                first = True
                for p in sorted(int(x) for x in depths.keys()):
                    dd = depths[str(p)] if str(p) in depths else depths[p]
                    inst = esc(r['label']) if first else ''
                    n = fmt_int(r['n']) if first else ''
                    E = fmt_int(r['edges']) if first else ''
                    k = r['k'] if first else ''
                    pmis = f"{dd['p_optimal']:.3f} ({dd['concentration']:.0f}$\\times$)"
                    f.write(f"{inst} & {n} & {E} & {k} & {p} & {dd['approximation_ratio']:.3f} & {pmis} & {dd['repair_to_d0l']:.3f} \\\\\n")
                    first = False
                if ri != len(qaoa_results)-1:
                    f.write('\\addlinespace\n')
            f.write('\\bottomrule\n\\end{tabular}\n')

    # Console summary
    print('CLASSICAL rows:', len(classical_rows), 'all_valid:', all(r['valid'] for r in classical_rows))
    print('NEGATIVE rows:', len(negative_rows), 'all_rejected:', all(r['decision']=='no compatible D0L' for r in negative_rows))
    print('GATE rows:', len(gate_rows))
    print('QAOA cases:', len(qaoa_results), '(run run_qaoa_cases.py for the statevector table)' if not qaoa_results else '')
    for r in qaoa_results:
        print(r['label'], 'n', r['n'], 'E', r['edges'], {p: round(dd['p_optimal'],4) for p,dd in r['depths'].items()})
    print('Wrote results to', out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lsystems-dir', default='lsystems')
    ap.add_argument('--out-dir', default='merged_results')
    ap.add_argument('--run-qaoa', action='store_true', help='Also run inline QAOA statevector simulations. For robust full QAOA tables, use run_qaoa_cases.py.')
    args = ap.parse_args()
    run_all(Path(args.lsystems_dir), Path(args.out_dir), run_qaoa=args.run_qaoa)

if __name__ == '__main__':
    main()
