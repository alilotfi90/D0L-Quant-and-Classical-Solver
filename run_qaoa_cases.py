#!/usr/bin/env python3
"""Run QAOA statevector cases for the D0L paper.

Each case can be run in a fresh Python interpreter.  Running the script without
--case launches four separate subprocesses and combines their JSON outputs.
This avoids cumulative state in long numerical runs and does not require Qiskit.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

import merged_d0l_experiments as exp


def get_cases(lsystems_dir: Path):
    instances = exp.load_instances(lsystems_dir)
    data_by_prefix = {fname[:3]: data for fname, data in instances}
    cases = [
        ("worked example", ['ab','aab'], set()),
        ("second example", ['ab','abab'], set()),
    ]
    algae = data_by_prefix['010']; seq = exp.generate(algae, 2)
    cases.append(("algae (A,AB,ABA)", seq, exp.observed_constants(seq, algae)))
    dragon = data_by_prefix['013']; seq = exp.generate(dragon, 1)
    cases.append(("dragon curve, m=1 pruned", seq, exp.observed_constants(seq, dragon)))
    return cases


def write_latex_table(results, path: Path):
    def esc(x):
        return str(x).replace('_','\\_').replace('#','\\#')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('''\\begin{tabular}{lrrrcccc}\n\\toprule\nInstance & $n$ & $|E|$ & $k$ & $p$ & appr. ratio & $P(\\mathrm{MIS})$ & repair$\\to$D0L \\\\\n\\midrule\n''')
        for ri, r in enumerate(results):
            first = True
            for p in sorted(int(x) for x in r['depths'].keys()):
                dd = r['depths'][str(p)] if str(p) in r['depths'] else r['depths'][p]
                inst = esc(r['label']) if first else ''
                n = f"{r['n']:,}" if first else ''
                E = f"{r['edges']:,}" if first else ''
                k = r['k'] if first else ''
                pmis = f"{dd['p_optimal']:.3f} ({dd['concentration']:.0f}$\\times$)"
                f.write(f"{inst} & {n} & {E} & {k} & {p} & {dd['approximation_ratio']:.3f} & {pmis} & {dd['repair_to_d0l']:.3f} \\\\\n")
                first = False
            if ri != len(results) - 1:
                f.write('\\addlinespace\n')
        f.write('\\bottomrule\n\\end{tabular}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lsystems-dir', default='lsystems')
    ap.add_argument('--out-dir', default='merged_results')
    ap.add_argument('--case', type=int, default=None, help='Run only one case index and write JSON to --case-out')
    ap.add_argument('--case-out', default=None)
    ap.add_argument('--shots', type=int, default=1000)
    args = ap.parse_args()
    lsystems_dir = Path(args.lsystems_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cases = get_cases(lsystems_dir)
    if args.case is not None:
        label, seq, constants = cases[args.case]
        result = exp.qaoa_instance(label, seq, constants, p_list=(1,2,3), shots=args.shots, seed=7+args.case)
        if args.case_out:
            Path(args.case_out).write_text(json.dumps(result, indent=2), encoding='utf-8')
        else:
            print(json.dumps(result, indent=2))
        return

    results = []
    for i in range(len(cases)):
        case_file = out_dir / f'qaoa_case_{i}.json'
        cmd = [sys.executable, str(Path(__file__).resolve()), '--lsystems-dir', str(lsystems_dir),
               '--out-dir', str(out_dir), '--case', str(i), '--case-out', str(case_file), '--shots', str(args.shots)]
        print('running QAOA case', i, cases[i][0], flush=True)
        subprocess.run(cmd, check=True, timeout=180)
        results.append(json.loads(case_file.read_text(encoding='utf-8')))
    (out_dir/'qaoa_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    write_latex_table(results, out_dir/'table_qaoa.tex')
    print('wrote', out_dir/'qaoa_results.json')
    print('wrote', out_dir/'table_qaoa.tex')

if __name__ == '__main__':
    main()
