#!/usr/bin/env python3
"""
Convert experiment results (JSON) into formatted tables for the paper.
Reads from results/ directory and prints tables ready for paper insertion.

Usage:
    python scripts/generate_tables.py --results-dir ./outputs/results
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir):
    results = {}
    for study in ["study1", "study2", "study3", "study4"]:
        path = os.path.join(results_dir, study, f"{study}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[study] = json.load(f)
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results["summary"] = json.load(f)
    return results


def fmt(val, precision=4):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def fmt_ci(point, lo, hi, precision=3):
    return f"{point:.{precision}f} [{lo:.{precision}f}, {hi:.{precision}f}]"


def print_table3(s1):
    """Table 3: Scaling of Deception Probe Performance."""
    print("\n" + "="*70)
    print("TABLE 3: Scaling of Deception Probe Performance on Gemma 3 (D-RepE)")
    print("="*70)
    print(f"{'Model':<10} {'AUROC':<30} {'Acc@1%FPR':<30} {'Best Layer':<12} {'B→W Boost':<12}")
    print("-"*94)
    
    for size in ["1B", "4B", "12B", "27B"]:
        data = s1.get("scaling", {}).get(size, {})
        if not data:
            continue
        
        metrics = data.get("test_metrics", {})
        auc = metrics.get("auroc", {})
        acc1 = metrics.get("acc_at_1pct_fpr", {})
        b2w = metrics.get("b2w_boost", {})
        
        auc_str = fmt_ci(auc.get("point",0), auc.get("ci_lower",0), auc.get("ci_upper",0)) if auc else "—"
        acc1_str = fmt_ci(acc1.get("point",0), acc1.get("ci_lower",0), acc1.get("ci_upper",0)) if acc1 else "—"
        
        print(f"{size+'-IT':<10} {auc_str:<30} {acc1_str:<30} {data.get('best_layer','—'):<12} {fmt(b2w.get('point')):<12}")


def print_table4(s1):
    """Table 4: Peak Layer Analysis."""
    print("\n" + "="*70)
    print("TABLE 4: Peak Layer Analysis with Bootstrap Uncertainty")
    print("="*70)
    print(f"{'Model':<10} {'Layers':<8} {'Peak (Median)':<15} {'95% CI':<15} {'% Depth':<12} {'2/3 Rule?':<10}")
    print("-"*70)
    
    for size in ["4B", "12B", "27B"]:
        data = s1.get("peak_layers", {}).get(size, {})
        scaling = s1.get("scaling", {}).get(size, {})
        if not data:
            continue
        
        print(f"{size+'-IT':<10} {scaling.get('n_layers','—'):<8} "
              f"{fmt(data.get('median_peak_layer'), 0):<15} "
              f"[{fmt(data.get('ci_lower'), 0)}, {fmt(data.get('ci_upper'), 0)}]{'':>3} "
              f"{fmt(data.get('peak_as_fraction'), 2):<12} "
              f"{'Yes' if data.get('two_thirds_rule_overlap') else 'No':<10}")


def print_table6(s2):
    """Table 6: Cross-Domain Transfer Matrix."""
    print("\n" + "="*70)
    print("TABLE 6: Cross-Domain Transfer AUROC Matrix (Gemma 3-27B-IT)")
    print("="*70)
    
    matrix = s2.get("transfer", {}).get("matrix", {})
    datasets = ["repe", "role", "sand", "mask"]
    
    header = f"{'Train↓/Test→':<15}" + "".join(f"{d:<12}" for d in ["D-RepE", "D-Role", "D-Sand", "D-MASK"])
    print(header)
    print("-"*63)
    
    for ds_train in datasets:
        row = f"{('D-'+ds_train.capitalize()):<15}"
        for ds_test in datasets:
            val = matrix.get(ds_train, {}).get(ds_test)
            cell = fmt(val) if val is not None else "—"
            if ds_train == ds_test:
                cell = f"*{cell}*"
            row += f"{cell:<12}"
        print(row)
    
    p1a = s2.get("transfer", {}).get("p1a_holds")
    print(f"\nP1a (all off-diagonal ≥ 0.90): {'SUPPORTED' if p1a else 'REJECTED'}")


def print_table10(s3):
    """Table 10: Five-Probe Battery."""
    print("\n" + "="*70)
    print("TABLE 10: Five-Probe Battery Against RL-Obfuscated Model")
    print("="*70)
    
    fp = s3.get("five_probe", {})
    
    print(f"{'Probe':<20} {'Description':<35} {'Pre-RL':<10} {'Post-RL':<10} {'ΔAUROC':<10}")
    print("-"*85)
    
    for probe_key in ["probe_a", "probe_b", "probe_c", "probe_d", "probe_e"]:
        data = fp.get(probe_key, {})
        name = data.get("name", probe_key)
        pre = data.get("pre_rl_auroc", "—")
        post = data.get("post_rl_auroc", data.get("best_auroc", "—"))
        delta = data.get("delta", "—")
        
        desc_map = {
            "probe_a": "Targeted (in RL loss)",
            "probe_b": "Retrained (fresh probe, post-RL)",
            "probe_c": "Unseen layer (±5 from target)",
            "probe_d": "Nonlinear (MLP, post-RL data)",
            "probe_e": "Ensemble (5-layer majority vote)",
        }
        
        print(f"{name:<20} {desc_map.get(probe_key, ''):<35} {fmt(pre):<10} {fmt(post):<10} {fmt(delta):<10}")
    
    print(f"\nScenario: {fp.get('scenario', '—')} — {fp.get('scenario_desc', '')}")


def print_table_entropy(s4):
    """Tables 13-15: Entropy results."""
    print("\n" + "="*70)
    print("TABLE 13-15: Entropy Analysis Results")
    print("="*70)
    
    for model_size in s4:
        if model_size == "summary":
            continue
        for ds_name in s4[model_size]:
            data = s4[model_size][ds_name]
            gap = data.get("entropy_gap", {})
            corr = data.get("correlation", {})
            resid = data.get("residualization", {})
            
            print(f"\n--- {model_size} / {ds_name} ---")
            print(f"  Entropy gap: H(honest)={fmt(gap.get('honest_mean'))}, "
                  f"H(deceptive)={fmt(gap.get('deceptive_mean'))}, "
                  f"Cohen's d={fmt(gap.get('cohens_d'))}, p={fmt(gap.get('welch_p'))}")
            print(f"  Correlation: ρ={fmt(corr.get('spearman_rho'))}, "
                  f"P4a={'✓' if corr.get('p4a_supported') else '✗'}")
            print(f"  Residualization: Original={fmt(resid.get('original_auroc'))}, "
                  f"Residualized={fmt(resid.get('residualized_auroc'))}, "
                  f"Δ={fmt(resid.get('delta_auroc'))}")
            print(f"  P4b={'✓' if resid.get('p4b_supported') else '✗'}, "
                  f"P4c={'✓' if resid.get('p4c_supported') else '✗'}")
            print(f"  Interpretation: {resid.get('interpretation', '—')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./outputs/results")
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    if "study1" in results:
        print_table3(results["study1"])
        print_table4(results["study1"])
    
    if "study2" in results:
        print_table6(results["study2"])
    
    if "study3" in results:
        print_table10(results["study3"])
    
    if "study4" in results:
        print_table_entropy(results["study4"])
    
    # Hypothesis summary
    print("\n" + "="*70)
    print("HYPOTHESIS ADJUDICATION SUMMARY")
    print("="*70)
    
    summary = results.get("summary", {})
    s2 = summary.get("study2", {})
    s3 = summary.get("study3", {})
    s4 = summary.get("study4", {})
    
    print(f"  H-LIN: {'SUPPORTED' if s2.get('p1a_holds') else 'REJECTED' if s2.get('p1a_holds') is not None else 'Not tested'}")
    print(f"  H-SUB vs H-CONE: {s2.get('geometry_conclusion', 'Not tested')}")
    print(f"  H-ENT: {s4.get('interpretation', 'Not tested')}")
    print(f"  Adversarial scenario: {s3.get('scenario_desc', 'Not tested')}")


if __name__ == "__main__":
    main()
