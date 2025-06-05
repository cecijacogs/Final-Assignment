import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import warnings

# Constants
PERCENTILES = [10, 30, 50, 70, 90]
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# === Load and Process Data ===

def read_data(file_path, prepare_for='sdt', display=False):
    data = pd.read_csv(file_path)
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)

    if prepare_for == 'sdt':
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({'accuracy': ['count', 'sum']}).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        return pd.DataFrame(sdt_data)

    elif prepare_for == 'delta plots':
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', *[f'p{p}' for p in PERCENTILES]])
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                if c_data.empty:
                    continue
                for mode, subset in [('overall', c_data),
                                     ('accurate', c_data[c_data['accuracy'] == 1]),
                                     ('error', c_data[c_data['accuracy'] == 0])]:
                    if not subset.empty:
                        quantiles = {f'p{p}': [np.percentile(subset['rt'], p)] for p in PERCENTILES}
                        dp_data = pd.concat([dp_data, pd.DataFrame({
                            'pnum': [pnum],
                            'condition': [condition],
                            'mode': [mode],
                            **quantiles
                        })])
        return dp_data.reset_index(drop=True)

# === Model ===

def apply_hierarchical_sdt_model(data):
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    with pm.Model() as sdt_model:
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
        pm.Binomial('hit_obs', 
                    n=data['nSignal'], 
                    p=hit_rate[data['pnum'] - 1, data['condition']], 
                    observed=data['hits'])
        pm.Binomial('false_alarm_obs', 
                    n=data['nNoise'], 
                    p=false_alarm_rate[data['pnum'] - 1, data['condition']], 
                    observed=data['false_alarms'])
    return sdt_model

# === Delta Plot Drawing ===

def draw_delta_plots(data, pnum):
    data = data[data['pnum'] == pnum]
    conditions = sorted(data['condition'].unique())
    n = len(conditions)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    marker_style = {'marker': 'o', 'markersize': 8, 'markerfacecolor': 'white', 'markeredgewidth': 2, 'linewidth': 2}
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i > j:
                continue
            ax = axes[i, j]
            ax.set_ylim(-0.3, 0.5)
            ax.axhline(0, color='gray', linestyle='--')
            if i == j:
                ax.axis('off')
                continue
            def get_deltas(mode):
                q1 = np.array([data[(data['condition'] == cond1) & (data['mode'] == mode)][f'p{p}'].values[0] for p in PERCENTILES])
                q2 = np.array([data[(data['condition'] == cond2) & (data['mode'] == mode)][f'p{p}'].values[0] for p in PERCENTILES])
                return q2 - q1
            ax.plot(PERCENTILES, get_deltas('overall'), color='black', **marker_style)
            axes[j, i].plot(PERCENTILES, get_deltas('error'), color='red', **marker_style)
            axes[j, i].plot(PERCENTILES, get_deltas('accurate'), color='green', **marker_style)
            axes[j, i].legend(['Error', 'Accurate'], loc='upper left')
            axes[i, j].set_title(f'{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'delta_plots_p{pnum}.png')
    plt.close()

# === Delta Contrast Summary Plot ===

def plot_delta_contrast(delta_data, cond_a, cond_b, label, mode='overall'):
    percentiles = [f'p{p}' for p in PERCENTILES]
    diffs = []
    for p in delta_data['pnum'].unique():
        d1 = delta_data[(delta_data['pnum'] == p) & (delta_data['condition'] == cond_a) & (delta_data['mode'] == mode)]
        d2 = delta_data[(delta_data['pnum'] == p) & (delta_data['condition'] == cond_b) & (delta_data['mode'] == mode)]
        if not d1.empty and not d2.empty:
            q1 = d1[percentiles].values[0]
            q2 = d2[percentiles].values[0]
            diffs.append(q2 - q1)
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs, axis=0)
    sem_diff = np.std(diffs, axis=0) / np.sqrt(len(diffs))

    plt.errorbar(PERCENTILES, mean_diff, yerr=sem_diff, label=label, marker='o', capsize=5)

# === Main ===

def main():
    print("Loading and preparing SDT data...")
    sdt_data = read_data("data.csv", prepare_for='sdt', display=True)
    if sdt_data.empty:
        raise ValueError("SDT data is empty. Check data format.")

    print("Fitting hierarchical SDT model...")
    sdt_model = apply_hierarchical_sdt_model(sdt_data)
    with sdt_model:
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    print("Model convergence summary:")
    summary = az.summary(trace, var_names=["mean_d_prime", "mean_criterion"], round_to=2)
    print(summary)
    summary.to_csv(OUTPUT_DIR / "sdt_summary.csv")

    az.plot_forest(trace, var_names=["mean_d_prime", "mean_criterion"], combined=True)
    plt.title("Posterior Distributions for d′ and Criterion")
    plt.savefig(OUTPUT_DIR / "sdt_posteriors.png")
    plt.close()

    # === Compare SDT condition means and contrasts ===
    mean_d = trace.posterior['mean_d_prime'].mean(dim=("chain", "draw")).values
    mean_c = trace.posterior['mean_criterion'].mean(dim=("chain", "draw")).values

    print("\n--- SDT Condition Means ---")
    for i in range(4):
        print(f"{CONDITION_NAMES[i]}: d′ = {mean_d[i]:.2f}, criterion = {mean_c[i]:.2f}")

    print("\n--- SDT Contrasts ---")
    print(f"Stimulus Type effect (Easy): Complex - Simple: d′ = {mean_d[1] - mean_d[0]:.2f}")
    print(f"Stimulus Type effect (Hard): Complex - Simple: d′ = {mean_d[3] - mean_d[2]:.2f}")
    print(f"Difficulty effect (Simple): Hard - Easy: d′ = {mean_d[2] - mean_d[0]:.2f}")
    print(f"Difficulty effect (Complex): Hard - Easy: d′ = {mean_d[3] - mean_d[1]:.2f}")

    print("Generating delta plots...")
    delta_data = read_data("data.csv", prepare_for='delta plots', display=False)
    for p in delta_data['pnum'].unique():
        draw_delta_plots(delta_data, p)

    print("Computing RT differences between Hard Complex and Easy Simple...")
    diff_list = []
    for p in delta_data['pnum'].unique():
        pdata = delta_data[delta_data['pnum'] == p]
        for mode in ['overall', 'accurate', 'error']:
            cond0 = pdata[(pdata['condition'] == 0) & (pdata['mode'] == mode)]
            cond3 = pdata[(pdata['condition'] == 3) & (pdata['mode'] == mode)]
            if not cond0.empty and not cond3.empty:
                rt0 = cond0[[f'p{q}' for q in PERCENTILES]].values[0].astype(float)
                rt3 = cond3[[f'p{q}' for q in PERCENTILES]].values[0].astype(float)
                diff = rt3 - rt0
                diff_list.append({
                    'pnum': p,
                    'mode': mode,
                    **{f'diff_p{q}': diff[i] for i, q in enumerate(PERCENTILES)}
                })

    diff_df = pd.DataFrame(diff_list)
    diff_df.to_csv(OUTPUT_DIR / 'rt_differences_HardComplex_vs_EasySimple.csv', index=False)
    print("RT differences saved to output/rt_differences_HardComplex_vs_EasySimple.csv")

    print("Plotting delta contrast summary...")
    plt.figure(figsize=(8, 6))
    plot_delta_contrast(delta_data, 0, 1, 'Stimulus Type (Easy)')
    plot_delta_contrast(delta_data, 2, 3, 'Stimulus Type (Hard)')
    plot_delta_contrast(delta_data, 0, 2, 'Difficulty (Simple)')
    plot_delta_contrast(delta_data, 1, 3, 'Difficulty (Complex)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Delta Plot: RT Difference (by Percentile)")
    plt.xlabel("Percentile")
    plt.ylabel("RT Difference (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delta_plot_contrasts.png")
    plt.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
