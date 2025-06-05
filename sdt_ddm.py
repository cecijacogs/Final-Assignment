"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
Modified to quantify effects of Stimulus Type and Trial Difficulty
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
from scipy import stats

# Mapping dictionaries for categorical variables
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex', 
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format."""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return None
    
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        if col in data.columns:
            data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id'] if 'participant_id' in data.columns else range(len(data))
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print(f"\nData shape: {data.shape}")
        print(f"Participants: {data['pnum'].nunique()}")
        print(f"Conditions: {sorted(data['condition'].unique())}")
        print(f"Signal values: {sorted(data['signal'].unique())}")
    
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        # Transform into SDT format
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    hits = signal_trials['correct'].iloc[0]
                    n_signal = signal_trials['nTrials'].iloc[0]
                    false_alarms = noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0]
                    n_noise = noise_trials['nTrials'].iloc[0]
                    
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': hits,
                        'misses': n_signal - hits,
                        'false_alarms': false_alarms,
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': n_signal,
                        'nNoise': n_noise,
                        'stimulus_type': condition % 2,  # 0=simple, 1=complex
                        'difficulty': condition // 2    # 0=easy, 1=hard
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display and not data.empty:
            print("\nSDT summary by condition:")
            summary = data.groupby('condition').agg({
                'hits': 'sum',
                'misses': 'sum', 
                'false_alarms': 'sum',
                'correct_rejections': 'sum',
                'nSignal': 'sum',
                'nNoise': 'sum'
            })
            summary['hit_rate'] = summary['hits'] / summary['nSignal']
            summary['fa_rate'] = summary['false_alarms'] / summary['nNoise']
            print(summary.round(3))
    
    elif prepare_for == 'delta plots':
        # Process data for delta plot analysis
        dp_data = []
        
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                if len(c_data) == 0:
                    continue
                
                # Calculate percentiles for different response types
                for mode in ['overall', 'accurate', 'error']:
                    if mode == 'overall':
                        rt_data = c_data['rt']
                    elif mode == 'accurate':
                        rt_data = c_data[c_data['accuracy'] == 1]['rt']
                    else:  # error
                        rt_data = c_data[c_data['accuracy'] == 0]['rt']
                    
                    if len(rt_data) > 0:
                        percentiles = {f'p{p}': np.percentile(rt_data, p) for p in PERCENTILES}
                        dp_data.append({
                            'pnum': pnum,
                            'condition': condition,
                            'mode': mode,
                            **percentiles,
                            'stimulus_type': condition % 2,
                            'difficulty': condition // 2
                        })
        
        data = pd.DataFrame(dp_data)
    
    return data

def apply_hierarchical_sdt_model_with_effects(data):
    """Apply hierarchical SDT model with explicit stimulus type and difficulty effects."""
    
    # Get unique participants and conditions
    participants = sorted(data['pnum'].unique())
    P = len(participants)
    
    # Map participant IDs to indices
    pnum_to_idx = {p: i for i, p in enumerate(participants)}
    data['pnum_idx'] = data['pnum'].map(pnum_to_idx)
    
    print(f"Building model with {P} participants")
    print(f"Conditions in data: {sorted(data['condition'].unique())}")
    
    with pm.Model() as sdt_model:
        # Grand mean parameters
        grand_mean_dprime = pm.Normal('grand_mean_dprime', mu=1.0, sigma=1.0)
        grand_mean_criterion = pm.Normal('grand_mean_criterion', mu=0.0, sigma=1.0)
        
        # Main effects on d-prime
        stimulus_effect_dprime = pm.Normal('stimulus_effect_dprime', mu=0.0, sigma=0.5)
        difficulty_effect_dprime = pm.Normal('difficulty_effect_dprime', mu=0.0, sigma=0.5)
        interaction_effect_dprime = pm.Normal('interaction_effect_dprime', mu=0.0, sigma=0.5)
        
        # Main effects on criterion
        stimulus_effect_criterion = pm.Normal('stimulus_effect_criterion', mu=0.0, sigma=0.5)
        difficulty_effect_criterion = pm.Normal('difficulty_effect_criterion', mu=0.0, sigma=0.5)
        interaction_effect_criterion = pm.Normal('interaction_effect_criterion', mu=0.0, sigma=0.5)
        
        # Individual participant random effects
        participant_dprime = pm.Normal('participant_dprime', mu=0.0, sigma=0.5, shape=P)
        participant_criterion = pm.Normal('participant_criterion', mu=0.0, sigma=0.5, shape=P)
        
        # Calculate condition-specific means
        dprime_means = pm.Deterministic('dprime_means', 
            grand_mean_dprime + 
            stimulus_effect_dprime * data['stimulus_type'].values +
            difficulty_effect_dprime * data['difficulty'].values +
            interaction_effect_dprime * data['stimulus_type'].values * data['difficulty'].values +
            participant_dprime[data['pnum_idx'].values]
        )
        
        criterion_means = pm.Deterministic('criterion_means',
            grand_mean_criterion +
            stimulus_effect_criterion * data['stimulus_type'].values +
            difficulty_effect_criterion * data['difficulty'].values +
            interaction_effect_criterion * data['stimulus_type'].values * data['difficulty'].values +
            participant_criterion[data['pnum_idx'].values]
        )
        
        # SDT calculations
        hit_rate = pm.Deterministic('hit_rate', pm.math.invprobit(dprime_means/2 - criterion_means))
        fa_rate = pm.Deterministic('fa_rate', pm.math.invprobit(-criterion_means))
        
        # Likelihood
        pm.Binomial('hits_obs', n=data['nSignal'].values, p=hit_rate, observed=data['hits'].values)
        pm.Binomial('fa_obs', n=data['nNoise'].values, p=fa_rate, observed=data['false_alarms'].values)
    
    return sdt_model

def fit_model_and_check_convergence(model, draws=2000, tune=1000, chains=4):
    """Fit the model and check convergence diagnostics."""
    
    print("Fitting model...")
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, 
                         target_accept=0.9, random_seed=42)
    
    print("\nChecking convergence...")
    
    try:
        # R-hat values - check individual parameters to avoid memory issues
        rhat_values = []
        param_names = list(trace.posterior.data_vars.keys())
        
        for param in param_names:
            param_rhat = az.rhat(trace.posterior[param])
            if hasattr(param_rhat, 'values'):
                if param_rhat.values.ndim == 0:  # scalar
                    rhat_values.append(float(param_rhat.values))
                else:  # array
                    rhat_values.extend(param_rhat.values.flatten())
            else:
                rhat_values.append(float(param_rhat))
        
        max_rhat = max(rhat_values)
        print(f"Maximum R-hat: {max_rhat:.4f}")
        
        if max_rhat > 1.01:
            print("WARNING: Some parameters may not have converged (R-hat > 1.01)")
        else:
            print("Good convergence: All R-hat values <= 1.01")
        
        # Effective sample size - check individual parameters
        ess_values = []
        for param in param_names:
            param_ess = az.ess(trace.posterior[param])
            if hasattr(param_ess, 'values'):
                if param_ess.values.ndim == 0:  # scalar
                    ess_values.append(float(param_ess.values))
                else:  # array
                    ess_values.extend(param_ess.values.flatten())
            else:
                ess_values.append(float(param_ess))
        
        min_ess = min(ess_values)
        print(f"Minimum ESS: {min_ess:.0f}")
        
        if min_ess < 400:
            print("WARNING: Low effective sample size for some parameters")
        else:
            print("Good effective sample size for all parameters")
            
    except Exception as e:
        print(f"Could not compute convergence diagnostics: {e}")
        print("Proceeding with analysis despite convergence check failure...")
    
    return trace

def plot_posterior_effects(trace):
    """Plot posterior distributions of main effects and interactions."""
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Extract effect parameters
    effects_dprime = ['stimulus_effect_dprime', 'difficulty_effect_dprime', 'interaction_effect_dprime']
    effects_criterion = ['stimulus_effect_criterion', 'difficulty_effect_criterion', 'interaction_effect_criterion']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot d-prime effects
    for i, effect in enumerate(effects_dprime):
        ax = axes[0, i]
        samples = trace.posterior[effect].values.flatten()
        ax.hist(samples, bins=50, alpha=0.7, density=True)
        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax.set_title(f"{effect.replace('_', ' ').title()}")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Density")
        
        # Add summary statistics
        mean_val = np.mean(samples)
        hdi = az.hdi(samples, hdi_prob=0.95)
        ax.text(0.05, 0.95, f"Mean: {mean_val:.3f}\n95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot criterion effects  
    for i, effect in enumerate(effects_criterion):
        ax = axes[1, i]
        samples = trace.posterior[effect].values.flatten()
        ax.hist(samples, bins=50, alpha=0.7, density=True)
        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax.set_title(f"{effect.replace('_', ' ').title()}")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Density")
        
        # Add summary statistics
        mean_val = np.mean(samples)
        hdi = az.hdi(samples, hdi_prob=0.95)
        ax.text(0.05, 0.95, f"Mean: {mean_val:.3f}\n95% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'posterior_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_effects_summary_table(trace):
    """Create a summary table of all effects."""
    
    effects = [
        'grand_mean_dprime', 'grand_mean_criterion',
        'stimulus_effect_dprime', 'difficulty_effect_dprime', 'interaction_effect_dprime',
        'stimulus_effect_criterion', 'difficulty_effect_criterion', 'interaction_effect_criterion'
    ]
    
    summary_data = []
    for effect in effects:
        samples = trace.posterior[effect].values.flatten()
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        hdi = az.hdi(samples, hdi_prob=0.95)
        prob_positive = np.mean(samples > 0)
        
        summary_data.append({
            'Parameter': effect,
            'Mean': mean_val,
            'SD': std_val,
            '95% HDI Lower': hdi[0],
            '95% HDI Upper': hdi[1],
            'P(Effect > 0)': prob_positive
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("POSTERIOR SUMMARY TABLE")
    print("="*80)
    print(summary_df.round(3).to_string(index=False))
    print("="*80)
    
    return summary_df

def draw_enhanced_delta_plots(data):
    """Draw delta plots comparing RT distributions between conditions."""
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    participants = sorted(data['pnum'].unique())
    
    for pnum in participants[:3]:  # Plot first 3 participants as examples
        participant_data = data[data['pnum'] == pnum]
        conditions = sorted(participant_data['condition'].unique())
        
        if len(conditions) < 2:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Delta Plots - Participant {pnum}', fontsize=16)
        
        comparisons = [
            ('Easy Simple vs Easy Complex', 0, 1),
            ('Easy Simple vs Hard Simple', 0, 2), 
            ('Easy Complex vs Hard Complex', 1, 3),
            ('Hard Simple vs Hard Complex', 2, 3)
        ]
        
        for idx, (title, cond1, cond2) in enumerate(comparisons):
            if cond1 not in conditions or cond2 not in conditions:
                continue
                
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # Get data for both conditions
            data1 = participant_data[participant_data['condition'] == cond1]
            data2 = participant_data[participant_data['condition'] == cond2]
            
            # Plot overall RT differences
            overall1 = data1[data1['mode'] == 'overall']
            overall2 = data2[data2['mode'] == 'overall']
            
            if not overall1.empty and not overall2.empty:
                delta_overall = []
                for p in PERCENTILES:
                    val1 = overall1[f'p{p}'].iloc[0] if not overall1.empty else 0
                    val2 = overall2[f'p{p}'].iloc[0] if not overall2.empty else 0
                    delta_overall.append(val2 - val1)
                
                ax.plot(PERCENTILES, delta_overall, 'ko-', linewidth=2, 
                       markersize=8, label='Overall', markerfacecolor='white')
            
            # Plot accurate vs error RT differences
            acc1 = data1[data1['mode'] == 'accurate']
            acc2 = data2[data2['mode'] == 'accurate']
            err1 = data1[data1['mode'] == 'error']
            err2 = data2[data2['mode'] == 'error']
            
            if not acc1.empty and not acc2.empty:
                delta_acc = []
                for p in PERCENTILES:
                    val1 = acc1[f'p{p}'].iloc[0] if not acc1.empty else 0
                    val2 = acc2[f'p{p}'].iloc[0] if not acc2.empty else 0
                    delta_acc.append(val2 - val1)
                ax.plot(PERCENTILES, delta_acc, 'go-', linewidth=2, 
                       markersize=6, label='Accurate', alpha=0.7)
            
            if not err1.empty and not err2.empty:
                delta_err = []
                for p in PERCENTILES:
                    val1 = err1[f'p{p}'].iloc[0] if not err1.empty else 0
                    val2 = err2[f'p{p}'].iloc[0] if not err2.empty else 0
                    delta_err.append(val2 - val1)
                ax.plot(PERCENTILES, delta_err, 'ro-', linewidth=2,
                       markersize=6, label='Error', alpha=0.7)
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('RT Difference (s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'delta_plots_participant_{pnum}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main_analysis():
    """Run the complete analysis pipeline."""
    
    print("="*80)
    print("ENHANCED SDT AND DELTA PLOT ANALYSIS")
    print("="*80)
    
    # Define the data file path (looking for data.csv in the current directory)
    data_file = "data.csv"
    
    # Load and prepare data
    print(f"\n1. Loading and preparing data from '{data_file}'...")
    sdt_data = read_data(data_file, prepare_for='sdt', display=True)
    
    if sdt_data is None or sdt_data.empty:
        print("ERROR: Could not load or process data")
        return
    
    # Fit hierarchical SDT model
    print("\n2. Building and fitting hierarchical SDT model...")
    model = apply_hierarchical_sdt_model_with_effects(sdt_data)
    trace = fit_model_and_check_convergence(model)
    
    # Analyze results
    print("\n3. Analyzing posterior distributions...")
    plot_posterior_effects(trace)
    summary_df = create_effects_summary_table(trace)
    
    # Delta plot analysis
    print("\n4. Preparing delta plot analysis...")
    dp_data = read_data(data_file, prepare_for='delta plots', display=True)
    
    if dp_data is not None and not dp_data.empty:
        print("\n5. Creating delta plots...")
        draw_enhanced_delta_plots(dp_data)
    
    print("\n6. Analysis complete! Check the 'output' directory for figures.")
    
    # Summary of findings
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    # Extract key effects
    stim_dprime = trace.posterior['stimulus_effect_dprime'].values.flatten()
    diff_dprime = trace.posterior['difficulty_effect_dprime'].values.flatten()
    stim_criterion = trace.posterior['stimulus_effect_criterion'].values.flatten()
    diff_criterion = trace.posterior['difficulty_effect_criterion'].values.flatten()
    
    print(f"\nStimulus Type Effects:")
    print(f"  - On Sensitivity (d'): Mean = {np.mean(stim_dprime):.3f}, P(>0) = {np.mean(stim_dprime > 0):.3f}")
    print(f"  - On Criterion: Mean = {np.mean(stim_criterion):.3f}, P(>0) = {np.mean(stim_criterion > 0):.3f}")
    
    print(f"\nTrial Difficulty Effects:")
    print(f"  - On Sensitivity (d'): Mean = {np.mean(diff_dprime):.3f}, P(>0) = {np.mean(diff_dprime > 0):.3f}")
    print(f"  - On Criterion: Mean = {np.mean(diff_criterion):.3f}, P(>0) = {np.mean(diff_criterion > 0):.3f}")
    
    return trace, summary_df

if __name__ == "__main__":
    trace, summary = main_analysis()