import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import seaborn as sns

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
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
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
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum', 
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    return data

def apply_hierarchical_sdt_model_with_effects(data):
    """Apply a hierarchical SDT model with stimulus type and difficulty effects."""
    
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    # Create design matrix for conditions
    # Condition coding: 0=Easy Simple, 1=Easy Complex, 2=Hard Simple, 3=Hard Complex
    stimulus_type = np.array([0, 1, 0, 1])  # 0=simple, 1=complex
    difficulty = np.array([0, 0, 1, 1])     # 0=easy, 1=hard
    
    with pm.Model() as sdt_model:
        # Group-level intercepts
        d_prime_intercept = pm.Normal('d_prime_intercept', mu=1.0, sigma=1.0)
        criterion_intercept = pm.Normal('criterion_intercept', mu=0.0, sigma=1.0)
        
        # Group-level effects
        d_prime_stimulus_effect = pm.Normal('d_prime_stimulus_effect', mu=0.0, sigma=0.5)
        d_prime_difficulty_effect = pm.Normal('d_prime_difficulty_effect', mu=0.0, sigma=0.5)
        d_prime_interaction = pm.Normal('d_prime_interaction', mu=0.0, sigma=0.5)
        
        criterion_stimulus_effect = pm.Normal('criterion_stimulus_effect', mu=0.0, sigma=0.5)
        criterion_difficulty_effect = pm.Normal('criterion_difficulty_effect', mu=0.0, sigma=0.5)
        criterion_interaction = pm.Normal('criterion_interaction', mu=0.0, sigma=0.5)
        
        # Group-level standard deviations
        d_prime_sigma = pm.HalfNormal('d_prime_sigma', sigma=0.5)
        criterion_sigma = pm.HalfNormal('criterion_sigma', sigma=0.5)
        
        # Condition-level means
        d_prime_mu = (d_prime_intercept + 
                     d_prime_stimulus_effect * stimulus_type +
                     d_prime_difficulty_effect * difficulty +
                     d_prime_interaction * stimulus_type * difficulty)
        
        criterion_mu = (criterion_intercept +
                       criterion_stimulus_effect * stimulus_type +
                       criterion_difficulty_effect * difficulty +
                       criterion_interaction * stimulus_type * difficulty)
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime', mu=d_prime_mu, sigma=d_prime_sigma, shape=(P, C))
        criterion = pm.Normal('criterion', mu=criterion_mu, sigma=criterion_sigma, shape=(P, C))
        
        # Calculate hit and false alarm rates
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
        
        # Likelihood
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
    
    return sdt_model

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs."""
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = sorted(data['condition'].unique())
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 8,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 2
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=10)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=10)
                
            # Skip diagonal
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Upper triangle: overall RT differences
            if i < j:
                # Create masks for condition and plotting mode
                cmask1 = data['condition'] == cond1
                cmask2 = data['condition'] == cond2
                overall_mask = data['mode'] == 'overall'
                
                # Calculate RT differences for overall performance
                quantiles1 = [data[cmask1 & overall_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                quantiles2 = [data[cmask2 & overall_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                overall_delta = np.array(quantiles2) - np.array(quantiles1)
                
                # Plot overall RT differences
                axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
                axes[i,j].set_title(f'{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}', fontsize=9)
            
            # Lower triangle: error vs accurate RT differences
            else:
                # Create masks for condition and plotting mode
                cmask1 = data['condition'] == cond1
                cmask2 = data['condition'] == cond2
                error_mask = data['mode'] == 'error'
                accurate_mask = data['mode'] == 'accurate'
                
                # Calculate RT differences for error responses
                try:
                    error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
                    axes[i,j].plot(PERCENTILES, error_delta, color='red', **marker_style)
                except:
                    pass
                
                # Calculate RT differences for accurate responses
                try:
                    accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
                    axes[i,j].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
                except:
                    pass
                
                axes[i,j].legend(['Error', 'Accurate'], loc='upper left', fontsize=8)
                axes[i,j].set_title(f'{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}', fontsize=9)
            
            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-0.3, top=0.5)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[i,j].grid(True, alpha=0.3)
            
    plt.tight_layout()
    return fig

# Load and analyze the data
print("Loading and preprocessing data...")
try:
    sdt_data = read_data('data.csv', prepare_for='sdt', display=True)
except FileNotFoundError:
    print("Error: data.csv not found. Please make sure the file is in the current directory.")
    # Create some example data for demonstration
    print("Creating example data for demonstration...")
    np.random.seed(42)
    n_participants = 10
    n_trials_per_condition = 50
    
    example_data = []
    for p in range(1, n_participants + 1):
        for condition in range(4):
            for signal in range(2):
                n_trials = n_trials_per_condition
                # Simulate different performance based on condition
                if condition == 0:  # Easy Simple
                    accuracy_prob = 0.85
                elif condition == 1:  # Easy Complex
                    accuracy_prob = 0.80
                elif condition == 2:  # Hard Simple
                    accuracy_prob = 0.75
                else:  # Hard Complex
                    accuracy_prob = 0.70
                
                correct = np.random.binomial(n_trials, accuracy_prob)
                
                example_data.append({
                    'pnum': p,
                    'condition': condition,
                    'signal': signal,
                    'nTrials': n_trials,
                    'correct': correct,
                    'hits': correct if signal == 0 else n_trials - correct,
                    'misses': n_trials - correct if signal == 0 else correct,
                    'false_alarms': n_trials - correct if signal == 1 else correct,
                    'correct_rejections': correct if signal == 1 else n_trials - correct,
                    'nSignal': n_trials if signal == 0 else 0,
                    'nNoise': n_trials if signal == 1 else 0
                })
    
    # Convert to proper SDT format
    sdt_data = []
    for p in range(1, n_participants + 1):
        for condition in range(4):
            signal_data = next(d for d in example_data if d['pnum'] == p and d['condition'] == condition and d['signal'] == 0)
            noise_data = next(d for d in example_data if d['pnum'] == p and d['condition'] == condition and d['signal'] == 1)
            
            sdt_data.append({
                'pnum': p,
                'condition': condition,
                'hits': signal_data['correct'],
                'misses': signal_data['nTrials'] - signal_data['correct'],
                'false_alarms': noise_data['nTrials'] - noise_data['correct'],
                'correct_rejections': noise_data['correct'],
                'nSignal': signal_data['nTrials'],
                'nNoise': noise_data['nTrials']
            })
    
    sdt_data = pd.DataFrame(sdt_data)
    print("Example SDT data created successfully.")

# Display descriptive statistics
print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)

# Calculate hit rates and false alarm rates
sdt_data['hit_rate'] = sdt_data['hits'] / sdt_data['nSignal']
sdt_data['fa_rate'] = sdt_data['false_alarms'] / sdt_data['nNoise']

print("\nHit rates and False Alarm rates by condition:")
condition_stats = sdt_data.groupby('condition').agg({
    'hit_rate': ['mean', 'std'],
    'fa_rate': ['mean', 'std'],
    'hits': 'sum',
    'false_alarms': 'sum',
    'nSignal': 'sum',
    'nNoise': 'sum'
}).round(3)

condition_stats.columns = ['Hit_Rate_Mean', 'Hit_Rate_SD', 'FA_Rate_Mean', 'FA_Rate_SD', 
                          'Total_Hits', 'Total_FAs', 'Total_Signal_Trials', 'Total_Noise_Trials']

for i, condition in enumerate([0, 1, 2, 3]):
    print(f"\n{CONDITION_NAMES[condition]}:")
    if condition in condition_stats.index:
        print(condition_stats.loc[condition])

# Fit the hierarchical SDT model
print("\n" + "="*50)
print("FITTING HIERARCHICAL SDT MODEL")
print("="*50)

model = apply_hierarchical_sdt_model_with_effects(sdt_data)

with model:
    # Sample from posterior
    print("Sampling from posterior...")
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True, 
                     target_accept=0.95, random_seed=42)

# Check convergence
print("\nConvergence diagnostics:")
print(az.summary(trace, var_names=['d_prime_intercept', 'd_prime_stimulus_effect', 
                                  'd_prime_difficulty_effect', 'd_prime_interaction',
                                  'criterion_intercept', 'criterion_stimulus_effect',
                                  'criterion_difficulty_effect', 'criterion_interaction']))

# Plot posterior distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# D-prime parameters
az.plot_posterior(trace, var_names=['d_prime_intercept'], ax=axes[0,0])
axes[0,0].set_title("D' Intercept")

az.plot_posterior(trace, var_names=['d_prime_stimulus_effect'], ax=axes[0,1])
axes[0,1].set_title("D' Stimulus Effect")

az.plot_posterior(trace, var_names=['d_prime_difficulty_effect'], ax=axes[0,2])
axes[0,2].set_title("D' Difficulty Effect")

az.plot_posterior(trace, var_names=['d_prime_interaction'], ax=axes[0,3])
axes[0,3].set_title("D' Interaction")

# Criterion parameters
az.plot_posterior(trace, var_names=['criterion_intercept'], ax=axes[1,0])
axes[1,0].set_title("Criterion Intercept")

az.plot_posterior(trace, var_names=['criterion_stimulus_effect'], ax=axes[1,1])
axes[1,1].set_title("Criterion Stimulus Effect")

az.plot_posterior(trace, var_names=['criterion_difficulty_effect'], ax=axes[1,2])
axes[1,2].set_title("Criterion Difficulty Effect")

az.plot_posterior(trace, var_names=['criterion_interaction'], ax=axes[1,3])
axes[1,3].set_title("Criterion Interaction")

plt.tight_layout()
plt.show()

# Create delta plots
print("\n" + "="*50)
print("PREPARING DELTA PLOTS DATA")
print("="*50)

try:
    delta_data = read_data('data.csv', prepare_for='delta plots', display=True)
    
    # Plot delta plots for first few participants
    unique_participants = sorted(delta_data['pnum'].unique())
    print(f"\nCreating delta plots for participants: {unique_participants[:3]}")
    
    for pnum in unique_participants[:3]:
        print(f"\nCreating delta plot for participant {pnum}...")
        fig = draw_delta_plots(delta_data, pnum)
        plt.show()
        
except FileNotFoundError:
    print("Cannot create delta plots without actual RT data from data.csv")
    print("Delta plots require reaction time data which is not available in the simulated data.")

print("\n" + "="*50)
print("ANALYSIS COMPLETE - INTERPRETATION")
print("="*50)

# Extract and interpret the main effects
posterior_samples = trace.posterior

print("\nKEY FINDINGS:")
print("="*30)

# D-prime effects
d_prime_stimulus_mean = float(posterior_samples['d_prime_stimulus_effect'].mean())
d_prime_difficulty_mean = float(posterior_samples['d_prime_difficulty_effect'].mean())
d_prime_interaction_mean = float(posterior_samples['d_prime_interaction'].mean())

print(f"\nSENSITIVITY (D-PRIME) EFFECTS:")
print(f"• Stimulus Type Effect: {d_prime_stimulus_mean:.3f}")
print(f"  → {'Complex stimuli DECREASE' if d_prime_stimulus_mean < 0 else 'Complex stimuli INCREASE'} sensitivity")
print(f"• Difficulty Effect: {d_prime_difficulty_mean:.3f}")
print(f"  → {'Hard trials DECREASE' if d_prime_difficulty_mean < 0 else 'Hard trials INCREASE'} sensitivity")
print(f"• Interaction Effect: {d_prime_interaction_mean:.3f}")
print(f"  → {'Negative' if d_prime_interaction_mean < 0 else 'Positive'} interaction between stimulus type and difficulty")

# Criterion effects  
criterion_stimulus_mean = float(posterior_samples['criterion_stimulus_effect'].mean())
criterion_difficulty_mean = float(posterior_samples['criterion_difficulty_effect'].mean())
criterion_interaction_mean = float(posterior_samples['criterion_interaction'].mean())

print(f"\nRESPONSE BIAS (CRITERION) EFFECTS:")
print(f"• Stimulus Type Effect: {criterion_stimulus_mean:.3f}")
print(f"  → {'Complex stimuli increase conservative bias' if criterion_stimulus_mean > 0 else 'Complex stimuli increase liberal bias'}")
print(f"• Difficulty Effect: {criterion_difficulty_mean:.3f}")
print(f"  → {'Hard trials increase conservative bias' if criterion_difficulty_mean > 0 else 'Hard trials increase liberal bias'}")
print(f"• Interaction Effect: {criterion_interaction_mean:.3f}")

print(f"\nCOMPARISON OF EFFECTS:")
print(f"• Stimulus Type vs Difficulty on Sensitivity: {abs(d_prime_stimulus_mean):.3f} vs {abs(d_prime_difficulty_mean):.3f}")
if abs(d_prime_stimulus_mean) > abs(d_prime_difficulty_mean):
    print("  → Stimulus Type has LARGER effect on sensitivity")
else:
    print("  → Difficulty has LARGER effect on sensitivity")

print(f"• Stimulus Type vs Difficulty on Bias: {abs(criterion_stimulus_mean):.3f} vs {abs(criterion_difficulty_mean):.3f}")
if abs(criterion_stimulus_mean) > abs(criterion_difficulty_mean):
    print("  → Stimulus Type has LARGER effect on response bias")
else:
    print("  → Difficulty has LARGER effect on response bias")

print(f"\nMODEL INTERPRETATION:")
print("• D-prime (sensitivity) measures the ability to discriminate signal from noise")
print("• Criterion (bias) measures the tendency to respond 'signal present' vs 'signal absent'")
print("• Negative d-prime effects indicate poorer discrimination ability")
print("• Positive criterion effects indicate more conservative responding")

print(f"\nPRACTICAL IMPLICATIONS:")
if d_prime_difficulty_mean < 0:
    print("• Harder trials make discrimination more difficult (expected)")
if d_prime_stimulus_mean < 0:
    print("• Complex stimuli make discrimination more difficult")
if abs(d_prime_difficulty_mean) > abs(d_prime_stimulus_mean):
    print("• The difficulty manipulation has a stronger impact than stimulus complexity")
else:
    print("• Stimulus complexity has a stronger impact than trial difficulty")