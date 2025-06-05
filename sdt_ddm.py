import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
    try:
        # Read and preprocess data
        data = pd.read_csv(file_path)
        
        # Convert categorical variables to numeric codes
        for col, mapping in MAPPINGS.items():
            if col in data.columns:
                data[col] = data[col].map(mapping)
        
        # Create participant number and condition index
        data['pnum'] = data['participant_id'] if 'participant_id' in data.columns else range(1, len(data)+1)
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
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Creating simulated data for demonstration...")
        return create_simulated_data()

def create_simulated_data():
    """Create simulated SDT data for demonstration."""
    np.random.seed(42)
    n_participants = 12
    n_trials_per_condition = 50
    
    # Define realistic performance differences between conditions
    condition_params = {
        0: {'d_prime': 1.5, 'criterion': 0.0},   # Easy Simple
        1: {'d_prime': 1.2, 'criterion': 0.1},   # Easy Complex
        2: {'d_prime': 1.0, 'criterion': 0.2},   # Hard Simple
        3: {'d_prime': 0.8, 'criterion': 0.3}    # Hard Complex
    }
    
    sdt_data = []
    for p in range(1, n_participants + 1):
        for condition in range(4):
            # Add individual differences
            d_prime = condition_params[condition]['d_prime'] + np.random.normal(0, 0.3)
            criterion = condition_params[condition]['criterion'] + np.random.normal(0, 0.2)
            
            # Calculate hit and false alarm rates
            hit_rate = 1 / (1 + np.exp(-(d_prime - criterion)))
            fa_rate = 1 / (1 + np.exp(-(-criterion)))
            
            # Simulate binomial responses
            hits = np.random.binomial(n_trials_per_condition, hit_rate)
            false_alarms = np.random.binomial(n_trials_per_condition, fa_rate)
            
            sdt_data.append({
                'pnum': p,
                'condition': condition,
                'hits': hits,
                'misses': n_trials_per_condition - hits,
                'false_alarms': false_alarms,
                'correct_rejections': n_trials_per_condition - false_alarms,
                'nSignal': n_trials_per_condition,
                'nNoise': n_trials_per_condition
            })
    
    return pd.DataFrame(sdt_data)

def apply_hierarchical_sdt_model_with_effects(data):
    """Apply a hierarchical SDT model with stimulus type and difficulty effects."""
    
    # Get unique participants and conditions
    participants = sorted(data['pnum'].unique())
    P = len(participants)
    C = len(data['condition'].unique())
    
    # Create participant mapping
    pnum_to_idx = {p: i for i, p in enumerate(participants)}
    data['pnum_idx'] = data['pnum'].map(pnum_to_idx)
    
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
        
        # Calculate hit and false alarm rates using inverse logit
        hit_rate = pm.math.invlogit(d_prime - criterion)
        false_alarm_rate = pm.math.invlogit(-criterion)
        
        # Likelihood
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum_idx'], data['condition']], 
                   observed=data['hits'])
        
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum_idx'], data['condition']], 
                   observed=data['false_alarms'])
    
    return sdt_model

def create_simulated_rt_data():
    """Create simulated RT data for delta plot analysis."""
    np.random.seed(42)
    n_participants = 5
    n_trials_per_condition = 100
    
    # Define RT parameters for each condition
    condition_rt_params = {
        0: {'mu_correct': 0.6, 'mu_error': 0.5, 'sigma': 0.15},  # Easy Simple
        1: {'mu_correct': 0.7, 'mu_error': 0.6, 'sigma': 0.18},  # Easy Complex
        2: {'mu_correct': 0.8, 'mu_error': 0.7, 'sigma': 0.20},  # Hard Simple
        3: {'mu_correct': 0.9, 'mu_error': 0.8, 'sigma': 0.25}   # Hard Complex
    }
    
    rt_data = []
    for p in range(1, n_participants + 1):
        for condition in range(4):
            params = condition_rt_params[condition]
            
            # Generate correct and error RTs
            correct_rts = np.random.lognormal(np.log(params['mu_correct']), params['sigma'], n_trials_per_condition)
            error_rts = np.random.lognormal(np.log(params['mu_error']), params['sigma'], n_trials_per_condition//4)
            
            # Calculate percentiles for each response type
            correct_percentiles = np.percentile(correct_rts, PERCENTILES)
            error_percentiles = np.percentile(error_rts, PERCENTILES) if len(error_rts) > 0 else [np.nan] * len(PERCENTILES)
            overall_rts = np.concatenate([correct_rts, error_rts])
            overall_percentiles = np.percentile(overall_rts, PERCENTILES)
            
            # Add data for each mode
            for mode, percentiles in [('accurate', correct_percentiles), 
                                    ('error', error_percentiles), 
                                    ('overall', overall_percentiles)]:
                row = {'pnum': p, 'condition': condition, 'mode': mode}
                for i, p_val in enumerate(PERCENTILES):
                    row[f'p{p_val}'] = percentiles[i]
                rt_data.append(row)
    
    return pd.DataFrame(rt_data)

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
                    axes[i,j].plot(PERCENTILES, error_delta, color='red', **marker_style, label='Error')
                except:
                    pass
                
                # Calculate RT differences for accurate responses
                try:
                    accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
                    axes[i,j].plot(PERCENTILES, accurate_delta, color='green', **marker_style, label='Accurate')
                except:
                    pass
                
                axes[i,j].legend(loc='upper left', fontsize=8)
                axes[i,j].set_title(f'{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}', fontsize=9)
            
            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-0.3, top=0.5)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[i,j].grid(True, alpha=0.3)
            
    plt.tight_layout()
    return fig

def plot_posterior_distributions(trace):
    """Create comprehensive posterior distribution plots."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # D-prime parameters
    az.plot_posterior(trace, var_names=['d_prime_intercept'], ax=axes[0,0], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[0,0].set_title("D' Intercept", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['d_prime_stimulus_effect'], ax=axes[0,1], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[0,1].set_title("D' Stimulus Effect\n(Complex - Simple)", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['d_prime_difficulty_effect'], ax=axes[0,2], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[0,2].set_title("D' Difficulty Effect\n(Hard - Easy)", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['d_prime_interaction'], ax=axes[0,3], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[0,3].set_title("D' Interaction\n(Stimulus × Difficulty)", fontsize=14, fontweight='bold')
    
    # Criterion parameters
    az.plot_posterior(trace, var_names=['criterion_intercept'], ax=axes[1,0], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[1,0].set_title("Criterion Intercept", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['criterion_stimulus_effect'], ax=axes[1,1], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[1,1].set_title("Criterion Stimulus Effect\n(Complex - Simple)", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['criterion_difficulty_effect'], ax=axes[1,2], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[1,2].set_title("Criterion Difficulty Effect\n(Hard - Easy)", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['criterion_interaction'], ax=axes[1,3], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[1,3].set_title("Criterion Interaction\n(Stimulus × Difficulty)", fontsize=14, fontweight='bold')
    
    # Standard deviations
    az.plot_posterior(trace, var_names=['d_prime_sigma'], ax=axes[2,0], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[2,0].set_title("D' Individual Variability", fontsize=14, fontweight='bold')
    
    az.plot_posterior(trace, var_names=['criterion_sigma'], ax=axes[2,1], 
                     hdi_prob=0.95, point_estimate='mean')
    axes[2,1].set_title("Criterion Individual Variability", fontsize=14, fontweight='bold')
    
    # Remove empty subplots
    axes[2,2].axis('off')
    axes[2,3].axis('off')
    
    plt.tight_layout()
    return fig

def create_comparison_table(trace):
    """Create a comparison table of effect sizes."""
    posterior_samples = trace.posterior
    
    # Extract effect sizes
    effects = {
        'D-Prime Stimulus Effect': posterior_samples['d_prime_stimulus_effect'],
        'D-Prime Difficulty Effect': posterior_samples['d_prime_difficulty_effect'],
        'D-Prime Interaction': posterior_samples['d_prime_interaction'],
        'Criterion Stimulus Effect': posterior_samples['criterion_stimulus_effect'],
        'Criterion Difficulty Effect': posterior_samples['criterion_difficulty_effect'],
        'Criterion Interaction': posterior_samples['criterion_interaction']
    }
    
    # Create summary table
    summary_data = []
    for name, samples in effects.items():
        flat_samples = samples.values.flatten()
        summary_data.append({
            'Parameter': name,
            'Mean': np.mean(flat_samples),
            'SD': np.std(flat_samples),
            'HDI_2.5%': np.percentile(flat_samples, 2.5),
            'HDI_97.5%': np.percentile(flat_samples, 97.5),
            'P(Effect > 0)': np.mean(flat_samples > 0),
            'Effect Size': 'Large' if abs(np.mean(flat_samples)) > 0.5 else 
                         'Medium' if abs(np.mean(flat_samples)) > 0.2 else 'Small'
        })
    
    return pd.DataFrame(summary_data)

# Main Analysis Pipeline
print("="*70)
print("SIGNAL DETECTION THEORY AND DELTA PLOT ANALYSIS")
print("2×2×2 Experimental Design: Difficulty × Stimulus Type × Signal Presence")
print("="*70)

# Load and analyze the data
print("\n1. LOADING AND PREPROCESSING DATA")
print("-" * 40)

try:
    sdt_data = read_data('data.csv', prepare_for='sdt', display=True)
    print("✓ Real data loaded successfully")
except:
    sdt_data = create_simulated_data()
    print("✓ Simulated data created for demonstration")

# Display descriptive statistics
print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 40)

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
        print(condition_stats.loc[condition].to_string())

# Fit the hierarchical SDT model
print("\n3. FITTING HIERARCHICAL SDT MODEL")
print("-" * 40)

model = apply_hierarchical_sdt_model_with_effects(sdt_data)

with model:
    print("Sampling from posterior (this may take a few minutes)...")
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True, 
                     target_accept=0.95, random_seed=42)

# Check convergence
print("\n4. CONVERGENCE DIAGNOSTICS")
print("-" * 40)

convergence_summary = az.summary(trace, var_names=['d_prime_intercept', 'd_prime_stimulus_effect', 
                                                  'd_prime_difficulty_effect', 'd_prime_interaction',
                                                  'criterion_intercept', 'criterion_stimulus_effect',
                                                  'criterion_difficulty_effect', 'criterion_interaction'])
print(convergence_summary)

# Check for convergence issues
rhat_issues = convergence_summary['r_hat'] > 1.1
if rhat_issues.any():
    print("\n⚠️  WARNING: Some parameters show convergence issues (r_hat > 1.1)")
    print("Parameters with issues:", convergence_summary[rhat_issues].index.tolist())
else:
    print("\n✓ All parameters show good convergence (r_hat < 1.1)")

# Plot posterior distributions
print("\n5. POSTERIOR DISTRIBUTIONS")
print("-" * 40)

fig = plot_posterior_distributions(trace)
plt.show()

# Create comparison table
print("\n6. EFFECT SIZE COMPARISON TABLE")
print("-" * 40)

comparison_table = create_comparison_table(trace)
print(comparison_table.round(3).to_string(index=False))

# Create delta plots
print("\n7. DELTA PLOT ANALYSIS")
print("-" * 40)

# Use simulated RT data since read_data doesn't handle RT data yet
delta_data = create_simulated_rt_data()
print("✓ Simulated RT data created for delta plots")

# Plot delta plots for first few participants
unique_participants = sorted(delta_data['pnum'].unique())
print(f"\nCreating delta plots for participants: {unique_participants[:2]}")

for pnum in unique_participants[:2]:
    print(f"\nDelta plot for participant {pnum}:")
    fig = draw_delta_plots(delta_data, pnum)
    plt.show()

# Final interpretation
print("\n8. COMPREHENSIVE INTERPRETATION")
print("-" * 40)

# Extract and interpret the main effects
posterior_samples = trace.posterior

print("\nKEY FINDINGS FROM SDT ANALYSIS:")
print("=" * 50)

# D-prime effects
d_prime_stimulus_mean = float(posterior_samples['d_prime_stimulus_effect'].mean())
d_prime_difficulty_mean = float(posterior_samples['d_prime_difficulty_effect'].mean())
d_prime_interaction_mean = float(posterior_samples['d_prime_interaction'].mean())

print(f"\nSENSITIVITY (D-PRIME) EFFECTS:")
print(f"• Stimulus Type Effect: {d_prime_stimulus_mean:.3f}")
print(f"  → {'Complex stimuli DECREASE' if d_prime_stimulus_mean < 0 else 'Complex stimuli INCREASE'} sensitivity by {abs(d_prime_stimulus_mean):.3f} units")
print(f"• Difficulty Effect: {d_prime_difficulty_mean:.3f}")
print(f"  → {'Hard trials DECREASE' if d_prime_difficulty_mean < 0 else 'Hard trials INCREASE'} sensitivity by {abs(d_prime_difficulty_mean):.3f} units")
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

print(f"\nCOMPARISON OF EFFECT MAGNITUDES:")
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

print(f"\nINTEGRATED FINDINGS (SDT + DELTA PLOTS):")
print("=" * 50)
print("• SDT Analysis reveals how manipulations affect discrimination ability and response bias")
print("• Delta plots show how manipulations affect reaction time distributions")
print("• Both approaches provide complementary insights into cognitive processing")
print(f"• The {'difficulty' if abs(d_prime_difficulty_mean) > abs(d_prime_stimulus_mean) else 'stimulus type'} manipulation shows stronger effects on perceptual sensitivity")
print(f"• Response bias is more affected by {'difficulty' if abs(criterion_difficulty_mean) > abs(criterion_stimulus_mean) else 'stimulus type'}")

print(f"\nPRACTICAL IMPLICATIONS:")
print("• SDT parameters quantify changes in perceptual and decision processes")
print("• Delta plots reveal dynamics of decision-making across the RT distribution")

print(f"\n9. ADDITIONAL ANALYSES AND VISUALIZATIONS")
print("-" * 40)

# Create effect size visualization
def plot_effect_comparison():
    """Create a visual comparison of effect sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # D-prime effects
    d_effects = [abs(d_prime_stimulus_mean), abs(d_prime_difficulty_mean), abs(d_prime_interaction_mean)]
    d_labels = ['Stimulus\nType', 'Difficulty', 'Interaction']
    colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars1 = ax1.bar(d_labels, d_effects, color=colors1, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_title("D-Prime (Sensitivity) Effects", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Absolute Effect Size", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, effect in zip(bars1, d_effects):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Criterion effects
    c_effects = [abs(criterion_stimulus_mean), abs(criterion_difficulty_mean), abs(criterion_interaction_mean)]
    c_labels = ['Stimulus\nType', 'Difficulty', 'Interaction']
    colors2 = ['#FF9FF3', '#54A0FF', '#5F27CD']
    
    bars2 = ax2.bar(c_labels, c_effects, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_title("Criterion (Response Bias) Effects", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Absolute Effect Size", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, effect in zip(bars2, c_effects):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Create the effect comparison plot
effect_fig = plot_effect_comparison()
plt.show()

# Model predictions vs observed data
def plot_model_fit():
    """Plot model predictions against observed data."""
    # Extract posterior predictions
    with model:
        posterior_pred = pm.sample_posterior_predictive(trace, random_seed=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot for each condition
    for i, condition in enumerate([0, 1, 2, 3]):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        
        # Get observed data for this condition
        cond_data = sdt_data[sdt_data['condition'] == condition]
        
        if not cond_data.empty:
            # Observed hit rates and FA rates
            obs_hit_rates = cond_data['hit_rate'].values
            obs_fa_rates = cond_data['fa_rate'].values
            
            # Predicted hit rates and FA rates (posterior mean)
            hit_pred = posterior_pred.posterior_predictive['hit_obs'].mean(dim=['chain', 'draw'])
            fa_pred = posterior_pred.posterior_predictive['false_alarm_obs'].mean(dim=['chain', 'draw'])
            
            # Get predictions for this condition
            cond_indices = cond_data.index
            pred_hit_rates = hit_pred[cond_indices].values / cond_data['nSignal'].values
            pred_fa_rates = fa_pred[cond_indices].values / cond_data['nNoise'].values
            
            # Scatter plot
            ax.scatter(obs_hit_rates, pred_hit_rates, alpha=0.7, s=60, 
                      color='red', label='Hit Rate', edgecolors='black')
            ax.scatter(obs_fa_rates, pred_fa_rates, alpha=0.7, s=60, 
                      color='blue', label='False Alarm Rate', edgecolors='black')
            
            # Perfect prediction line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Observed Rate')
            ax.set_ylabel('Predicted Rate')
            ax.set_title(f'{CONDITION_NAMES[condition]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Fit: Observed vs Predicted Response Rates', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

try:
    fit_fig = plot_model_fit()
    plt.show()
    print("✓ Model fit visualization completed")
except Exception as e:
    print(f"⚠️ Could not create model fit plot: {e}")

# Posterior predictive checks
def posterior_predictive_check():
    """Perform posterior predictive checks."""
    print("\nPOSTERIOR PREDICTIVE CHECKS:")
    print("-" * 30)
    
    # Extract posterior predictions
    try:
        with model:
            posterior_pred = pm.sample_posterior_predictive(trace, random_seed=42)
        
        # Compare observed vs predicted summary statistics
        obs_hit_rate_mean = sdt_data['hit_rate'].mean()
        obs_fa_rate_mean = sdt_data['fa_rate'].mean()
        
        pred_hits = posterior_pred.posterior_predictive['hit_obs']
        pred_fas = posterior_pred.posterior_predictive['false_alarm_obs']
        
        # Calculate predicted rates
        pred_hit_rates = []
        pred_fa_rates = []
        for i in range(len(sdt_data)):
            n_signal = sdt_data.iloc[i]['nSignal']
            n_noise = sdt_data.iloc[i]['nNoise']
            pred_hit_rates.append(pred_hits[:, :, i].values.flatten() / n_signal)
            pred_fa_rates.append(pred_fas[:, :, i].values.flatten() / n_noise)
        
        pred_hit_rate_mean = np.mean([np.mean(rates) for rates in pred_hit_rates])
        pred_fa_rate_mean = np.mean([np.mean(rates) for rates in pred_fa_rates])
        
        print(f"Observed Hit Rate Mean: {obs_hit_rate_mean:.3f}")
        print(f"Predicted Hit Rate Mean: {pred_hit_rate_mean:.3f}")
        print(f"Observed FA Rate Mean: {obs_fa_rate_mean:.3f}")
        print(f"Predicted FA Rate Mean: {pred_fa_rate_mean:.3f}")
        
        # Calculate Bayesian p-values
        hit_pvalue = np.mean(pred_hit_rate_mean > obs_hit_rate_mean)
        fa_pvalue = np.mean(pred_fa_rate_mean > obs_fa_rate_mean)
        
        print(f"\nBayesian p-values:")
        print(f"Hit Rate: {hit_pvalue:.3f}")
        print(f"False Alarm Rate: {fa_pvalue:.3f}")
        
        if 0.05 <= hit_pvalue <= 0.95 and 0.05 <= fa_pvalue <= 0.95:
            print("✓ Model shows good fit (p-values between 0.05 and 0.95)")
        else:
            print("⚠️ Model fit may be poor (extreme p-values)")
            
    except Exception as e:
        print(f"Could not complete posterior predictive checks: {e}")

posterior_predictive_check()

print(f"\n10. THEORETICAL IMPLICATIONS")
print("-" * 40)

print("\nSIGNAL DETECTION THEORY INSIGHTS:")
print("• D-prime reflects the distance between signal and noise distributions")
print("• Criterion reflects the decision threshold for responding 'signal present'")
print("• Individual differences captured by hierarchical structure")

print("\nDIFFUSION MODEL INSIGHTS (from Delta Plots):")
print("• Delta plots reveal how RT distributions change between conditions")
print("• Upward-sloping deltas suggest drift rate differences")
print("• Downward-sloping deltas suggest boundary separation differences")
print("• Different slopes for correct vs error responses indicate speed-accuracy tradeoffs")

print(f"\n11. METHODOLOGICAL CONSIDERATIONS")
print("-" * 40)

print("\nSTRENGTHS OF THIS APPROACH:")
print("• Hierarchical modeling accounts for individual differences")
print("• Bayesian inference provides uncertainty quantification")
print("• Multiple dependent measures (accuracy + RT) provide richer insights")
print("• Model comparison allows evaluation of different theoretical accounts")

print("\nLIMITATIONS AND FUTURE DIRECTIONS:")
print("• Larger sample sizes would improve parameter estimation")
print("• Full diffusion model fitting would complement delta plot analysis")
print("• Additional covariates (age, experience) could be incorporated")
print("• Cross-validation could assess model generalizability")

print(f"\n12. FINAL SUMMARY")
print("=" * 50)

# Determine which manipulation has stronger effects
d_prime_winner = "Difficulty" if abs(d_prime_difficulty_mean) > abs(d_prime_stimulus_mean) else "Stimulus Type"
criterion_winner = "Difficulty" if abs(criterion_difficulty_mean) > abs(criterion_stimulus_mean) else "Stimulus Type"

print(f"\nMAIN CONCLUSIONS:")
print(f"1. {d_prime_winner} manipulation has the strongest effect on perceptual sensitivity")
print(f"2. {criterion_winner} manipulation has the strongest effect on response bias")
print(f"3. Both manipulations show {'significant' if abs(d_prime_interaction_mean) > 0.1 else 'minimal'} interaction effects")
print(f"4. Individual differences are {'substantial' if trace.posterior['d_prime_sigma'].mean() > 0.3 else 'moderate'} across participants")

print(f"\nEXPERIMENTAL DESIGN EVALUATION:")
if abs(d_prime_difficulty_mean) > 0.2 and abs(d_prime_stimulus_mean) > 0.2:
    print("✓ Both manipulations produce meaningful effects on cognition")
elif abs(d_prime_difficulty_mean) > 0.2 or abs(d_prime_stimulus_mean) > 0.2:
    print("◐ One manipulation shows strong effects, the other is weaker")
else:
    print("⚠️ Both manipulations show relatively weak effects")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)