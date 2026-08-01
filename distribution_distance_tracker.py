import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp, entropy

### Tracks how far apart two distributions are, over time
# Built for something like comparing a pitcher's velo distribution to a league/baseline
# distribution, and watching whether they drift together or apart across a season
# (works just as well for eg model score distributions, drift-checking a model in production)

### This is done with a generic "test_df" of dates, group labels, and a random stat value
# These can & should be replaced with a dataframe of your own observed date/group/stat values:
# - Remove the test_df generator code (below)
# - Replace references to "test_df" with references to your own dataframe
# - Replace any instance of "stat" with the stat you want to compare (eg velo, IVB, PLV, etc)

df_size = 4000
test_data = {
    'date': np.random.choice(pd.date_range('2024-04-01', '2024-09-30'), df_size),
    'group': np.random.choice(['pitcher', 'league'], df_size),
    'stat': np.random.normal(94, 2, df_size)
}
test_df = pd.DataFrame(test_data)

# Nudge the "pitcher" group's stat up over the course of the season, so there's an
# actual drift signal to pick up (rather than two distributions that never move apart)
days_in = (test_df['date'] - test_df['date'].min()).dt.days
is_pitcher = test_df['group'] == 'pitcher'
test_df.loc[is_pitcher, 'stat'] += days_in[is_pitcher] / 30 * 0.3

### Distance metrics between two 1D samples
def wasserstein(a, b):
    """Earth mover's distance - avg distance the mass has to move; same units as the stat itself"""
    return wasserstein_distance(a, b)

def ks_stat(a, b):
    """Max gap between the two empirical CDFs, on a 0-1 scale"""
    return ks_2samp(a, b).statistic

def js_divergence(a, b, bins=30):
    """Jensen-Shannon divergence - bounded [0, 1] (log base 2), a symmetric, smoothed version of KL"""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    eps = 1e-12  # avoids log(0) for empty bins
    return 0.5 * entropy(p + eps, m + eps, base=2) + 0.5 * entropy(q + eps, m + eps, base=2)

METRICS = {
    'wasserstein': wasserstein,
    'ks': ks_stat,
    'js_divergence': js_divergence,
}

### Buckets the data by time period, then computes distance(s) between the two groups in each bucket
def track_distance_over_time(df, date_col, group_col, stat_col, group_a, group_b,
                              freq='W', metrics=('wasserstein',), min_n=10):
    """
    Returns a dataframe with one row per time period, showing how far apart group_a's and
    group_b's distributions of `stat_col` are in that period.

    freq follows pandas period aliases (eg 'D' daily, 'W' weekly, 'ME' monthly)
    min_n is the fewest samples either group needs in a period for its distance to be trusted
    """
    df = df.copy()
    df['_period'] = pd.to_datetime(df[date_col]).dt.to_period(freq).dt.start_time

    rows = []
    for period, period_df in df.groupby('_period'):
        a = period_df.loc[period_df[group_col] == group_a, stat_col].dropna().values
        b = period_df.loc[period_df[group_col] == group_b, stat_col].dropna().values
        if len(a) < min_n or len(b) < min_n:
            continue  # too few samples that period to trust a distance estimate

        row = {'period': period, 'n_' + str(group_a): len(a), 'n_' + str(group_b): len(b)}
        for metric_name in metrics:
            row[metric_name] = METRICS[metric_name](a, b)
        rows.append(row)

    return pd.DataFrame(rows).sort_values('period').reset_index(drop=True)

### Same idea, but against a single fixed reference sample instead of a second group that
# also moves over time (eg tracking drift away from a model's training/validation distribution)
def track_distance_from_baseline(df, date_col, stat_col, baseline, freq='W',
                                  metrics=('wasserstein',), min_n=10):
    baseline = np.asarray(baseline)
    df = df.copy()
    df['_period'] = pd.to_datetime(df[date_col]).dt.to_period(freq).dt.start_time

    rows = []
    for period, period_df in df.groupby('_period'):
        sample = period_df[stat_col].dropna().values
        if len(sample) < min_n:
            continue

        row = {'period': period, 'n': len(sample)}
        for metric_name in metrics:
            row[metric_name] = METRICS[metric_name](sample, baseline)
        rows.append(row)

    return pd.DataFrame(rows).sort_values('period').reset_index(drop=True)

### Track it, using both wasserstein (in the stat's own units) and JS divergence (bounded, unitless)
dist_df = track_distance_over_time(
    test_df, date_col='date', group_col='group', stat_col='stat',
    group_a='pitcher', group_b='league',
    freq='W', metrics=['wasserstein', 'js_divergence'], min_n=10
)

### Plot the tracked distance(s) over time
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(dist_df['period'], dist_df['wasserstein'], color='tab:blue', marker='o', label='Wasserstein')
ax1.set_ylabel('Wasserstein Distance (units of stat)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(dist_df['period'], dist_df['js_divergence'], color='tab:red', marker='s', label='JS Divergence')
ax2.set_ylabel('Jensen-Shannon Divergence (0-1)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.set_xlabel('Week')
ax1.set_title('Distance Between Pitcher & League Distributions, Over Time')
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig('distribution_distance_tracker_example.png', dpi=150)
