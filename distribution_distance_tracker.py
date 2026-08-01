import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp, entropy

### Tracks how far apart two distributions are, over time
# Built for something like comparing a pitcher's velo distribution to a league/baseline
# distribution, and watching whether they drift together or apart across a season
# (works just as well for eg model score distributions, drift-checking a model in production)

### This is done with a generic "test_df" of dates, group labels, and random x/z location
# values (eg pitch location) - a 2D stat, since that's what this is built for
# These can & should be replaced with a dataframe of your own observed date/group/x/z values:
# - Remove the test_df generator code (below)
# - Replace references to "test_df" with references to your own dataframe
# - Replace 'x'/'z' with whichever 2 columns make up the joint stat you want to compare
#   (eg pitch location, or velo & IVB together) - also works with just 1 column

df_size = 4000
test_data = {
    'date': np.random.choice(pd.date_range('2024-04-01', '2024-09-30'), df_size),
    'group': np.random.choice(['pitcher', 'league'], df_size),
    'x': np.random.normal(0, 3, df_size),
    'z': np.random.normal(30, 4, df_size)
}
test_df = pd.DataFrame(test_data)

# Nudge the "pitcher" group's location up & in over the course of the season, so there's an
# actual drift signal to pick up (rather than two distributions that never move apart)
days_in = (test_df['date'] - test_df['date'].min()).dt.days
is_pitcher = test_df['group'] == 'pitcher'
test_df.loc[is_pitcher, 'x'] += days_in[is_pitcher] / 30 * 0.2
test_df.loc[is_pitcher, 'z'] += days_in[is_pitcher] / 30 * 0.3

### Distance metrics between two samples. `a` & `b` can each be a 1D array of shape (n,)
# for a single stat, or a 2D array of shape (n, d) for a joint/multi-dimensional stat
def wasserstein(a, b):
    """Earth mover's distance - avg distance the mass has to move; same units as the stat itself
    1D samples only"""
    return wasserstein_distance(a, b)

def ks_stat(a, b):
    """Max gap between the two empirical CDFs, on a 0-1 scale
    1D samples only"""
    return ks_2samp(a, b).statistic

def js_divergence(a, b, bins=20):
    """Jensen-Shannon divergence - bounded [0, 1] (log base 2), a symmetric, smoothed version of KL
    Works for 1D or multi-dimensional (eg joint x/z location) samples. `bins` is per dimension,
    so drop it for higher-dimensional stats where samples get spread thin across the grid"""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 1:
        a, b = a.reshape(-1, 1), b.reshape(-1, 1)

    lo, hi = np.minimum(a.min(axis=0), b.min(axis=0)), np.maximum(a.max(axis=0), b.max(axis=0))
    edges = [np.linspace(lo[dim], hi[dim], bins + 1) for dim in range(a.shape[1])]

    p, _ = np.histogramdd(a, bins=edges, density=True)
    q, _ = np.histogramdd(b, bins=edges, density=True)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    eps = 1e-12  # avoids log(0) for empty bins
    return 0.5 * entropy(p.ravel() + eps, m.ravel() + eps, base=2) \
        + 0.5 * entropy(q.ravel() + eps, m.ravel() + eps, base=2)

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

    stat_col can be a single column name (1D stat) or a list of column names (joint/multi-
    dimensional stat, eg ['x', 'z'] for location) - use 'js_divergence' as the metric for the
    multi-dimensional case, since wasserstein/ks are 1D-only
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
                                  metrics=('js_divergence',), min_n=10):
    """stat_col & baseline follow the same 1D-vs-multi-dimensional shape rules as
    track_distance_over_time above (baseline should have the same shape as stat_col: an
    (n,) array for a single stat, or an (n, d) array/dataframe of the same d columns)"""
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

### Track it, comparing the joint (x, z) location distributions with JS divergence, since
# wasserstein/ks don't extend to multi-dimensional stats
dist_df = track_distance_over_time(
    test_df, date_col='date', group_col='group', stat_col=['x', 'z'],
    group_a='pitcher', group_b='league',
    freq='W', metrics=['js_divergence'], min_n=10
)

### Plot the tracked distance over time
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dist_df['period'], dist_df['js_divergence'], color='tab:red', marker='o')
ax.set_ylabel('Jensen-Shannon Divergence (0-1)')
ax.set_xlabel('Week')
ax.set_title('Distance Between Pitcher & League Location Distributions, Over Time')
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig('distribution_distance_tracker_example.png', dpi=150)
