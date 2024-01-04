import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.nonparametric.kernel_regression import KernelReg

### This is done with a generic "test_df" of X-locations, Z-locations, and a random value ("test_stat")
# These can & should be replaced with a dataframe of observed X/Z/stat values (rounded to the nearest inch):
# - Remove the test_df generator code (below)
# - Replace references to "test_df" with references to your own dataframe 
# - Replace any instance of "test_stat" with the stat you want to show 

### Generate a test dataframe with 1000 observations, using normally distributed X/Z coords and stat values
# X is centered at 0"
# Z (vertical) is centered at 30" (aka 2.5ft above ground)

df_size = 1000
avg_test_value = 0.5 # This is the population's average/median value for the stat

test_data = {
  'x':np.random.normal(0,3,df_size).astype('int'),
  'z':np.random.normal(30,4,df_size).astype('int'),
  'test_stat':np.random.normal(avg_test_value,0.15,df_size) # 0.15 is a randomly chosen StDev
}
test_df = pd.DataFrame(test_data)

### Generate a df with every possible X/Z location to fill missing data
# Adds 20" horizontally and 30 inches vertically, in each direction
# I'm sure there's a better way to do this lol
zone_df = pd.DataFrame(columns=['x','z'])
for x in range(-20,21):
    for y in range(0,61):
        zone_df.loc[len(zone_df)] = [x,y]

### Merge observed data onto the completed 2D space
# Fill missing values with the average stat, from the *whole population*
test_df = pd.merge(zone_df,
                   test_df,
                   on=['x','z'],
                   how='left').fillna(avg_test_value)

### Smooth the stat values
# Specify the bandwidth
# I eyeballed the value for my stats, but I'm sure there's plenty of literature on it
# Lower = peakier, higher = smoother
bandwidth = 2

# Train a kernel regression model on the location and stat values, using the provided bandwidth
kernel_regression = KernelReg(endog=test_df['test_stat'], 
                              exog= [test_df['x'], test_df['z']], 
                              bw=[bandwidth,bandwidth],
                              var_type='cc') # States that X & Z are both continuous

# Apply the model onto the X/Z coords
test_df['smoothed_stat'] = kernel_regression.fit([test_df['x'], test_df['z']])[0]

### Generate viz
# Pivot df to 2D, to better play with Seaborn's heatmap code 
heatmap_df = test_df.pivot_table(columns='x',index='z',values=['smoothed_stat'], aggfunc='mean')

fig, ax = plt.subplots(figsize=(5,6))
# Create Chart
sns.heatmap(data=heatmap_df['smoothed_stat'].astype('float'),
            cmap='vlag', # My preferred diverging palette, but use whatever you want
            center=avg_test_value # Force the color scale to center on your average/median value
            )

# Set chart direction+scale
ax.set(xlim=(0,40), # for hitter's perspective (40,0) if pitcher's perspective
       ylim=(0,60), 
       aspect=1) # so that each increment of X and Z are visually the same

### Add a strikezone
# Replace these values with individual's values if comparing hitters
sz_top = 42 # 42 = 3.5ft
sz_bot = 18 # 18 = 1.5ft

# Outer Strikezone
ax.axhline(18, xmin=10/40, xmax=30/40, color='black', linewidth=2)
ax.axhline(42, xmin=10/40, xmax=30/40, color='black', linewidth=2)
ax.axvline(10, ymin=sz_bot/60, ymax=sz_top/60, color='black', linewidth=2)
ax.axvline(30, ymin=sz_bot/60, ymax=sz_top/60, color='black', linewidth=2)

# Inner Strikezone
ax.axhline(26, xmin=10/40, xmax=30/40, color='black', linewidth=1)
ax.axhline(34, xmin=10/40, xmax=30/40, color='black', linewidth=1)
ax.axvline(10+20/3, ymin=sz_bot/60, ymax=sz_top/60, color='black', linewidth=1)
ax.axvline(30-20/3, ymin=sz_bot/60, ymax=sz_top/60, color='black', linewidth=1)

# Plate (assuming hitter perspective)
# Yes, this plate is "above the ground", but it's for reference NOT measurement
ax.plot([11.73,27.23], [3.1,3.1], color='k', linewidth=1) # Front of plate
ax.plot([11.75,11.5], [3,2], color='k', linewidth=1) # Side of plate
ax.plot([27.25,27.5], [3,2], color='k', linewidth=1) # Side of plate
ax.plot([27.3,20], [2,1], color='k', linewidth=1) # Back of plate
ax.plot([11.7,20], [2,1], color='k', linewidth=1) # Back of plate

ax.axis('off') # Remove axes and labels (I prefer this look)
sns.despine()
