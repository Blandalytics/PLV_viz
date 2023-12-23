import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.nonparametric.kernel_regression import KernelReg

### Dataframe with 1000 observations, using normally distributed X/Z coords and random stat values
df_size = 1000

test_data = {
  'x':np.random.normal(0,3,df_size).astype('int'),
  'z':np.random.normal(30,4,df_size).astype('int'),
  'test_stat':np.random.rand(df_size)
}
test_df = pd.DataFrame(test_data)

### Smooth the stat values
# Specify the bandwidth.
# I eyeballed the value for my stats, but I'm sure there's plenty of literature on it
bandwidth = 1

# Train the kernel regression model on the location and stat values, using the provided bandwidth
kernel_regression = KernelReg(endog=test_df['test_stat'], 
                              exog= [test_df['x'], test_df['z']], 
                              bw=[bandwidth,bandwidth],
                              var_type='cc') # States that X & Z are both continuous (c)

# Apply the model onto the X/Z coords
test_df['smoothed_stat'] = kernel_regression.fit([test_df['x'], test_df['z']])[0]

### Generate viz
# Pivot df to 2D, to better play with Seaborn's heatmap code 
heatmap_df = test_df.pivot_table(columns='x',index='z',values=['smoothed_stat'], aggfunc='mean')

# Generate Chart
sns.heatmap(data=heatmap_df['smoothed_stat'].astype('float'))
sns.despine()
