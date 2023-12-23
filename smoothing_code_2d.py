import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.nonparametric.kernel_regression import KernelReg

df_size = 1000

test_data = {
  'x':np.random.normal(0,3,df_size).astype('int'),
  'z':np.random.normal(30,4,df_size).astype('int'),
  'test_stat':np.random.rand(df_size)
}
test_df = pd.DataFrame(test_data)

bandwidth = 1

kernel_regression = KernelReg(endog=test_df['test_stat'], 
                              exog= [test_df['x'], test_df['z']], 
                              bw=[bandwidth,bandwidth],
                              var_type='cc')

test_df['smoothed_stat'] = kernel_regression.fit([test_df['x'], test_df['z']])[0]
heatmap_df = test_df.pivot_table(columns='x',index='z',values=['smoothed_stat'], aggfunc='mean')

sns.heatmap(data=heatmap_df['smoothed_stat'].astype('float'))
sns.despine()
