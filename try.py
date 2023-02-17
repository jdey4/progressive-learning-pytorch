#%%
import numpy as np
import pickle
import pandas as pd
#%%
with open('store/results/dict-food1k-N50--C3-5x16-bn_F-4096x2000x2000_c1000--i500-lr0.0001-b256--current-Di2.0.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
file_to_process = "store/results/dict-food1k-N50--C3-5x16-bn_F-4096x2000x2000_c1000--i500-lr0.0001-b256--current-Di2.0.pkl"

multitask_df = pd.DataFrame()
df_single_task = pd.DataFrame()
base_task = []
task = []
accuracy = []

with open(file_to_process, 'rb') as f:
    data = pickle.load(f)['R']

    for ii in range(50):
        for jj in range(ii+1):
            base_task.append(ii+1)
            task.append(jj+1)
            accuracy.append(data['task {}'.format(jj+1)].iloc[ii+1])

    multitask_df['task'] = task
    multitask_df['base_task'] = base_task
    multitask_df['accuracy'] = accuracy
        
    df_single_task['task'] = range(1, 51)
    df_single_task['accuracy'] = list(data.iloc[51])

    summary = (multitask_df,df_single_task)
    with open('./reformed_res/None.pickle', 'wb') as f:
        pickle.dump(summary, f)
# %%
