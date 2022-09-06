#%%
import numpy as np
import pickle
import pandas as pd
#%%
with open('./store/results/dict-dataset5-N20--C3-5x16-bn_F-9216x2000x2000_c100--i100-lr0.0001-b256--current-Di2.0.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
#file_to_process = '/Users/jayantadey/progressive-learning-pytorch/store/results/dict-dataset5-N5--C3-5x16-bn_F-1024x2000x2000_c50--i1000-lr0.0001-b256--EWC10000.0-1000.pkl'


multitask_df = pd.DataFrame()
df_single_task = pd.DataFrame()
shft = []
base_task = []
task = []
accuracy = []

filename = '/Users/jayantadey/progressive-learning-pytorch/store/results/dict-dataset5-N20--C3-5x16-bn_F-9216x2000x2000_c100--i100-lr0.0001-b256.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)['R']

for ii in range(20):
    for jj in range(ii+1):
        base_task.append(ii+1)
        task.append(jj+1)
        accuracy.append(data['task {}'.format(jj+1)].iloc[ii+1])

multitask_df['task'] = task
multitask_df['base_task'] = base_task
multitask_df['accuracy'] = accuracy

df_single_task['task'] = range(1, 21)
df_single_task['accuracy'] = list(data.iloc[21])

summary = (multitask_df,df_single_task)
with open('./reformed_res/imagenet/None.pickle', 'wb') as f:
    pickle.dump(summary, f)
# %%
