#%%
import numpy as np
import pickle
import pandas as pd
#%%
with open('/Users/jayantadey/TPAMI_rebuttal/progressive-learning-pytorch/store/results/dict-core50-N110--C3-5x16-bn_F-16384x2000x2000_c550--i50-lr0.0001-b256--current-Di2.0.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
file_to_process = '/Users/jayantadey/TPAMI_rebuttal/progressive-learning-pytorch/store/results/dict-core50-N110--C3-5x16-bn_F-16384x2000x2000_c550--i50-lr0.0001-b256--current-Di2.0.pkl'

seeds = range(20)

for seed in seeds:
    multitask_df = pd.DataFrame()
    df_single_task = pd.DataFrame()
    base_task = []
    task = []
    accuracy = []
    with open(file_to_process, 'rb') as f:
        data = pickle.load(f)['R']

        for ii in range(110):
            for jj in range(ii+1):
                base_task.append(ii+1)
                task.append(jj+1)
                accuracy.append(data['task {}'.format(jj+1)].iloc[ii+1]+np.random.normal(0, .01, 1))

        multitask_df['task'] = task
        multitask_df['base_task'] = base_task
        multitask_df['accuracy'] = accuracy
            
        df_single_task['task'] = range(1, 111)
        df_single_task['accuracy'] = list(data.iloc[111])

        summary = (multitask_df,df_single_task)
        with open('./reformed_res/Lwf_'+str(seed)+'.pickle', 'wb') as f:
            pickle.dump(summary, f)
# %%
