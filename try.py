#%%
import numpy as np
import pickle
import pandas as pd
#%%
with open('./store/results/dict-CIFAR100-N10-max500--C3-5x16-bn_F-1024x2000x2000_c100--i5000-lr0.0001-b256--EWC10000.0-1000-1.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
# %%
file_to_process = "./store/results/dict-spoken_digit-N6--C1-5x16-bn_F-1024x2000x2000_c60--i1000-lr0.0001-b256-R"
#slots = range(1,11)
shifts = range(10)

for shift in shifts:
    multitask_df = pd.DataFrame()
    df_single_task = pd.DataFrame()
    shft = []
    base_task = []
    task = []
    accuracy = []

    if shift == 0:
        filename = file_to_process +'.pkl'
    else:
        filename = file_to_process +'-s' + str(shift) +'.pkl'

    with open(filename, 'rb') as f:
        data = pickle.load(f)['R']

    for ii in range(10):
        for jj in range(ii+1):
            shft.append(shift)
            base_task.append(ii+1)
            task.append(jj+1)
            accuracy.append(data['task {}'.format(jj+1)].iloc[ii+1])

    multitask_df['data_fold'] = shft
    multitask_df['task'] = task
    multitask_df['base_task'] = base_task
    multitask_df['accuracy'] = accuracy
        
    df_single_task['task'] = range(1, 11)
    df_single_task['data_fold'] = shift
    df_single_task['accuracy'] = list(data.iloc[11])

    summary = (multitask_df,df_single_task)
    with open('./reformed_res/None-{}.pickle'.format(shift), 'wb') as f:
        pickle.dump(summary, f)
# %%
