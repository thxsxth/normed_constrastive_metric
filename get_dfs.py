import torch
import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as weight_init
import scipy

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_and_process_df():
    df = pd.read_csv('sepsis3-cohort/RL_new.csv')
    df = df.iloc[:, 1:]
    td = pd.read_csv('sepsis3-cohort/time_to_deaths.csv')
    metric = pd.read_csv('Metric.csv')
    df['Death'] = (td.time_to_deaths == 1).values.astype('int')
    df['Fluids'] = df['Fluids'].apply(lambda x: max(x, 0))
    df['Metric'] = metric.metric
    df['Metric_ma'] = metric.metric_ma
    # td=pd.read_csv('time_to_deaths.csv')
    df['times_to_death'] = td.time_to_deaths

    #Terminal or Not
    df['Terminal'] = (1-(df.Pat == df.Pat.shift(-1)).astype('int'))

    df['Release'] = ((df.times_to_death == 0) & (df.Terminal == 1)).astype('int')
    df['Death'] = ((df.times_to_death == 1) & (df.Terminal == 1)).astype('int')


    temp = df.describe()
    means = temp.loc['mean'].values
    stds = temp.loc['std'].values

 
    quants = df.iloc[:, :27].quantile(0.9, axis=0)
    df.iloc[:, :27] = np.where(df.iloc[:, :27].values <
                           quants.values, df.iloc[:, :27].values, quants.values)

    cols = list(np.arange(39))+[40]

    df.iloc[:, cols] = (df.iloc[:, cols]-means[cols])/stds[cols]
    print('Normalized')

    get_actions(df)

    return df

def get_actions(df):
    df['Fluids'] = df['Fluids'].apply(lambda x: max(x, 0))
    Vaso_cuts, vaso_bins = pd.cut(
        df['Vaso'], bins=[-1e-6, 1e-8/2, 0.15, np.inf], labels=False, retbins=True)

    df['Vaso_cuts'] = Vaso_cuts.values
    Fluids_cuts, fluid_bins = pd.cut(
        df['Fluids'], bins=[-1e-6, 6.000000e-03/2, 5.000000e+01, np.inf], labels=False, retbins=True)

    df['Fluid_cuts'] = Fluids_cuts.values
    df['Fluid_cuts'].value_counts(
        normalize=True), df['Vaso_cuts'].value_counts(normalize=True)

    all_acts = np.arange(9).reshape(3, 3)

    df['action'] = all_acts[df.Vaso_cuts, df.Fluid_cuts]



