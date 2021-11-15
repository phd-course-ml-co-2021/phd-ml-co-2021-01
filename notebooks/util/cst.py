#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import datasets
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as k
import tensorflow_lattice as tfl

figsize=(9,3)

# ==============================================================================
# Loading data
# ==============================================================================

def load_cifar_data():
    (xtr, ytr), (xts, yts) = datasets.cifar100.load_data(label_mode="fine")
    (_, ytr_c), (_, yts_c) = datasets.cifar100.load_data(label_mode="coarse")
    return (xtr, ytr, ytr_c), (xts, yts, yts_c)


def load_communities_data(data_folder, nan_discard_thr=0.05):
    # Read the raw data
    fname = os.path.join(data_folder, 'CommViolPredUnnormalizedData.csv')
    data = pd.read_csv(fname, sep=';', na_values='?')
    # Discard columns
    dcols = list(data.columns[-18:-2]) # directly crime related
    dcols = dcols + list(data.columns[7:12]) # race related
    dcols = dcols + ['nonViolPerPop']
    data = data.drop(columns=dcols)
    # Use relative values
    for aname in data.columns:
        if aname.startswith('pct'):
            data[aname] = data[aname] / 100
        elif aname in ('numForeignBorn', 'persEmergShelt',
                       'persHomeless', 'officDrugUnits',
                       'policCarsAvail', 'policOperBudget', 'houseVacant'):
            data[aname] = data[aname] / (data['pop'] / 100e3)
    # Remove redundant column (a relative columns is already there)
    data = data.drop(columns=['persUrban', 'numPolice',
                              'policeField', 'policeCalls', 'gangUnit'])
    # Discard columns with too many NaN values
    thr = nan_discard_thr * len(data)
    cols = data.columns[data.isnull().sum(axis=0) >= thr]
    cols = [c for c in cols if c != 'violentPerPop']
    data = data.drop(columns=cols)
    # Remove all NaN values
    data = data.dropna()
    # Shuffle
    rng = np.random.default_rng(42)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    return data.iloc[idx]


# def load_building_data(data_folder, noise=0):
#     data = pd.read_csv(f'{data_folder}/ENB2012_data.csv')
#     data.columns = [
#             'Relative Compactness',
#             'Surface Area',
#             'Wall Area',
#             'Roof Area',
#             'Overall Height',
#             'Orientation',
#             'Glazing Area',
#             'Glazing Area Distribution',
#             'Heating Load',
#             'Cooling Load'
#             ]
#     # Inject noise
#     nf = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
#           'Overall Height', 'Glazing Area', 'Heating Load', 'Cooling Load']
#     cf = ['Orientation', 'Glazing Area Distribution']
#     np.random.seed(42)
#     for cname in nf:
#         sigma = np.random.randn(len(data)) * noise
#         data[cname] = data[cname] * np.exp(sigma)
#     return data, nf, cf


def click_through_rate(avg_ratings, num_reviews, dollar_ratings):
    dollar_rating_baseline = {"D": 3, "DD": 2, "DDD": 4, "DDDD": 4.5}
    return 1 / (1 + np.exp(
        np.array([dollar_rating_baseline[d] for d in dollar_ratings]) -
        avg_ratings * np.log1p(num_reviews) / 4))


def load_restaurant_data():
    def sample_restaurants(n):
        avg_ratings = np.random.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(np.random.uniform(0.0, np.log(200), n)))
        dollar_ratings = np.random.choice(["D", "DD", "DDD", "DDDD"], n)
        ctr_labels = click_through_rate(avg_ratings, num_reviews, dollar_ratings)
        return avg_ratings, num_reviews, dollar_ratings, ctr_labels


    def sample_dataset(n, testing_set):
        (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = sample_restaurants(n)
        if testing_set:
            # Testing has a more uniform distribution over all restaurants.
            num_views = np.random.poisson(lam=3, size=n)
        else:
            # Training/validation datasets have more views on popular restaurants.
            num_views = np.random.poisson(lam=ctr_labels * num_reviews / 40.0, size=n)

        return pd.DataFrame({
                "avg_rating": np.repeat(avg_ratings, num_views),
                "num_reviews": np.repeat(num_reviews, num_views),
                "dollar_rating": np.repeat(dollar_ratings, num_views),
                "clicked": np.random.binomial(n=1, p=np.repeat(ctr_labels, num_views))
            })

    # Generate
    np.random.seed(42)
    data_train = sample_dataset(2000, testing_set=False)
    data_val = sample_dataset(1000, testing_set=False)
    data_test = sample_dataset(1000, testing_set=True)
    return data_train, data_val, data_test



def load_cmapss_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data


# ==============================================================================
# Plotting
# ==============================================================================


def plot_ctr_estimation(estimator, scale,
        split_input=False, one_hot_categorical=True,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    nrev = np.tile(np.linspace(0, 200, res), res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        if one_hot_categorical:
            # Categorical encoding for the dollar rating
            dr_cat = np.zeros((1, 4))
            dr_cat[0, i] = 1
            dr_cat = np.repeat((dr_cat), res*res, axis=0)
            # Concatenate all inputs
            x = np.hstack((avgr, nrev, dr_cat))
        else:
            # Integer encoding for the categorical attribute
            dr_cat = np.full((res*res, 1), i)
            x = np.hstack((avgr, nrev, dr_cat))
        # Split input, if requested
        if split_input:
            x = [x[:, i] for i in range(x.shape[1])]
        # Obtain the predictions
        ctr = estimator.predict(x)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('avrage rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_calibration(calibrators, scale, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3

    # Average rating calibration
    avgr = np.linspace(0, 5, res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    avgr_cal = calibrators[0].predict(avgr)
    plt.subplot(131)
    plt.plot(avgr, avgr_cal)
    plt.xlabel('avg_rating')
    plt.ylabel('cal. output')
    # Num. review calibration
    nrev = np.linspace(0, 200, res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    nrev_cal = calibrators[1].predict(nrev)
    plt.subplot(132)
    plt.plot(nrev, nrev_cal)
    plt.xlabel('num_reviews')
    # Dollar rating calibration
    drating = np.arange(0, 4).reshape(-1, 1)
    drating_cal = calibrators[2].predict(drating).ravel()
    plt.subplot(133)
    xticks = np.linspace(0.5, 3.5, 4)
    plt.bar(xticks, drating_cal)
    plt.xticks(xticks, ['D', 'DD', 'DDD', 'DDDD'])

    plt.tight_layout()



def plot_ctr_truth(figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res)
    nrev = np.tile(np.linspace(0, 200, res), res)
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        drt = [drating] * (res*res)
        ctr = click_through_rate(avgr, nrev, drt)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('avrage rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_distribution(data, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    nbins = 15
    plt.subplot(131)
    plt.hist(data['avg_rating'], density=True, bins=nbins)
    plt.xlabel('average rating')
    plt.subplot(132)
    plt.hist(data['num_reviews'], density=True, bins=nbins)
    plt.xlabel('num. reviews')
    plt.subplot(133)
    vcnt = data['dollar_rating'].value_counts()
    vcnt /= vcnt.sum()
    plt.bar([0.5, 1.5, 2.5, 3.5],
            [vcnt['D'], vcnt['DD'], vcnt['DDD'], vcnt['DDDD']])
    plt.xlabel('dollar rating')
    plt.tight_layout()


def plot_training_history(history, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.tight_layout()


def plot_pred_scatter(y_pred, y_true, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=0.1)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.tight_layout()



def plot_ce(data, xname, yname, nn, vmin=-1, vmax=1,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    x = np.linspace(vmin, vmax, len(data)).astype(np.float32)
    tmp = data.copy()
    tmp[xname] = x
    y = nn.predict(tmp)
    plt.scatter(x, y, marker='x', s=5)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.tight_layout()


def plot_pred_by_protected(data, pred, protected,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    # Prepare the data for the boxplot
    x, lbls = [], []
    # Append the baseline
    pred = pred.ravel()
    x.append(pred)
    lbls.append('all')
    # Append the sub-datasets
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            x.append(pred[mask])
            lbls.append(f'{aname}={val}')
    plt.boxplot(x, labels=lbls)
    plt.tight_layout()


def plot_lr_weights(weights, attributes, cap_num=None,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    # Sort attributes by decreasing absolute weights
    idx = np.argsort(np.abs(weights))[::-1]
    if cap_num is not None:
        idx = idx[:cap_num]
    fontsize = min(8, 300 / len(idx))
    x = np.linspace(0.5, 0.5+len(idx), len(idx))
    plt.bar(x, weights[idx])
    plt.xticks(x, labels=attributes[idx], rotation=45, fontsize=fontsize)
    plt.tight_layout()


def print_crime_merics(model, tr, ts, attributes, target):
    tr_pred = model.predict(tr[attributes])
    r2_tr = r2_score(tr[target], tr_pred)
    mae_tr = mean_absolute_error(tr[target], tr_pred)

    ts_pred = model.predict(ts[attributes])
    r2_ts = r2_score(ts[target], ts_pred)
    mae_ts = mean_absolute_error(ts[target], ts_pred)

    print(f'R2 score: {r2_tr:.2f} (training), {r2_ts:.2f} (test)')
    print(f'MAE: {mae_tr:.2f} (training), {mae_ts:.2f} (test)')


# ==============================================================================
# Data manipulation
# ==============================================================================

def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    if len(ts_list) > 0:
        ts_data = pd.concat(ts_list)
    else:
        ts_data = pd.DataFrame(columns=tr_data.columns)
    return tr_data, ts_data


# ==============================================================================
# Models and optimization
# ==============================================================================

class MLPRegressor(keras.Model):
    def __init__(self, input_shape, hidden=[]):
        super(MLPRegressor, self).__init__()
        # Build the model
        self.lrs = [layers.Dense(h, activation='relu') for h in hidden]
        self.lrs.append(layers.Dense(1, activation='linear'))

    def call(self, data):
        x = data
        for layer in self.lrs:
            x = layer(x)
        return x


class MLPClassifier(keras.Model):
    def __init__(self, input_shape, hidden=[]):
        super(MLPClassifier, self).__init__()
        # Build the model
        self.lrs = [layers.Dense(h, activation='relu') for h in hidden]
        self.lrs.append(layers.Dense(1, activation='sigmoid'))

    def call(self, data):
        x = data
        for layer in self.lrs:
            x = layer(x)
        return x


class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval

    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=['machine', 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for mcn, gtmp in tmp.groupby('machine'):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack


def opt_threshold_and_plot(machine, pred, th_range, cmodel,
        plot=True, figsize=figsize, autoclose=True):
    # Compute the optimal threshold
    costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
    opt_th = th_range[np.argmin(costs)]
    # Plot
    if plot:
        if autoclose:
            plt.close('all')
        plt.figure(figsize=figsize)
        plt.plot(th_range, costs)
        plt.tight_layout()
    # Return the threshold
    return opt_th



class CstBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, in_cols, batch_size, seed=42):
        super(CstBatchGenerator).__init__()
        self.data = data
        self.in_cols = in_cols
        self.dpm = split_by_field(data, 'machine')
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        # Build the first sequence of batches
        self.__build_batches()

    def __len__(self):
        return len(self.batches)

    # def __getitem__(self, index):
    #     idx = self.batches[index]
    #     mcn = self.machines[index]
    #     x = self.data[self.in_cols].loc[idx].values
    #     y = self.data['rul'].loc[idx].values
    #     return x, y


    def __getitem__(self, index):
        idx = self.batches[index]
        # mcn = self.machines[index]
        x = self.data[self.in_cols].loc[idx].values
        y = self.data['rul'].loc[idx].values
        flags = (y != -1)
        info = np.vstack((y, flags, idx)).T
        return x, info

    def on_epoch_end(self):
        self.__build_batches()

    def __build_batches(self):
        self.batches = []
        self.machines = []
        # Randomly sort the machines
        # self.rng.shuffle(mcns)
        # Loop over all machines
        mcns = list(self.dpm.keys())
        for mcn in mcns:
            # Obtain the list of indices
            index = self.dpm[mcn].index
            # Padding
            padsize = self.batch_size - (len(index) % self.batch_size)
            padding = self.rng.choice(index, padsize)
            idx = np.hstack((index, padding))
            # Shuffle
            self.rng.shuffle(idx)
            # Split into batches
            bt = idx.reshape(-1, self.batch_size)
            # Sort each batch individually
            bt = np.sort(bt, axis=1)
            # Store
            self.batches.append(bt)
            self.machines.append(np.repeat([mcn], len(bt)))
        # Concatenate all batches
        self.batches = np.vstack(self.batches)
        self.machines = np.hstack(self.machines)
        # Shuffle the batches
        bidx = np.arange(len(self.batches))
        self.rng.shuffle(bidx)
        self.batches = self.batches[bidx, :]
        self.machines = self.machines[bidx]



# class CstSMBatchGenerator(SMBatchGenerator):
#     def __init__(self, data, in_cols, batch_size, seed=42):
#         super(CstSMBatchGenerator, self).__init__(data, in_cols,
#                 batch_size, seed)

#     def __getitem__(self, index):
#         idx = self.batches[index]
#         mcn = self.machines[index]
#         x = self.data[self.in_cols].loc[idx].values
#         y = self.data['rul'].loc[idx].values
#         flags = (y != -1)
#         info = np.vstack((y, flags, idx)).T
#         return x, info


# class CstRULRegressor(MLPRegressor):
#     def __init__(self, input_shape, alpha, beta, hidden=[]):
#         super(CstRULRegressor, self).__init__(input_shape, hidden)
#         self.alpha = alpha
#         self.beta = beta
#         self.ls_tracker = keras.metrics.Mean(name='loss')
#         self.mse_tracker = keras.metrics.Mean(name='mse')
#         self.cst_tracker = keras.metrics.Mean(name='cst')

#     def train_step(self, data):
#         x, info = data
#         y_true = info[:, 0:1]
#         flags = info[:, 1:2]
#         idx = info[:, 2:3]

#         with tf.GradientTape() as tape:
#             # Obtain the predictions
#             y_pred = self(x, training=True)
#             # Compute the main loss
#             mse = k.mean(flags * k.square(y_pred-y_true))
#             # Compute the constraint regularization term
#             delta_pred = y_pred[1:] - y_pred[:-1]
#             delta_rul = -(idx[1:] - idx[:-1])
#             deltadiff = delta_pred - delta_rul
#             cst = k.mean(k.square(deltadiff))
#             loss = self.alpha * mse + self.beta * cst

#         # Compute gradients
#         tr_vars = self.trainable_variables
#         grads = tape.gradient(loss, tr_vars)

#         # Update the network weights
#         self.optimizer.apply_gradients(zip(grads, tr_vars))

#         # Track the loss change
#         self.ls_tracker.update_state(loss)
#         self.mse_tracker.update_state(mse)
#         self.cst_tracker.update_state(cst)
#         return {'loss': self.ls_tracker.result(),
#                 'mse': self.mse_tracker.result(),
#                 'cst': self.cst_tracker.result()}

#     @property
#     def metrics(self):
#         return [self.ls_tracker,
#                 self.mse_tracker,
#                 self.cst_tracker]


class CstRULRegressor(MLPRegressor):
    def __init__(self, input_shape, alpha, beta, maxrul, hidden=[]):
        super(CstRULRegressor, self).__init__(input_shape, hidden)
        # Weights
        self.alpha = alpha
        self.beta = beta
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def train_step(self, data):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        with tf.GradientTape() as tape:
            # Obtain the predictions
            y_pred = self(x, training=True)
            # Compute the main loss
            mse = k.mean(flags * k.square(y_pred-y_true))
            # Compute the constraint regularization term
            delta_pred = y_pred[1:] - y_pred[:-1]
            delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
            deltadiff = delta_pred - delta_rul
            cst = k.mean(k.square(deltadiff))
            loss = self.alpha * mse + self.beta * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


def DIDI_r(data, pred, protected):
    res = 0
    avg = np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            res += abs(avg - np.mean(pred[mask]))
    return res


class CstDIDIRegressor(MLPRegressor):
    def __init__(self, attributes, protected, alpha, thr, hidden=[]):
        super(CstDIDIRegressor, self).__init__(len(attributes), hidden)
        # Weight and threshold
        self.alpha = alpha
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mse = self.compiled_loss(y_true, y_pred)
            # Compute the constraint regularization term
            ymean = k.mean(y_pred)
            didi = 0
            for aidx, dom in self.protected.items():
                for val in dom:
                    mask = (x[:, aidx] == val)
                    didi += k.abs(ymean - k.mean(y_pred[mask]))
            cst = k.maximum(0.0, didi - self.thr)
            loss = mse + self.alpha * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]



class LagDualDIDIRegressor(MLPRegressor):
    def __init__(self, attributes, protected, thr, hidden=[]):
        super(LagDualDIDIRegressor, self).__init__(len(attributes), hidden)
        # Weight and threshold
        self.alpha = tf.Variable(0., name='alpha')
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')


    def __custom_loss(self, x, y_true, sign=1):
        y_pred = self(x, training=True)
        # loss, mse, cst = self.__custom_loss(x, y_true, y_pred)
        mse = self.compiled_loss(y_true, y_pred)
        # Compute the constraint regularization term
        ymean = k.mean(y_pred)
        didi = 0
        for aidx, dom in self.protected.items():
            for val in dom:
                mask = (x[:, aidx] == val)
                didi += k.abs(ymean - k.mean(y_pred[mask]))
        cst = k.maximum(0.0, didi - self.thr)
        loss = mse + self.alpha * cst
        return sign*loss, mse, cst

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=1)

        # Separate training variables
        tr_vars = self.trainable_variables
        wgt_vars = tr_vars[:-1]
        mul_vars = tr_vars[-1:]

        # Update the network weights
        grads = tape.gradient(loss, wgt_vars)
        self.optimizer.apply_gradients(zip(grads, wgt_vars))

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=-1)

        grads = tape.gradient(loss, mul_vars)
        self.optimizer.apply_gradients(zip(grads, mul_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]
