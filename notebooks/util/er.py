#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as k

from ortools.sat.python import cp_model

import collections
Task = collections.namedtuple('task', 'start end interval ttype')

import copy
from ortools.linear_solver import pywraplp
from scipy import sparse

status_string = {
    cp_model.OPTIMAL: 'optimal',
    cp_model.FEASIBLE: 'feasible',
    cp_model.INFEASIBLE: 'infeasible',
    cp_model.MODEL_INVALID: 'invalid',
    cp_model.UNKNOWN: 'unknown'
}

# Configuration
figsize = (9, 3)

def load_raw_data(data_folder):
    # Read the CSV files
    fnames = ['2018', '2019']
    datalist = []
    for fstem in fnames:
        # Read data and append
        data = pd.read_csv(f'{data_folder}/{fstem}.csv', sep='|')
        data['Anno'] = int(fstem)
        data = data[['Anno'] + data.columns[:-1].tolist()]
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    return data

def translate_data(data_folder):
    # Read the CSV files
    fnames = ['2018', '2019']
    datalist = []
    for fstem in fnames:
        # Read data and append
        data = pd.read_csv(f'{data_folder}/{fstem}.csv', sep='|')
        data['Anno'] = int(fstem)
        data = data[['Anno'] + data.columns[:-1].tolist()]
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Translate the "CODICE" entry
    data['CODICE'].replace('VERDE', 'green', inplace=True)
    data['CODICE'].replace('BIANCO', 'white', inplace=True)
    data['CODICE'].replace('ROSSO', 'red', inplace=True)
    data['CODICE'].replace('GIALLO', 'yellow', inplace=True)
    # Translate the "CODICE" entry
    data['Esito'].replace('Ammessi', 'admitted', inplace=True)
    data['Esito'].replace('Abbandoni', 'abandoned', inplace=True)
    # Translate the "path" entry
    data['Percorso'] = data['Percorso'].str.replace('"Visita"', 'visit')
    data['Percorso'] = data['Percorso'].str.replace('"LABORATORIO"', 'lab')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA ORTOPEDICA"', 'orthopaedic visit')
    data['Percorso'] = data['Percorso'].str.replace('"Prescrizione Terapia"', 'prescription')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA OTORINOLARINGOIATRICA"', 'otolaryngological visit')
    data['Percorso'] = data['Percorso'].str.replace('"ECOGRAFIA"', 'ultrasound')
    data['Percorso'] = data['Percorso'].str.replace('"TAC"', 'CT scan')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA NEUROLOGICA"', 'neurological visit')
    data['Percorso'] = data['Percorso'].str.replace('"CONSULENZA CHIRURGICA VASCOLARE"', 'vascular surgery consultation')
    data['Percorso'] = data['Percorso'].str.replace('"CORONAROGRAFIA"', 'coronarography')
    data['Percorso'] = data['Percorso'].str.replace('"ESAME COMPLESSIVO DELL\'OCCHIO"', 'eye examination')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA CHIRURGICA"', 'surgical visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA NEUROPSICHIATRICA INFANTILE"', 'child psychiatrist visit')
    data['Percorso'] = data['Percorso'].str.replace('"ORTOPANTOMOGRAFIA DIGITALE"', 'digital orthopantomography')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA UROLOGICA"', 'urological visit')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CAROTIDE COMUNE SN."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CAROTIDE INTERNA SN"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CEREBRALE PRIMO  VASO"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"TROMBOLISI"', 'thrombolysis')
    data['Percorso'] = data['Percorso'].str.replace('"ARTERIOGRAFIA SELETTIVA"', 'arteriography')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA DIABETOLOGICA"', 'diabetological visit')
    data['Percorso'] = data['Percorso'].str.replace('"ESECUZIONE CD / COPIE CD"', 'CD execution and copy')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CAROTIDE COMUNE DS."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CAROTIDE INTERNA DS."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"PROVA DA SFORZO"', 'stress test')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA DERMATOLOGICA"', 'dermatological visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA CARDIOLOGICA"', 'cardiological visit')
    data['Percorso'] = data['Percorso'].str.replace('"INVIO/SPEDIZIONE ESAME AD ALTRO OSPEDALE"', 'sending results to a second hospital')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO \(VASI NON CODIFICATI\)"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"POSIZIONAMENTO DI CVC IN SAL"', 'CVC positioning')
    data['Percorso'] = data['Percorso'].str.replace('"ELETTROENCEFALOGRAMMA"', 'EEG')
    data['Percorso'] = data['Percorso'].str.replace('"SIGMOIDOSCOPIA CON ENDOSCOPIO FLESSIBILE"', 'sigmodoscopy')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA INFETTIVOLOGICA"', 'infectious disease visit')
    data['Percorso'] = data['Percorso'].str.replace('"EMBOLIZZAZIONE"', 'embolization')
    data['Percorso'] = data['Percorso'].str.replace('"EGDS CON BIOPSIA"', 'EGDS')
    data['Percorso'] = data['Percorso'].str.replace('"PEG - POSIZIONAMENTO"', 'PEG')
    data['Percorso'] = data['Percorso'].str.replace('"PEG - RIMOZIOE"', 'PEG')
    data['Percorso'] = data['Percorso'].str.replace('"2\^ VISITA NEUROLOGICA"', 'neurological visit')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO  ARCO E VASI EPIAORTICI"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CAROTIDE ESTERNA DS."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO CAROTIDE ESTERNA SN."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO PERCUTANEA \(PER VASO\)"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO VERTEBRALE DS."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO VERTEBRALE SN."', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO-RM CEREBRALE ARTERIOSA"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIO-RM CIRCOLO ARTERIOSO INTRA-CRANICO"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOGRAFIA RICOSTRUZIONI"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOGRAFIA ROTATORIA"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOPLASTICA PERCUTANEA CAROTIDI"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOPLASTICA PERCUTANEA CORONARICA MULTIVASALE"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOPLASTICA PERCUTANEA ILIACA"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOPLASTICA PERCUTANEA"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ANGIOPLASTICA PERCUTANEA"', 'angiography')
    data['Percorso'] = data['Percorso'].str.replace('"ARCHIVIAZIONE DIGITALE PRECEDENTI"', 'medical record digitalization')
    data['Percorso'] = data['Percorso'].str.replace('"ARTERIOGRAFIA ARTO INF. DX"', 'arteriography')
    data['Percorso'] = data['Percorso'].str.replace('"ARTERIOGRAFIA ARTO INF. SX"', 'arteriography')
    data['Percorso'] = data['Percorso'].str.replace('"ARTERIOGRAFIA SELETTIVA ARTERIE BRONCHIALI"', 'arteriography')
    data['Percorso'] = data['Percorso'].str.replace('"ARTERIOGRAFIA SELETTIVA RENALE"', 'arteriography')
    data['Percorso'] = data['Percorso'].str.replace('"CISTOGRAFIA"', 'cystography')
    data['Percorso'] = data['Percorso'].str.replace('"COLLOQUIO PSICHIATRICO"', 'psychiatrict counseling')
    data['Percorso'] = data['Percorso'].str.replace('"COLONSCOPIA"', 'colonscopy')
    data['Percorso'] = data['Percorso'].str.replace('"CONSULENZA NEURORADIOLOGICA"', 'neuroradiological consultation')
    data['Percorso'] = data['Percorso'].str.replace('"CONSULENZA RADIOLOGICA"', 'radiological consultation')
    data['Percorso'] = data['Percorso'].str.replace('"CONSULENZA RIANIMATORIA"', 'IC consultation')
    data['Percorso'] = data['Percorso'].str.replace('"CURA ODONTOIATRICA \(BERETTA\)"', 'dental care')
    data['Percorso'] = data['Percorso'].str.replace('"DRENAGGIO BILIARE ESTERNO"', 'biliary drainage')
    data['Percorso'] = data['Percorso'].str.replace('"ELETTROENCEFALOGRAMMA CON VIDEOREGISTRAZIONE"', 'EEG')
    data['Percorso'] = data['Percorso'].str.replace('"EMBOLIZZAZIONE CEREB. \(SPIRALI MAX.N.3\)"', 'embolization')
    data['Percorso'] = data['Percorso'].str.replace('"EMBOLIZZAZIONE COLLO-MASS. FACC. \(COLLE O PART.\)"', 'embolization')
    data['Percorso'] = data['Percorso'].str.replace('"EMBOLIZZAZIONE COLLO-MASS. FACC. \(SPIRALI\)"', 'embolization')
    data['Percorso'] = data['Percorso'].str.replace('"ESAME DEL FUNDUS OCULI"', 'eye examination')
    data['Percorso'] = data['Percorso'].str.replace('"GATED SPECT"', 'gated spect')
    data['Percorso'] = data['Percorso'].str.replace('"NEFROPIELOSTOMIA PERCUTANEA"', 'nephrolytotomy')
    data['Percorso'] = data['Percorso'].str.replace('"PEG - RIMOZIONE"', 'PEG')
    data['Percorso'] = data['Percorso'].str.replace('"PIELOGRAFIA TRANSPIELOSTOMICA DX"', 'pyelography')
    data['Percorso'] = data['Percorso'].str.replace('"PIELOGRAFIA TRANSPIELOSTOMICA SX"', 'pyelography')
    data['Percorso'] = data['Percorso'].str.replace('"POSIZIONAMENTO DI CVC IN SALA"', 'CVC positioning')
    data['Percorso'] = data['Percorso'].str.replace('"POSIZIONAMENTO DI CVC"', 'CVC positioning')
    data['Percorso'] = data['Percorso'].str.replace('"RM ADDOME INFERIORE"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM ADDOME SUPERIORE"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM DEL BACINO SMDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM DIFFUSIONE CEREBRALE"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM ENCEFALO CON MDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM ENCEFALO SMDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM RACHIDE CERVICALE CON MDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM RACHIDE CERVICALE SMDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM RACHIDE DORSALE CON MDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM RACHIDE DORSALE SMDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM RACHIDE LOMBARE CON MDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"RM RACHIDE LOMBARE SMDC"', 'MRI')
    data['Percorso'] = data['Percorso'].str.replace('"SCOPIA INTRAOPERATORIA  60 MIN."', 'intraoperative scopy')
    data['Percorso'] = data['Percorso'].str.replace('"SPECT MIOCARDICA RIPOSO"', 'myocardial SPECT')
    data['Percorso'] = data['Percorso'].str.replace('"STENT BILIARE \(X PEZZO\)"', 'biliary stent')
    data['Percorso'] = data['Percorso'].str.replace('"STENT RENALI.F114"', 'renal stent')
    data['Percorso'] = data['Percorso'].str.replace('"STUDIO ETA\' OSSEA"', 'bone age estimation')
    data['Percorso'] = data['Percorso'].str.replace('"TROMBECTOMIA"', 'thrombectomy')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA ANESTESIOLOGICA"', 'anesthesiological visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA ANGIOLOGICA"', 'angiological visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA DIETOLOGICA"', 'diet visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA GASTROENTEROLOGICA"', 'gastroenterological visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA MAXILLO FACCIALE"', 'maxillofacial visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA ODONTOIATRICA"', 'dental visit')
    data['Percorso'] = data['Percorso'].str.replace('"VISITA PNEUMOLOGICA"', 'pneumological visit')
    data['Percorso'] = data['Percorso'].str.replace('"Triage"', 'triage')
    data['Percorso'] = data['Percorso'].str.replace('"RX"', 'RX')
    data['Percorso'] = data['Percorso'].str.replace('"EGDS"', 'EGDS')
    data['Percorso'] = data['Percorso'].str.replace(', ', ',')
    # Fix NaN flows
    data['Percorso'].fillna('[]', inplace=True)
    # # Split flow strings
    # data['Percorso'] = data['Percorso'].apply(lambda s: s[1:-1].split(',') if s != np.NaN else np.NaN)
    # Change the column names
    data.columns = ['year', 'ID', 'Triage', 'TkCharge', 'Code', 'Outcome', 'Flow']
    # Write back
    data.to_csv(f'{data_folder}/er.csv', sep=';', index=False)

def load_data(data_folder):
    # Read the CSV file
    data = pd.read_csv(f'{data_folder}/er.csv', sep=';', parse_dates=[2, 3])
    # Convert a few fields to categorical format
    data['Code'] = data['Code'].astype('category')
    data['Outcome'] = data['Outcome'].astype('category')
    # Sort by triage time
    data.sort_values(by='Triage', inplace=True)
    # Discard the firl
    return data


def plot_bars(data, figsize=figsize, autoclose=True, tick_gap=1,
        series=None):
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    # x = np.arange(len(data))
    # x = 0.5 + np.arange(len(data))
    # plt.bar(x, data, width=0.7)
    # x = data.index-0.5
    x = data.index
    plt.bar(x, data, width=0.7)
    # plt.bar(x, data, width=0.7)
    if series is not None:
        # plt.plot(series.index-0.5, series, color='tab:orange')
        plt.plot(series.index, series, color='tab:orange')
    if tick_gap > 0:
        plt.xticks(x[::tick_gap], data.index[::tick_gap], rotation=45)
    plt.tight_layout()


def plot_series(series, std=None,
        figsize=figsize,
        autoclose=True, s=0):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if std is not None:
        lb = series.values.ravel() - std.values.ravel()
        ub = series.values.ravel() + std.values.ravel()
        plt.fill_between(std.index, lb, ub, alpha=0.3, label='+/- std')
    if s > 0:
        plt.plot(series.index, series, label='data')
    else:
        plt.plot(series.index, series, label='data',
                marker='.', markersize=s)
    if std is not None:
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


def plot_training_history(history, 
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], label='val. loss')
        plt.legend()
    plt.tight_layout()


def sliding_window_2D(data, wlen, stride=1):
    # Get shifted tables
    m = len(data)
    lt = [data.iloc[i:m-wlen+i+1:stride, :].values for i in range(wlen)]
    # Reshape to add a new axis
    s = lt[0].shape
    for i in range(wlen):
        lt[i] = lt[i].reshape(s[0], 1, s[1])
    # Concatenate
    wdata = np.concatenate(lt, axis=1)
    return wdata


def plot_autocorrelation(data, max_lag=100, figsize=figsize):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(data)
    # Customized x limits
    plt.xlim(0, max_lag)
    # Rotated x ticks
    plt.xticks(rotation=45)
    plt.tight_layout()


def flow_to_levels(fstring):
    # Parse all the activity names
    activities = fstring[1:-1].split(',')
    # Partition into levels
    levels = [[]]
    for a in activities:
        if a == 'triage':
            pass
        elif a == 'visit':
            if len(levels[-1]) == 0:
                levels[-1].append(a)
            else:
                levels.append([a])
            levels.append([])
        else:
            levels[-1].append(a)
    return levels[:-1]


def get_dur(ttype):
    durs = {'visit': 1, 'ultrasound': 2, 'RX': 2}
    return durs[ttype] if ttype in durs else 4

def get_level_dur(levels):
    res = 0
    for lvl in levels:
        res += sum(get_dur(a) for a in lvl)
    return res

def get_horizon(levels, setups=None):
    eoh = sum(get_level_dur(l) for l in levels.values())
    return eoh


def levels_to_vars(mdl, levels, idx, tasks, last, rl, dl):
    # Build a variable for each activity
    for k, lvl in enumerate(levels):
        for i, a in enumerate(lvl):
            # Obtain duration and build variables
            dur = get_dur(a)
            start = mdl.NewIntVar(rl, dl, f's_{idx}_{k}_{i}')
            end = mdl.NewIntVar(rl, dl, f'e_{idx}_{k}_{i}')
            interval = mdl.NewIntervalVar(start, dur, end, f'i_{idx}_{k}_{i}')
            # Store
            tasks[(idx,k,i)] = Task(start, end, interval, a)
    last[idx] = Task(start, end, interval, a)


def add_all_vars(mdl, levels, rl, dl):
    tasks, last = {}, {}
    for idx, lvl in levels.items():
        levels_to_vars(mdl, lvl, idx, tasks, last, rl[idx], dl=dl[idx])
    return tasks, last


def build_levels(fdata):
    levels, codes_by_idx = {}, {}
    for idx in fdata.index:
        levels[idx] = flow_to_levels(fdata.loc[idx]['Flow'])
        codes_by_idx[idx] = fdata.loc[idx]['Code']
    return levels, codes_by_idx


def print_solution(slv, levels, tasks, codes_by_idx):
    # Collect tasks by idx
    tasks_by_idx = {idx: [] for idx in levels.keys()}
    for (idx, k, i), t in tasks.items():
        tasks_by_idx[idx].append(t)
    # Sort by increasing start time
    strings_by_idx = {}
    starts_by_idx = {}
    for idx in tasks_by_idx.keys():
        ttasks = sorted(tasks_by_idx[idx], key=lambda t: slv.Value(t.start))
        starts_by_idx[idx] = slv.Value(ttasks[0].start)
        stasks = [f'{t.ttype}({slv.Value(t.start)}-{slv.Value(t.end)})' for t in ttasks]
        stasks = ', '.join(stasks)
        strings_by_idx[idx] = f'{idx}({codes_by_idx[idx]}): {stasks}'
    # Sort idx by starting time
    sorted_idx = sorted([k for k in starts_by_idx], key=lambda k: starts_by_idx[k])
    for idx in sorted_idx:
        print(strings_by_idx[idx])


def print_outcome(slv, levels, tasks, codes_by_idx, status, verbose=1):
    print(f'Solver status: {status_string[status]}', end='')
    print(f', time(CPU sec): {slv.UserTime():.2f}', end='')
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f', objective: {slv.ObjectiveValue()}')
        if verbose > 0:
            print()
            print_solution(slv, levels, tasks, codes_by_idx)
    else:
        print()


def add_precedences(mdl, levels, idx, tasks):
    for k, lvl in enumerate(levels[:-1]):
        for i, _ in enumerate(lvl):
            for j, _ in enumerate(levels[k+1]):
                mdl.Add(tasks[idx,k,i].end <= tasks[idx,k+1,j].start)

def add_all_precedences(mdl, levels, tasks):
    for idx, lvl in levels.items():
        add_precedences(mdl, lvl, idx, tasks)


def add_makespan_variables(mdl, levels, codes_by_idx, last, eoh):
    codes = ['red', 'yellow', 'green', 'white']
    last_by_code = {c: [] for c in codes}
    for idx in levels.keys():
        last_by_code[codes_by_idx[idx]].append(last[idx])

    obj_by_code = {}   
    for code in codes:
        obj_by_code[code] = mdl.NewIntVar(0, eoh, f'mk_{code}')
        if len(last_by_code[code]) > 0:
            mdl.AddMaxEquality(obj_by_code[code], [t.end for t in last_by_code[code]])
    obj_by_code['all'] = mdl.NewIntVar(0, eoh, f'mk_global')
    mdl.AddMaxEquality(obj_by_code['all'], [obj_by_code[c] for c in codes])
    return obj_by_code


def add_waittime_variables(mdl, levels, codes_by_idx, tasks, eoh):
    codes = ['red', 'yellow', 'green', 'white']
    fbc = {c: [] for c in codes}
    for idx in levels.keys():
        fbc[codes_by_idx[idx]].append(tasks[idx, 0, 0])

    obj_by_code = {}
    for code in codes:
        obj_by_code[code] = mdl.NewIntVar(0, len(codes_by_idx)*eoh, f'wt_{code}')
        if len(fbc[code]) > 0:
            mdl.Add(obj_by_code[code] ==
                    sum(t.start for t in fbc[code]))
    obj_by_code['all'] = mdl.NewIntVar(0, len(codes_by_idx) * eoh, f'wt_globai')
    mdl.Add(obj_by_code['all'] ==  sum(obj_by_code[c] for c in codes))
    return obj_by_code


def add_cumulatives(mdl, tasks, capacities):
    res_intervals = {r: [] for r in capacities.keys()}
    for task in tasks.values():
        if task.ttype in capacities.keys():
            res_intervals[task.ttype].append(task.interval)

    for rtype, cap in capacities.items():
        intervals = res_intervals[rtype]
        mdl.AddCumulative(intervals, [1]*len(intervals), cap)

def add_no_overlap(mdl, levels, idx, tasks):
    intervals = []
    for k, lvl in enumerate(levels):
        for i, _ in enumerate(lvl):
            intervals.append(tasks[idx, k, i].interval)
    mdl.AddNoOverlap(intervals)


def add_all_no_overlap(mdl, levels, tasks):
    for idx, lvl in levels.items():
        add_no_overlap(mdl, lvl, idx, tasks)


def plot_scalability_evalulation(plist, tlist, plist_cmp=None, tlist_cmp=None,
                                 lbl=None, lbl_cmp=None,
                                 figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(plist, tlist, label=lbl)
    if plist_cmp is not None:
        plt.plot(plist_cmp, tlist_cmp, label=lbl_cmp)
    if lbl is not None and lbl_cmp is not None:
        plt.legend()
    # sl2 = [f'{p}({status_string[s]})' for p, s in zip(plist, slist)]
    # plt.xticks(plist, sl2, rotation=45)
    plt.tight_layout()


def get_seq_hints(levels, codes_by_idx):
    codes = ['red', 'yellow', 'green', 'white']
    hints = {}
    totdur = 0
    for idx, lvls in sorted(levels.items(),
            key=lambda t: codes.index(codes_by_idx[t[0]])):
        for k, lvl in enumerate(lvls):
            for i, a in enumerate(lvl):
                hints[idx,k,i] = totdur
                totdur += get_dur(a)
    return hints


def solve_bounded_waittime_problem(levels, codes_by_idx, codes, capacities,
        ub_by_code={}, tlim=None, hints=None, aplus=None):
    mdl = cp_model.CpModel()    
    eoh = get_horizon(levels)
    rl, dl = {idx:0 for idx in levels}, {idx:eoh for idx in levels}
    tasks, last = add_all_vars(mdl, levels, rl, dl)
    add_all_precedences(mdl, levels, tasks)

    # Add extra precedences
    if aplus is not None:
        for (idx, k, i), out_arcs in aplus.items():
            for idx2, k2, i2 in out_arcs:
                mdl.Add(tasks[idx,k,i].end <= tasks[idx2,k2,i2].start)

    add_cumulatives(mdl, tasks, {'visit': 3, 'ultrasound': 2, 'RX': 2})
    add_all_no_overlap(mdl, levels, tasks)
    obj_by_code = add_waittime_variables(mdl, levels, codes_by_idx, tasks, eoh)
    
    for code in ub_by_code: # <-- Enforce upper bounds
        mdl.Add(obj_by_code[code] <= ub_by_code[code])

    # Add hints
    if hints is not None:
        for (idx, k, i), stval in hints.items():
            mdl.AddHint(tasks[idx,k,i].start, stval)

    mdl.Minimize(obj_by_code[codes[-1]]) # <-- Focus by default on the last code
    slv = cp_model.CpSolver()
    if tlim is not None: slv.parameters.max_time_in_seconds = tlim
    status = slv.Solve(mdl)    
    return status, slv, tasks


def goal_programming(levels, codes_by_idx, codes, capacities,
        tlim=None, verbose=1, hints=None, ub_by_code=None, aplus=None):
    # Handle the upper bounds
    if ub_by_code is None:
        wt_by_code = {}
    else:
        wt_by_code = copy.deepcopy(ub_by_code)
    ttime = 0
    for i, code in enumerate(codes):
        l_tlim = None if tlim is None else (tlim-ttime) / (len(codes)-i)
        status, slv, tasks = solve_bounded_waittime_problem(levels, codes_by_idx,
                codes[:i+1], capacities, ub_by_code=wt_by_code, tlim=l_tlim,
                hints=hints, aplus=aplus)
        ttime += slv.UserTime()
        if verbose:
            print_outcome(slv, levels, tasks, codes_by_idx, status, verbose=(i == len(codes)-1))
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            wt_by_code[code] = int(slv.ObjectiveValue())
            starts = {tidx:slv.Value(tasks[tidx].start) for tidx in tasks}
            hints = starts
        else:
            wt_by_code, starts = None, None
            break
    if wt_by_code is not None:
        wt_by_code['all'] = sum(wt_by_code[code] for code in codes)
    return ttime, wt_by_code, starts


def scheduling_lns(levels, codes_by_idx, codes, capacities,
        tlim, init_time, it_time, nb_size, verbose=1):
    # Build initial solution
    ttime, wt_by_code, starts = goal_programming(levels, codes_by_idx, codes,
            capacities=capacities,
                               tlim=init_time, verbose=0)
    if verbose:
        print(f'Initial solution in {ttime:.2f} sec, {wt_by_code}')

    # Loop until the time limit is not over
    patients = [idx for idx in codes_by_idx.keys()]
    np.random.seed(42)
    while tlim - ttime > 0:
        aplus, aminus = sol_to_pos(levels, starts, capacities) # Obtain a POS
        relaxed = np.random.choice(patients, nb_size, replace=False) # Relax patients
        for idx in relaxed:
            aplus, aminus = remove_patient_from_pos(aplus, aminus, idx)
        ub_by_code = copy.deepcopy(wt_by_code) # Require an improvement
        ub_by_code['all'] -= 1
        # Re-solve
        itlim = min(it_time, tlim - ttime)
        itime, iwt_by_code, istarts = goal_programming(levels, codes_by_idx, codes,
                            capacities=capacities, tlim=itlim, verbose=0, aplus=aplus,
                            ub_by_code=ub_by_code)
        ttime += max(itime, 0.1) # Update the time limit
        # Update the best solution
        if iwt_by_code is not None:
            if verbose:
                objs = ','.join(f'{iwt_by_code[c]}' for c in codes + ['all'])
                print(f'Total time: {ttime:.2f}, neighborhood explored in {itime:.2f} sec, waiting times {objs}')
            wt_by_code = iwt_by_code
            starts = istarts
    return ttime, wt_by_code, starts

# def goal_programming(levels, codes_by_idx, codes, capacities,
#                      tlim=None, verbose=2, aplus={}, ub_by_code=None, hints=None):
#     if ub_by_code is None:
#         wt_by_code = {}
#     else:
#         wt_by_code = copy.deepcopy(ub_by_code)
#     ttime = 0
#     # Start the main loop
#     for i, code in enumerate(codes):
#         # Determine what to print
#         if verbose > 0:
#             if code == codes[-1]: vrb = 2
#             else: vrb = 1
#         else:
#             vrb = 0
#         # Determine the time limit
#         ctlim = None
#         if tlim is not None:
#             if ub_by_code is not None:
#                 ctlim = tlim-ttime
#             else:
#                 ctlim = (tlim-ttime) / (len(codes)-i)
#         # Solve
#         status, slv, tasks = solve_bounded_waittime_problem(levels, codes_by_idx,
#                 codes[:i+1], capacities, ub_by_code=wt_by_code, tlim=ctlim,
#                 hints=hints, aplus=aplus)
#         ttime += slv.UserTime()
#         if vrb:
#             print_outcome(slv, levels, tasks, codes_by_idx, status, verbose=vrb)
#         if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
#             starts = {tidx:slv.Value(tasks[tidx].start) for tidx in tasks}
#             hints = starts
#             wt_by_code[code] = int(slv.ObjectiveValue())
#         else:
#             starts = None
#             wt_by_code = None
#             break
#     if wt_by_code is not None:
#         wt_by_code['all'] = sum(wt_by_code[code] for code in codes)
#     return ttime, wt_by_code, starts


def unit_flow(levels, stasks, starts, aplus, aminus):
    nonprocessed = []
    src, dur = None, 0
    for dst in stasks:
        if src is not None and starts[src] + dur <= starts[dst]:
            # Build an arc
            aplus[src].append(dst)
            aminus[dst].append(src)
            # Reset the source
            src = None
        if src is None:
            # Set a new source
            src = dst
            dur = get_dur(levels[src[0]][src[1]][src[2]])
        else:
            # Store as non-processed
            nonprocessed.append(dst)
    return nonprocessed


def sol_to_pos(levels, starts, capacities):
    # Prepare data structures to store the graph
    aplus, aminus = {}, {}
    # Split tasks by resource
    stasks = {r:[] for r in capacities}
    for idx, k, i in starts:
        ttype = levels[idx][k][i]
        if ttype in capacities:
            stasks[ttype].append((idx, k, i))
            aplus[idx, k, i] = []
            aminus[idx, k, i] = []
    # Sort all collections by increasing start
    for res in capacities:
        stasks[res] = sorted(stasks[res], key=lambda t: starts[t])
    # Loop over all resources
    for res, cap in capacities.items():
        for _ in range(cap):
            # Route one flow unit
            stasks[res] = unit_flow(levels, stasks[res], starts, aplus, aminus)
    return aplus, aminus


def validate_pos(aplus, aminus):
    for src in aplus:
        for dst in aplus[src]:
            assert(src in aminus[dst])
    for dst in aminus:
        for src in aminus[dst]:
            assert(dst in aplus[src])


def remove_task_from_pos(aplus, aminus, task_key):
    # Process all predecessors
    for src in aminus[task_key]:
        # Process all successors
        for dst in aplus[task_key]:
            # Transfer arcs
            aminus[dst].append(src)
            aplus[src].append(dst)
    # Remove ingoing arcs
    for src in aminus[task_key]:
        aplus[src].remove(task_key)
    # Remove outgoing arcs
    for dst in aplus[task_key]:
        aminus[dst].remove(task_key)
    # Remove the node from the arc
    del aplus[task_key]
    del aminus[task_key]


def remove_patient_from_pos(aplus, aminus, target_idx):
    aplus_res = copy.deepcopy(aplus)
    aminus_res = copy.deepcopy(aminus)
    for idx, k, i in aplus:
        if idx == target_idx:
            remove_task_from_pos(aplus_res, aminus_res, (idx, k, i))
    return aplus_res, aminus_res


def generate_compatibility_graph(size, seed=42, icp_chance=0.05):
    # Blood type prevalence
    brp = {'O+':.385, 'A+':.355, 'B+':.075, 'AB+':.025,
           'O-':.070, 'A-':.060, 'B-':.015, 'AB-':.015}
    # Compatibility matrix (receipient, donor)
    cpm = {('O-', 'O-'),
            ('O+', 'O-'), ('O+', 'O+'),
            ('A-', 'O-'), ('A-', 'A-'),
            ('A+', 'O-'), ('A+', 'O+'), ('A+', 'A-'), ('A+', 'A+'),
            ('B-', 'O-'), ('B-', 'B-'),
            ('B+', 'O-'), ('B+', 'O+'), ('B+', 'B-'), ('B+', 'B+'),
            ('AB-', 'O-'), ('AB-', 'A-'), ('AB-', 'B-'), ('AB-', 'AB-'),
            ('AB+', 'O-'), ('AB+', 'O+'), ('AB+', 'A-'), ('AB+', 'A+'),
                ('AB+', 'B-'), ('AB+', 'B+'), ('AB+', 'AB-'), ('AB+', 'AB+'),
            }
    # Build an incompatibility matrix
    icpm = {(b1, b2) for b1 in brp.keys() for b2 in brp.keys()}
    icpm -= cpm
    icpm = sorted(list(icpm))
    # Define the probability of each pair
    icpm_p = np.array([brp[t[0]]*brp[t[1]] for t in icpm])
    icpm_p /= icpm_p.sum()
    # Generate incompatible pairs
    rng = np.random.default_rng(seed)
    pair_idxs = rng.choice(range(len(icpm)), size, p=icpm_p)
    Pair = collections.namedtuple('pair', 'recipient donor')
    pairs = {i:Pair(*icpm[j]) for i, j in enumerate(pair_idxs)}
    # Generate basic compatibility arcs
    arcs = [(i, j) for i, pd in pairs.items()
                   for j, pr in pairs.items()
                   if (pr.recipient, pd.donor) in cpm]
    # Remove some arcs to simulate other sources of incompatibility
    aidx = np.arange(len(arcs))
    np.random.seed(seed)
    mask = rng.choice([False, True], len(aidx),
            p=[icp_chance, 1-icp_chance])
    aidx = aidx[mask]
    arcs = [arcs[i] for i in aidx]
    # Convert the graph in forward star and sparse matrix format
    aplus = {i:[] for i in pairs.keys()}
    for src, dst in arcs:
        aplus[src].append(dst)
    # Convert the graph in to a sparse matrix
    return pairs, arcs, aplus


# def shortest_cycles2(aplus, weights, max_len, cap=None):
#     # Build an adjacency matrix
#     nnodes = len(weights)
#     adj = np.full((nnodes, nnodes), np.inf)
#     for i, alist in aplus.items():
#         for j in alist:
#             adj[i,j] = 1
#     # Prepare a list with candidate roots
#     roots = set(range(nnodes))
#     cycles = []
#     ccosts = []
#     processed = set()
#     while len(roots) > 0:
#         root = roots.pop()
#         tcl, tct = shortest_cycles_from_root2(root, adj, weights, max_len)
#         cycles += tcl
#         ccosts += tct
#         # Stop if at full capcity
#         if cap is not None and len(cycles) >= cap:
#             return cycles[:cap]
#         # Mark the nodes in the cycles as processed
#         for cl in tcl:
#             for i in cl:
#                 if i in roots:
#                     roots.remove(i)
#     return cycles, ccosts


# def shortest_cycles_from_root2(root, adj, weights, max_len):
#     nnodes = len(weights)
#     # Initial distance
#     dst = np.full(nnodes, np.inf)
#     dst[root] = 0
#     # Initial predecessor along the shortest path
#     prd = np.full((nnodes, max_len), -1)
#     # Time-unfolded graph traversal
#     idx = np.arange(nnodes)
#     cycles = []
#     ccosts = []
#     for k in range(max_len):
#         # Compute distances from all predecessors
#         # print(root, k)
#         # print(dst)
#         prd_dst = (adj * weights).T + dst
#         prd_dst = np.nan_to_num(prd_dst, nan=np.inf,
#                 posinf=np.inf, neginf=np.inf)
#         # print(prd_dst)
#         # Compute the best predecessors
#         prd[:, k] = np.argmin(prd_dst, axis=1).ravel()
#         # print(prd[:, k])
#         # Recomput the mimum distances
#         dst = prd_dst[idx, prd[:, k]]
#         # print(dst)
#         # Detect cycles
#         if dst[root] < np.inf:
#             cnode = root
#             cycle = []
#             for h in range(k, -1, -1):
#                 cpred = prd[cnode, h]
#                 cycle.append(cpred)
#                 cnode = cpred
#             # print('cycle detected:', cycle)
#             cycles.append(tuple(cycle))
#             ccosts.append(dst[root])
#     return cycles, ccosts


def shortest_cycles(aplus, weights, max_len):
    # Store the graph in backward star format
    aminus = {i:[] for i in aplus}
    for i, alist in aplus.items():
        for j in alist:
            aminus[j].append(i)
    aplus = copy.deepcopy(aplus)
    # Prepare a list with candidate roots
    cycles, ccosts = [], []
    for root in range(len(weights)):
        tcl, tct = shortest_cycles_from_root(root, aplus, weights, max_len)
        cycles += tcl
        ccosts += tct
        # Remove all forward arcs to the processed root
        for i in aminus[root]:
            aplus[i].remove(root)
    return cycles, ccosts


def shortest_cycles_from_root(root, aplus, weights, max_len):
    spt = {root: {root}} # initial shortest paths
    dst = {root: weights[root]} # shortest path distances
    cycles, ccosts = [], []
    for k in range(max_len): # loop over the possible cycle lengths
        ndst, nspt = {}, {}
        for i in dst: # process all visited nodes
            for j in aplus[i]: # loop over outgoing arcs
                if j == root: # detect cycles
                    cycles.append(spt[i])
                    ccosts.append(dst[i])
                elif j in spt[i]: # skip subcycles
                    continue 
                elif j not in ndst or dst[i] + weights[j] < ndst[j]:
                    ndst[j] = dst[i] + weights[j]
                    nspt[j] = spt[i] | {j}
        dst, spt = ndst, nspt
    return cycles, ccosts


def cycle_next(seq, nsteps, aplus, cycles, cap=None):
    node = seq[-1]
    # Consider all possible extensions
    successors = np.array(aplus[node])
    np.random.shuffle(successors)
    for dst in successors:
        # Early exit
        if cap is not None and len(cycles) >= cap: return
        # Try to close the cycle
        if dst == seq[0] and dst == min(seq):
            cycles.add(tuple(seq))
        elif nsteps > 0 and dst not in seq:
            cycle_next(seq+[dst], nsteps-1, aplus, cycles, cap)


def find_all_cycles(aplus, max_length, cap=None, seed=42):
    cycles = set()
    roots = np.array(list(aplus.keys()))
    np.random.seed(seed)
    np.random.shuffle(roots)
    for node in roots:
        if cap is None or len(cycles) < cap:
            cycle_next([node], max_length-1, aplus, cycles, cap)
    return list(cycles)


def cycle_formulation(pairs, cycles, tlim=None, relaxation=False, verbose=1):
    # Build the solver
    if relaxation:
        slv = pywraplp.Solver.CreateSolver('CLP')
    else:
        slv = pywraplp.Solver.CreateSolver('CBC')
    # Some convenience values
    infinity = slv.infinity()
    ncycles = len(cycles)
    npairs = len(pairs)
    # Group cycles by pair (saves time later on)
    cpp = {i:[] for i in range(npairs)}
    for j, cycle in enumerate(cycles):
        for i in cycle: cpp[i].append(j)
    # Build all variables
    if relaxation:
        x = [slv.NumVar(0, infinity, f'x_{j}') for j in range(ncycles)]
    else:
        x = [slv.IntVar(0, 1, f'x_{j}') for j in range(ncycles)]
    # Build all the constraints
    for i in range(npairs):
        if relaxation:
            slv.Add(-sum(x[j] for j in cpp[i]) >= -1)
        else:
            slv.Add(sum(x[j] for j in cpp[i]) <= 1)
    # Define the objective
    obj = sum(len(c) * x[j] for j, c in enumerate(cycles))
    if relaxation:
        slv.Minimize(-obj)
    else:
        slv.Maximize(obj)
    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000*tlim)
    # Solve
    status = slv.Solve()
    # Extract results
    duals = None
    sol = None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol = {f'x_{j}':x[j].solution_value() for j in range(len(x))}
        sol['objective'] = slv.Objective().Value()
        if relaxation:
            duals = np.array([c.dual_value() for c in slv.constraints()])
    # Print stuff
    if verbose > 0:
        print_cycle_formulation_outcome(status, sol, slv, verbose, relaxation)
    # Return results
    return sol, slv.WallTime()/1000, duals


def print_cycle_formulation_outcome(status, sol, slv, verbose, relaxation):
    print(f'Solution time: {slv.WallTime()/1000:.3f}', end='')
    if status not in (slv.OPTIMAL, slv.FEASIBLE):
        print(', no feasible solution found')
    else:
        print(f', objective value: {slv.Objective().Value()}', end='')
        if status == slv.OPTIMAL: print(' (optimal)', end='')
        else: print(' (optimal)', end='')
        if verbose > 1:
            if not relaxation:
                sol = {k:int(v) for k, v in sol.items()}
            print(f', solution: {sol}')
        else:
            print()


def cycle_formulation_cg(pairs, aplus, max_len, itcap=10, tol=1e-3, verbose=1):
    # Initial pool of arcs
    weights = -np.ones(len(pairs)) 
    cycles, _ = shortest_cycles(aplus, weights, max_len=max_len)
    # Start the main loop
    converged = False
    for itn in range(itcap):
        # Solve the LP relaxation
        sol, stime, duals = cycle_formulation(pairs, cycles, verbose=0, relaxation=True)
        if verbose > 0:
            print(f'(CG, it. {itn}), #cycles: {len(cycles)}, time: {stime}, relaxation objective: {sol["objective"]:.2f}')
        # Find shortest paths
        weights = -np.ones(len(pairs)) + duals
        scl, sct = shortest_cycles(aplus, weights, max_len=max_len)
        # Cycles with negative reduced cost
        nrc_cycles = [scl[i] for i, c in enumerate(sct) if c < -tol]
        if verbose > 0:
            print(f'(CG, it. {itn}), #cycles with negative reduced cost: {len(nrc_cycles)}')
        if len(nrc_cycles) == 0:
            converged = True
            break
        else: cycles += nrc_cycles
    return cycles, converged

# ==============================================================================
# OLD STUFF
# ==============================================================================

# def load_data(data_folder):
#     # Read the CSV files
#     fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
#     cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
#     datalist = []
#     nmcn = 0
#     for fstem in fnames:
#         # Read data
#         data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
#         # Drop the last two columns (parsing errors)
#         data.drop(columns=[26, 27], inplace=True)
#         # Replace column names
#         data.columns = cols
#         # Add the data source
#         data['src'] = fstem
#         # Shift the machine numbers
#         data['machine'] += nmcn
#         nmcn += len(data['machine'].unique())
#         # Generate RUL data
#         cnts = data.groupby('machine')[['cycle']].count()
#         cnts.columns = ['ftime']
#         data = data.join(cnts, on='machine')
#         data['rul'] = data['ftime'] - data['cycle']
#         data.drop(columns=['ftime'], inplace=True)
#         # Store in the list
#         datalist.append(data)
#     # Concatenate
#     data = pd.concat(datalist)
#     # Put the 'src' field at the beginning and keep 'rul' at the end
#     data = data[['src'] + cols + ['rul']]
#     # data.columns = cols
#     return data


# def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
#         figsize=figsize, autoclose=True, s=4):
#     if autoclose: plt.close('all')
#     plt.figure(figsize=figsize)
#     plt.imshow(data.T.iloc[:, :], aspect='auto',
#             cmap='RdBu', vmin=vmin, vmax=vmax)
#     if labels is not None:
#         # nonzero = data.index[labels != 0]
#         ncol = len(data.columns)
#         lvl = - 0.05 * ncol
#         # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
#         #         s=s, color='tab:orange')
#         plt.scatter(labels.index, np.ones(len(labels)) * lvl,
#                 s=s,
#                 color=plt.get_cmap('tab10')(np.mod(labels, 10)))
#     plt.tight_layout()


# def split_by_field(data, field):
#     res = {}
#     for fval, gdata in data.groupby(field):
#         res[fval] = gdata
#     return res



# def partition_by_machine(data, tr_machines):
#     # Separate
#     tr_machines = set(tr_machines)
#     tr_list, ts_list = [], []
#     for mcn, gdata in data.groupby('machine'):
#         if mcn in tr_machines:
#             tr_list.append(gdata)
#         else:
#             ts_list.append(gdata)
#     # Collate again
#     tr_data = pd.concat(tr_list)
#     ts_data = pd.concat(ts_list)
#     return tr_data, ts_data




# def sliding_window_by_machine(data, wlen, cols, stride=1):
#     l_w, l_m, l_r = [], [], []
#     for mcn, gdata in data.groupby('machine'):
#         # Apply a sliding window
#         tmp_w = sliding_window_2D(gdata[cols], wlen, stride)
#         # Build the machine vector
#         tmp_m = gdata['machine'].iloc[wlen-1::stride]
#         # Build the RUL vector
#         tmp_r = gdata['rul'].iloc[wlen-1::stride]
#         # Store everything
#         l_w.append(tmp_w)
#         l_m.append(tmp_m)
#         l_r.append(tmp_r)
#     res_w = np.concatenate(l_w)
#     res_m = np.concatenate(l_m)
#     res_r = np.concatenate(l_r)
#     return res_w, res_m, res_r



# def plot_rul(pred=None, target=None,
#         stddev=None,
#         q1_3=None,
#         same_scale=True,
#         figsize=figsize, autoclose=True):
#     if autoclose:
#         plt.close('all')
#     plt.figure(figsize=figsize)
#     if target is not None:
#         plt.plot(range(len(target)), target, label='target',
#                 color='tab:orange')
#     if pred is not None:
#         if same_scale or target is None:
#             ax = plt.gca()
#         else:
#             ax = plt.gca().twinx()
#         ax.plot(range(len(pred)), pred, label='pred',
#                 color='tab:blue')
#         if stddev is not None:
#             ax.fill_between(range(len(pred)),
#                     pred-stddev, pred+stddev,
#                     alpha=0.3, color='tab:blue', label='+/- std')
#         if q1_3 is not None:
#             ax.fill_between(range(len(pred)),
#                     q1_3[0], q1_3[1],
#                     alpha=0.3, color='tab:blue', label='1st/3rd quartile')
#     plt.legend()
#     plt.tight_layout()


# class RULCostModel:
#     def __init__(self, maintenance_cost, safe_interval=0):
#         self.maintenance_cost = maintenance_cost
#         self.safe_interval = safe_interval

#     def cost(self, machine, pred, thr, return_margin=False):
#         # Merge machine and prediction data
#         tmp = np.array([machine, pred]).T
#         tmp = pd.DataFrame(data=tmp,
#                            columns=['machine', 'pred'])
#         # Cost computation
#         cost = 0
#         nfails = 0
#         slack = 0
#         for mcn, gtmp in tmp.groupby('machine'):
#             idx = np.nonzero(gtmp['pred'].values < thr)[0]
#             if len(idx) == 0:
#                 cost += self.maintenance_cost
#                 nfails += 1
#             else:
#                 cost -= max(0, idx[0] - self.safe_interval)
#                 slack += len(gtmp) - idx[0]
#         if not return_margin:
#             return cost
#         else:
#             return cost, nfails, slack


# def opt_threshold_and_plot(machine, pred, th_range, cmodel,
#         plot=True, figsize=figsize, autoclose=True):
#     # Compute the optimal threshold
#     costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
#     opt_th = th_range[np.argmin(costs)]
#     # Plot
#     if plot:
#         if autoclose:
#             plt.close('all')
#         plt.figure(figsize=figsize)
#         plt.plot(th_range, costs)
#         plt.tight_layout()
#     # Return the threshold
#     return opt_th



# class MLPplusNormal(keras.Model):
#     def __init__(self, input_shape, hidden):
#         super(MLPplusNormal, self).__init__()
#         # Build the estimation model for mu and sigma
#         model_in = keras.Input(shape=input_shape, dtype='float32')
#         x = model_in
#         for h in hidden:
#             x = layers.Dense(h, activation='relu')(x)
#         mu = layers.Dense(1, activation='linear')(x)
#         logsigma = layers.Dense(1, activation='linear')(x)
#         self.model = keras.Model(model_in, [mu, logsigma])
#         # Choose what to track at training time
#         self.loss_tracker = keras.metrics.Mean(name="loss")

#     @property
#     def metrics(self):
#         return [self.loss_tracker]

#     def call(self, data, training=True):
#         if training:
#             x, y = data
#         else:
#             x = data
#         mu, logsigma = self.model(x)
#         dist = tfp.distributions.Normal(mu, k.exp(logsigma))
#         print(dist)
#         raise Exception('HAHAHA')
#         if training:
#             return dist.log_prob(y)
#         else:
#             return dist.mean(), dist.stddev()

#     # def log_loss(self, data):
#     #     log_densities = self(data)
#     #     return -tf.reduce_mean(log_densities)

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             log_densities = self(data)
#             loss = -tf.reduce_mean(log_densities)
#             # loss = self.log_loss(data)
#         g = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(g, self.trainable_variables))
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}

# ==============================================================================
# OLD FUNCTIONS
# ==============================================================================


# def plot_distribution_2D(estimator=None, samples=None,
#         xr=None, yr=None,
#         figsize=figsize, autoclose=True):
#     if autoclose:
#         plt.close('all')
#     plt.figure(figsize=figsize)
#     if samples is not None:
#         plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
#     if estimator is not None:
#         if xr is None:
#             xr = np.linspace(-1, 1, 100)
#         if yr is None:
#             yr = np.linspace(-1, 1, 100)
#         nx = len(xr)
#         ny = len(yr)
#         xc = np.repeat(xr, ny)
#         yc = np.tile(yr, nx)
#         data = np.vstack((xc, yc)).T
#         dvals = np.exp(estimator.score_samples(data))
#         dvals = dvals.reshape((nx, ny))
#         plt.imshow(dvals.T[::-1, :], aspect='auto')
#         plt.xticks(np.linspace(0, len(xr), 5), np.linspace(xr[0], xr[-1], 5))
#         plt.xticks(np.linspace(0, len(yr), 5), np.linspace(yr[0], yr[-1], 5))
#     plt.tight_layout()


# def get_errors(signal, labels, thr, tolerance=1):
#     pred = signal[signal > thr].index
#     anomalies = labels[labels != 0].index

#     fp = set(pred)
#     fn = set(anomalies)
#     for lag in range(-tolerance, tolerance+1):
#         fp = fp - set(anomalies+lag)
#         fn = fn - set(pred+lag)
#     return fp, fn


# class HPCMetrics:
#     def __init__(self, c_alarm, c_missed, tolerance):
#         self.c_alarm = c_alarm
#         self.c_missed = c_missed
#         self.tolerance = tolerance

#     def cost(self, signal, labels, thr):
#         # Obtain errors
#         fp, fn = get_errors(signal, labels, thr, self.tolerance)

#         # Compute the cost
#         return self.c_alarm * len(fp) + self.c_missed * len(fn)


# def opt_threshold(signal, labels, th_range, cmodel):
#     costs = [cmodel.cost(signal, labels, th) for th in th_range]
#     best_th = th_range[np.argmin(costs)]
#     best_cost = np.min(costs)
#     return best_th, best_cost




# def collect_training(data, tr_ts_ratio):
#     if isinstance(data, dict):
#         tr_list = []
#         for key, kdata in data.items():
#             sep = int(np.round(tr_ts_ratio * len(kdata)))
#             tr_list.append(kdata.iloc[:sep])
#         tr_data = pd.concat(tr_list)
#     else:
#         sep = int(np.round(tr_ts_ratio * len(data)))
#         tr_data = data.iloc[:sep]
#     return tr_data


# def kde_ad(x_tr, x_vs, y_vs, x_ts, y_ts, th_range, h_range, cmodel):
#     # Optimize the bandwidth, if more than one value is given
#     if len(h_range) > 1:
#         params = {'bandwidth': h_range}
#         opt = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
#         opt.fit(x_tr)
#         h = opt.best_params_['bandwidth']
#     else:
#         h = h_range[0]
#     # Traing a KDE estimator
#     kde = KernelDensity(bandwidth=h)
#     kde.fit(x_tr)
#     # Generate a signal for the validation set
#     ldens_vs = kde.score_samples(x_vs)
#     signal_vs = pd.Series(index=x_vs.index, data=-ldens_vs)
#     # Threshold optimization
#     th, val_cost = opt_threshold(signal_vs, y_vs, th_range, cmodel)
#     # Compute the cost over the test set
#     ldens_ts = kde.score_samples(x_ts)
#     signal_ts = pd.Series(index=x_ts.index, data=-ldens_ts)
#     ts_cost = cmodel.cost(signal_ts, y_ts, th)
#     # Return results
#     return kde, ts_cost


# def ae_ad(ae, x_tr, x_vs, y_vs, x_ts, y_ts, th_range, cmodel):
#     # Traing the autoencoder
#     ae.compile(optimizer='RMSProp', loss='mse')
#     cb = [callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
#     ae.fit(x_tr, x_tr, validation_split=0.1, callbacks=cb,
#            batch_size=32, epochs=20, verbose=0)
#     # Generate a signal for the validation set
#     preds_vs = pd.DataFrame(index=x_vs.index,
#             columns=x_vs.columns, data=ae.predict(x_vs))
#     sse_vs = np.sum(np.square(preds_vs - x_vs), axis=1)
#     signal_vs = pd.Series(index=x_vs.index, data=sse_vs)
#     # Threshold optimization
#     th, val_cost = opt_threshold(signal_vs, y_vs, th_range, cmodel)
#     # Compute the cost over the test set
#     preds_ts = pd.DataFrame(index=x_ts.index,
#             columns=x_ts.columns, data=ae.predict(x_ts))
#     sse_ts = np.sum(np.square(preds_ts - x_ts), axis=1)
#     signal_ts = pd.Series(index=x_ts.index, data=sse_ts)
#     ts_cost = cmodel.cost(signal_ts, y_ts, th)
#     # Return results
#     return ts_cost

# def coupling(input_shape, nunits=64, nhidden=2, reg=0.01):
#     assert(nhidden >= 0)    
#     x = keras.layers.Input(shape=input_shape)
#     # Build the layers for the t transformation (translation)
#     t = x
#     for i in range(nhidden):
#         t = Dense(nunits, activation="relu", kernel_regularizer=l2(reg))(t)
#     t = Dense(input_shape, activation="linear", kernel_regularizer=l2(reg))(t)
#     # Build the layers for the s transformation (scale)
#     s = x
#     for i in range(nhidden):
#         s = Dense(nunits, activation="relu", kernel_regularizer=l2(reg))(s)
#     s = Dense(input_shape, activation="tanh", kernel_regularizer=l2(reg))(s)
#     # Return the layers, wrapped in a keras Model object
#     return keras.Model(inputs=x, outputs=[s, t])


# class RealNVP(keras.Model):
#     def __init__(self, input_shape, num_coupling, units_coupling=32, depth_coupling=0,
#             reg_coupling=0.01):
#         super(RealNVP, self).__init__()
#         self.num_coupling = num_coupling
#         # Distribution of the latent space
#         self.distribution = tfp.distributions.MultivariateNormalDiag(
#             loc=np.zeros(input_shape, dtype=np.float32),
#             scale_diag=np.ones(input_shape, dtype=np.float32)
#         )
#         # Build a mask
#         half_n = int(np.ceil(input_shape/2))
#         m1 = ([0, 1] * half_n)[:input_shape]
#         m2 = ([1, 0] * half_n)[:input_shape]
#         self.masks = np.array([m1, m2] * (num_coupling // 2), dtype=np.float32)
#         # Choose what to track at training time
#         self.loss_tracker = keras.metrics.Mean(name="loss")
#         #  Build layers
#         self.layers_list = [coupling(input_shape, units_coupling, depth_coupling, reg_coupling)
#                             for i in range(num_coupling)]

#     @property
#     def metrics(self):
#         """List of the model's metrics.
#         We make sure the loss tracker is listed as part of `model.metrics`
#         so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
#         at the start of each epoch and at the start of an `evaluate()` call.
#         """
#         return [self.loss_tracker]

#     def call(self, x, training=True):
#         log_det_inv, direction = 0, 1
#         if training: direction = -1
#         for i in range(self.num_coupling)[::direction]:
#             x_masked = x * self.masks[i]
#             reversed_mask = 1 - self.masks[i]
#             s, t = self.layers_list[i](x_masked)
#             s, t = s*reversed_mask, t*reversed_mask
#             gate = (direction - 1) / 2
#             x = reversed_mask * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s)) \
#                 + x_masked
#             log_det_inv += gate * tf.reduce_sum(s, axis=1)
#         return x, log_det_inv

#     def log_loss(self, x):
#         log_densities = self.score_samples(x)
#         return -tf.reduce_mean(log_densities)

#     def score_samples(self, x):
#         y, logdet = self(x)
#         log_probs = self.distribution.log_prob(y) + logdet
#         return log_probs

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             loss = self.log_loss(data)
#         g = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(g, self.trainable_variables))
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}

#     def test_step(self, data):
#         loss = self.log_loss(data)
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}



# def plot_rnvp_transformation(rnvp, xr=None, yr=None,
#         figsize=figsize, autoclose=True):
#     if autoclose:
#         plt.close('all')
#     plt.figure(figsize=figsize)
#     # Define ranges
#     if xr is None:
#         xr = np.linspace(-1, 1, 7, dtype=np.float32)
#     if yr is None:
#         yr = np.linspace(-1, 1, 7, dtype=np.float32)
#     # Build the input set
#     nx = len(xr)
#     ny = len(yr)
#     xc = np.repeat(xr, ny)
#     yc = np.tile(yr, nx)
#     data = np.vstack((xc, yc)).T
#     # Transform the input step
#     z, _ = rnvp(data, training=False)
#     # Obtain traces
#     traces = np.concatenate((
#         data.reshape(1, -1, 2),
#         z.numpy().reshape(1, -1, 2),
#         ))
#     # Plot traces
#     for i in range(traces.shape[1]):
#         plt.plot(traces[:, i, 0], traces[:, i, 1], ':',
#                 color='0.8', zorder=0)
#         xh = plt.scatter(data[:, 0], data[:, 1],
#                 color='tab:blue', s=4, zorder=1)
#         zh = plt.scatter(z.numpy()[:, 0], z.numpy()[:, 1],
#                 color='tab:orange', s=4, zorder=1)
#     plt.legend([xh, zh], ['x', 'z'])
#     plt.tight_layout()


