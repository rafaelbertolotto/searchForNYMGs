import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import warnings
import dbscan_astro as db
import mass_photometry_tools as massPhoto
from scipy import interpolate
import pickle
import sys
from joblib import Parallel, delayed
import time as time

warnings.filterwarnings("ignore")
sns.set_context("talk")
mpl.style.use("seaborn")
sns.set_context("paper",font_scale=1.5)
sns.set_style("whitegrid")


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


# data reading:

# This is GOG final field with RVs
gogRVFinalField = pd.read_csv('builded_or_modified_cat/gogRVFinalField.csv')


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


# Variables definition:
AGE             = float(sys.argv[1])
ages            = np.arange(10e6,110e6,10e6).tolist()
delta_Reps      = np.arange(1,31,1)                 # range of Xeps
delta_Veps      = np.arange(0.1,7.1,0.1)            # range of Veps
delta_Nmin      = np.arange(2,31,1)                 # range of Nmin
maxProcessors   = 15
nClusterMin     = 2
alphaV          = 0.867
alphaMu         = 0.847
sample_class    = '6d'

if AGE == 100e6:
    maxMG = 10
else:
    maxMG = 0.5

print(AGE)

# We define the optimal number of processors to be used
Nprocessors = int(np.min([delta_Reps[len(delta_Reps)-1] , maxProcessors]))

dbscan_time     = pd.DataFrame(columns=["N_DBSCAN_input","computation_time","dbscan_space"]) 


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def NdMethod(Nmin,eps1,eps2,data,sample_class,alpha):
    if sample_class == '6d':
        columns         = ['X','Y','Z','U','V','W']
        weights         = [eps1,eps1,eps1,eps2,eps2,eps2]
    elif sample_class == '5d':
        columns         = ['X','Y','Z','pmra','pmdec']
        weights         = [eps1,eps1,eps1,eps2,eps2]
    
    clusters        = db.DEA(data       = data, 
                             columnList = columns, 
                             weightList = weights, 
                             Nmin       = Nmin, 
                             alpha      = alpha, 
                             nH0        = 0, 
                             nMax       = 1e10, 
                             Njobs      = 1)
    return clusters    


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


# ages isochrones generation (SP locus generation is based on 100Myr isochrone, which is for ages[0]):
Xage = massPhoto.isochroneSelector(data           = gogRVFinalField,
                                   age            = AGE,
                                   deltaMag      =  [1e6,0.5],
                                   dataMagName    = 'MG',
                                   dataColorNames = ['phot_bp_mean_mag','phot_rp_mean_mag'])

Xage = Xage.set_index(Xage.source_id)

nvrClusterInfo  = {}
NminStart       = 0

for nind in np.arange(NminStart,len(delta_Nmin),1):
    Nmin = round(delta_Nmin[nind],1)
    vrClustersInfo  = {}
    
    for vind in np.arange(0,len(delta_Veps),1):
        time0 = time.time()
        veps            = round(delta_Veps[vind],1)
        rClusterInfo    = {}
        clustersList    = Parallel(n_jobs=Nprocessors)(delayed(NdMethod)\
                                                      (Nmin,
                                                       reps,
                                                       veps,
                                                       Xage,
                                                       sample_class,
                                                       alphaV) for reps in delta_Reps)
        for rind in np.arange(0,len(delta_Reps),1):
            reps                = round(delta_Reps[rind],1)
            clusters            = clustersList[rind]
            
            nClustersMembers    = clusters.cluster_labels.value_counts().value_counts().sort_index()
            
            nClustersFrecuence  = pd.DataFrame({'nMembers':nClustersMembers.index.values,
                                                'nClusters':nClustersMembers}).reset_index(drop=True)
            
            rClusterInfo.update({reps: nClustersFrecuence})
        vrClustersInfo.update({veps: rClusterInfo})
    nvrClusterInfo.update({Nmin: vrClustersInfo})
    with open(f'dataframes_output/Ndmethod/field_characterized_RVs_{AGE}.pickle', 'wb') as handle:
        pickle.dump(nvrClusterInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# def nMaxFinder(data,confidenceLevel):
#     nTotal          = data.nClusters.sum()
#     for i in np.arange(0,data.shape[0],1):
#         nMembersSum = data.iloc[0:i+1].nClusters.sum()
#         realLevel = nMembersSum/nTotal
#         if realLevel >= confidenceLevel:
#             break
#     return data.nMembers[i]











            
            
