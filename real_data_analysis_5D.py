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
import os.path

warnings.filterwarnings("ignore")
sns.set_context("talk")
mpl.style.use("seaborn")
sns.set_context("paper",font_scale=1.5)
sns.set_style("whitegrid")


#%%


# data reading:

# Isochrone age
AGE             = 20000000.0#float(sys.argv[1])

# MG error data builded based on GDR3 MG error
MGerrInterp     = pd.read_csv('builded_or_modified_cat/MGerrInterp.csv')


# Real data MS and PMS from 5D sample
MSPMSRV = pd.read_csv('builded_or_modified_cat/gdr3MSPMS.csv')

# This table contains the coordinates of the parameter space to be explored
nvrClusterFinalCoord = pd.read_csv(f'dataframes_output/Ndmethod/nvrClusterFinalCoord_5D_{AGE}.csv')

# This dictionary contains the minimum number of members associated to a 0.01 pvalue from the field characterization
with open(f'dataframes_output/Ndmethod/nvrClusterFinal_5D_{AGE}.pickle', 'rb') as handle:
    nvrClusterFinal = pickle.load(handle)


#%%


def DEAmethod(Nmin,eps1,eps2,data,sample_class,nH0,nMax,alpha):
    if sample_class == '6d':
        columns         = ['X','Y','Z','U','V','W']
        weights         = [eps1,eps1,eps1,eps2,eps2,eps2]
    elif sample_class == '5d':
        columns         = ['X','Y','Z','pmraLSR','pmdecLSR']
        weights         = [eps1,eps1,eps1,eps2,eps2]
    
    clusters        = db.DEA(data       = data,
                             columnList = columns,
                             weightList = weights,
                             Nmin       = Nmin,
                             alpha      = alpha,
                             nH0        = nH0,
                             nMax       = nMax,
                             Njobs      = 1)
    return clusters    


#%%


# Isochrone variables definition:
ages            = np.arange(10e6,110e6,10e6).tolist()

if AGE == 100e6:
    maxMG = 10
else:
    maxMG = 0.5

print(AGE)

#%%


# General variables definition:
sample_class    = '5d'
maxProcessors   = 15
alphaV          = 0.867
alphaMu         = 0.847
nMax            = 1.5e3



MSPMSRVage = massPhoto.isochroneSelector(data           = MSPMSRV,
                                         age            = AGE,
                                         deltaMag       =  [1e6,0.5],
                                         dataMagName    = 'MG',
                                         dataColorNames = ['phot_bp_mean_mag','phot_rp_mean_mag'])



MSPMSRVage = MSPMSRVage.set_index(MSPMSRVage.source_id)

delta_Nmin      = nvrClusterFinalCoord.Nmin.unique()
nvrClusterSet   = {}
for nind in np.arange(0,len(delta_Nmin),1):
    Nmin        = round(delta_Nmin[nind],1)
    delta_Veps  = nvrClusterFinalCoord[nvrClusterFinalCoord.Nmin == Nmin].veps.unique()
    vrClusterSet  = {}
    for vind in np.arange(0,len(delta_Veps),1):
        veps            = round(delta_Veps[vind],1)
        delta_Reps      = nvrClusterFinalCoord[(nvrClusterFinalCoord.Nmin == Nmin) &
                                                   (nvrClusterFinalCoord.veps == veps)].reps.unique()
        rClusterSet     = {}
        Nprocessors     = int(np.min([delta_Reps[len(delta_Reps)-1] , maxProcessors]))
        clustersList    = Parallel(n_jobs=Nprocessors)(delayed(DEAmethod)\
                                                      (Nmin         = Nmin,
                                                       eps1         = reps,
                                                       eps2         = veps,
                                                       data         = MSPMSRVage,
                                                       sample_class = sample_class,
                                                       nH0          = nvrClusterFinal[Nmin][veps][reps],
                                                       nMax         = nMax,
                                                       alpha        = alphaMu) for reps in delta_Reps)
        for rind in np.arange(0,len(delta_Reps),1):
            reps                = round(delta_Reps[rind],1)
            clusters            = clustersList[rind]
            
            rClusterSet.update({reps:clusters})
        vrClusterSet.update({veps: rClusterSet})
    nvrClusterSet.update({Nmin: vrClusterSet})
    with open(f'dataframes_output/Ndmethod/realData_5D_{AGE}.pickle', 'wb') as handle:
        pickle.dump(nvrClusterSet, handle, protocol=pickle.HIGHEST_PROTOCOL)





# Nmin    = 5
# veps    = 2.0
# reps    = 30
# alpha   = alphaV
# clusters=NdMethod(Nmin, reps, veps, MSPMSRVage, sample_class, nClusterMin)


# clusters = pd.concat([clusters,MSPMSRVage],axis=1,join='inner')
# labels   = clusters.cluster_labels.unique()




# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# import warnings
# sns.set_context("talk")
# mpl.style.use("seaborn")
# sns.set_context("paper",font_scale=1.5)
# sns.set_style("whitegrid")
# warnings.filterwarnings("ignore")

# col1 = 'X'
# col2 = 'Y'

# # plt.scatter(MSPMSRV[col1],MSPMSRV[col2],s=0.01,color='grey')
# for labelsInd in np.arange(0,len(labels),1):
#     label   = labels[labelsInd]
#     if clusters[clusters.cluster_labels == label].shape[0] > 10:
#         cluster = clusters[clusters.cluster_labels == label]
#         plt.scatter(cluster[col1],
#                     cluster[col2],
#                     s=10,
#                     c=cluster.cluster_labels,
#                     cmap='gist_ncar',
#                     vmin=clusters.cluster_labels.min(),
#                     vmax=clusters.cluster_labels.max())
#         # plt.xlim(-50,50)
#         # plt.ylim(-50,50)
#         plt.show()
    
    


