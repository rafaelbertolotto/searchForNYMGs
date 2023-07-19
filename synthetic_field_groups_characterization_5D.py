# We set the path of the (so nicely nice) gaiaunlimited stuff BEFORE importing gaiaunlimited stuff
# import os
# os.environ['GAIAUNLIMITED_DATADIR'] = '/home/rafaelbertolotto/gaiaunlimited_files'

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import warnings
import pickle
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from gaiaunlimited.selectionfunctions import DR3RVSSelectionFunction
# from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG_hpx7
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG

import dbscan_astro as db
import mass_photometry_tools as massPhoto
import DBSCANoutputAnalysisTools as dbat
import group_simulator_tools as GST

warnings.filterwarnings("ignore")
sns.set_context("talk")
mpl.style.use("seaborn")
sns.set_context("paper",font_scale=1.5)
sns.set_style("whitegrid")


#%%

# data reading:

# MG error data builded based on GDR3 MG error
MGerrInterp     = pd.read_csv('builded_or_modified_cat/MGerrInterp.csv')

# This is GOG final field with RVs
gogRVFinalField = pd.read_csv('builded_or_modified_cat/gogFinalField.csv',
                              usecols=['source_id',
                                       'MG',
                                       'phot_g_mean_mag',
                                       'phot_bp_mean_mag',
                                       'phot_rp_mean_mag',
                                       'ra',
                                       'dec',
                                       'X',
                                       'Y',
                                       'Z',
                                       'U',
                                       'V',
                                       'W',
                                       'pmraLSR',
                                       'pmdecLSR',
                                       'pdr3sf',
                                       'prvssf',
                                       'p'])

# Photometric and kinematic errors data from GDR3
allErrorsOriginal = pd.read_csv('builded_or_modified_cat/gdr3MSPMS.csv',usecols=['phot_g_mean_mag',
                                                                                 'BP_err',
                                                                                 'RP_err',
                                                                                 'G_err',
                                                                                 'parallax_error',
                                                                                 'radial_velocity_error',
                                                                                 'pmra_error',
                                                                                 'pmdec_error'])

# These are the general observed statistics of the chosen NYMGs used here to define the synthetic NYMGs
statsForSyntheticNymgs = pd.read_csv('statsForSyntheticNymgs.csv')


#%%


def runDetectionAlgorithm(clusterField,
                          trueCluster,
                          ageIso,
                          deltaData,
                          nH0Values,
                          sample_class,
                          alpha,
                          nymgName):
    
    # We first check if the analysis has to be done in 5D or 6D
    if sample_class == '5d':
        columns1                = ['pmraLSR','pmdecLSR']
    elif sample_class == '6d':
        columns1                = ['U','V','W']
    columns2                    = ['X','Y','Z']

    columnsAlgorithm        = ['Nmin','veps','reps']
    # Columns for the kinematic information
    kinematicColumns            = columns1+columns2
    
    # We make the photometric selection of the sample
    clusterFieldPhoto               = massPhoto.isochroneSelector(data           = clusterField,
                                                                  age            = ageIso,
                                                                  deltaMag       =  [1e6,0.5],
                                                                  dataMagName    = 'MG',
                                                                  dataColorNames = ['phot_bp_mean_mag','phot_rp_mean_mag'])
    
    # We set the indexes values to IDs values
    clusterField      = clusterField.set_index(clusterField.source_id)
    clusterFieldPhoto = clusterFieldPhoto.set_index(clusterFieldPhoto.source_id)
    
    # We build the first columns of the output of this function that contains the general information
    # of the cluster before applying the DEA    
    generalGroupColumns             = ['nTrueCluster',
                                       'nObservedCluster',
                                       'nPhotoCluster',
                                       'nField',
                                       'nPhotoField']
    generalGroupColumns             = generalGroupColumns+kinematicColumns
    
    finalInfo                       = pd.DataFrame(np.zeros([1,len(generalGroupColumns)]),columns = generalGroupColumns)
    finalInfo.nTrueCluster          = trueCluster.shape[0] 
    finalInfo.nObservedCluster      = clusterField[clusterField.source_id < 0].shape[0]
    finalInfo.nPhotoCluster         = clusterFieldPhoto[clusterFieldPhoto.source_id < 0].shape[0]
    finalInfo.nField                = clusterField[clusterField.source_id > 0].shape[0]
    finalInfo.nPhotoField           = clusterFieldPhoto[clusterFieldPhoto.source_id > 0].shape[0]
    
    for name in kinematicColumns:
        finalInfo[name+'Mean'] = trueCluster[name].mean()
    for name in kinematicColumns:
        finalInfo[name+'BoxMin'] = clusterField[name].min()
    for name in kinematicColumns:
        finalInfo[name+'BoxMax'] = clusterField[name].max()


    # We set the similarity tolerance for the search of the best case and initialize the dataframe that will contain all the
    # output of all the cases of measuring purity, recovery and the rest of the cool stuff
    similarity_tolerance    = 0.1
    allLocalInfo            = pd.DataFrame()
    
    # Here start the detection algorithm that must use all of the algorithm parameters defined by the field characterization
    deltaNmin                   = deltaData.Nmin.unique()
    
    for nind in np.arange(0,len(deltaNmin),1):
        Nmin                    = deltaNmin[nind]
        delta1                  = deltaData[deltaData.Nmin == Nmin][columnsAlgorithm[1]].unique()
        
        for ind1 in np.arange(0,len(delta1),1):
            eps1                = delta1[ind1]
            delta2              = deltaData[(deltaData.Nmin == Nmin) &
                                            (deltaData[columnsAlgorithm[1]] == eps1)][columnsAlgorithm[2]].unique()
            for ind2 in np.arange(0,len(delta2),1):
                eps2                    = delta2[ind2]
                nH0                     = nH0Values[Nmin][eps1][eps2]
                
                # Here we run the DEA for the parameters chosen
                clusters                = db.DEA(data       = clusterFieldPhoto, 
                                                 columnList = kinematicColumns, 
                                                 weightList = [eps1,
                                                               eps1,
                                                               eps1,
                                                               eps2,
                                                               eps2,
                                                               eps2], 
                                                 Nmin       = Nmin, 
                                                 alpha      = alpha, 
                                                 nH0        = nH0, 
                                                 nMax       = nMax, 
                                                 Njobs      = 1)
                
                if clusters.shape[0] > 0:
                    clusters            = clusters.set_index(clusters.input_index)
                    clusters            = pd.concat([clusters,clusterFieldPhoto.source_id],axis=1,join='inner')
                    
                    # We identify the observed clusters
                    observedCluster     = clusterFieldPhoto[clusterFieldPhoto.source_id < 0]
                    # We measure completness, purity and the rest of the cool stuff
                    localInfo           = dbat.purityRecoveryEtcMeasurer(clusters                = clusters,
                                                                         observedRealCluster     = observedCluster,
                                                                         realCluster             = trueCluster,
                                                                         realClusterLabel        = nymgName,
                                                                         parametersCombination   = [Nmin,eps1,eps2])
                    
                    allLocalInfo        = pd.concat([allLocalInfo,localInfo],axis=0).reset_index(drop=True)
                    
    if allLocalInfo.shape[0] > 0:
        if allLocalInfo.reps.unique()[0] == 0:
            allLocalInfo = pd.DataFrame(columns=allLocalInfo.columns)
    # We finally join the information of the best case to the rest of the finalInfo  
    bestLocalInfo   = dbat.bestCaseFinder(allLocalInfo,similarity_tolerance,'minMax',sample_class).reset_index(drop=True)
    finalInfo       = pd.concat([finalInfo.reset_index(drop=True),bestLocalInfo],axis=1,join='inner')
    
    return finalInfo


#%%


def groupFieldCharacterizer(nymgName,
                            NstarsInPop,
                            ageIso,
                            agePop,
                            dG,
                            sample_class,
                            dr3maps,
                            rvmaps,
                            allErrorFunctions,
                            GLimToFilter,
                            lim1,
                            lim2,
                            std1,
                            std2,
                            plim,
                            pcolumn,
                            field,
                            dxBoxLim,
                            centers,
                            deltaData,
                            nH0Values,
                            alpha,
                            nControl):
    
# This is just to check everything's going ok
    print(nControl)
    # We build the real and the observed cluster
    observedCluster,realCluster = GST.randomObservedGroup(N                             = NstarsInPop,
                                                          Age                           = agePop,
                                                          dG                            = dG,
                                                          sample_class                  = sample_class,
                                                          dr3maps                       = dr3maps,
                                                          rvmaps                        = rvmaps,
                                                          allErrorFunctions             = allErrorFunctions,
                                                          GLimToFilter                  = GLimToFilter,
                                                          lim1                          = lim1,
                                                          lim2                          = lim2,
                                                          std1                          = uvwstd,
                                                          std2                          = xyzstd,
                                                          plim                          = plim,
                                                          pcolumn                       = pcolumn,
                                                          field                         = field,
                                                          dxBoxLim                      = dxBoxLim,
                                                          centers                       = centers)


    # We run the detection algorithm (this includes the photometric selection)
    parametersToAppend         = runDetectionAlgorithm(clusterField        = observedCluster,
                                                        trueCluster        = realCluster,
                                                        ageIso             = agePop,
                                                        deltaData          = deltaData,
                                                        nH0Values          = nH0Values,
                                                        sample_class       = sample_class,
                                                        alpha              = alpha,
                                                        nymgName           = nymgName)

    
    return parametersToAppend
    

#%%


# We load the selection function for Gaia DR3 and the Gaia DR3 RVs
dr3sf  = DR3SelectionFunctionTCG()#DR3SelectionFunctionTCG_hpx7()
rvssf  = DR3RVSSelectionFunction()

# We define the general variables for the clusters generation
centersUVW      = [-10,-20,-5]              # center of the populated velocity space
nCluster        = 1                         # number of clusters simulated for fixed values of stds and the number of members
maxProcessors   = 20                        # maximum number of processors available
Rlim0           = 200                       # initial radial limit for the generation of the mean position in space
Vlim0           = 70                        # initial radial limit for the generation of the mean position in velocity space
MUlim0          = 90                        # initial radial limit for the generation of the mean position in proper motion space
dG              = 0.25                      # bin size for G magnitude
sample_class    = '5d'                      # this variable specifies if we are working in the 5D or the 6D sample

if sample_class == '6d':
    sample      = 'RVs'
elif sample_class == '5d':
    sample      = '5D'
    
alphaV          = 0.867
alphaMu         = 0.847
nMax            = 1e3

# We define the number of processors to use
Nprocessors     = int(np.min([nCluster , maxProcessors]))

# Minimum probability limit value for the selection function filter
plim            = gogRVFinalField[gogRVFinalField.p.isnull().values==False].p.mean()-gogRVFinalField[gogRVFinalField.p.isnull().values==False].p.std()
  

#%%


# We build the family of kinematic error cumulative distribution per G bin
allErrors                       = allErrorsOriginal.drop([]).copy()
allErrorDistributions           = {}
if sample_class == '6d':
    allErrors                   = allErrors[allErrors.radial_velocity_error.isnull().values == False]

errorNames                  = allErrors.drop(['phot_g_mean_mag'],axis=1).columns
    
for i in np.arange(0,len(errorNames),1):
    errorDistributions          = GST.accumulativeDistributionInterpolator1D(dataToInterpolate   = allErrors,
                                                                             dBin                = dG,
                                                                             columnBin           = 'phot_g_mean_mag',
                                                                             columnY             = errorNames[i])
    allErrorDistributions.update({errorNames[i]:errorDistributions})   
    
# We set the G magnitude limit to filter the generated clusters
GLimToFilter = [allErrors.phot_g_mean_mag.min(),allErrors.phot_g_mean_mag.max()]
  

#%%

nymgAges          = statsForSyntheticNymgs.age
isochroneAges     = pd.DataFrame({'age':np.arange(1e7,11e7,1e7)})
isochroneNymgAges = pd.DataFrame(columns=isochroneAges.columns)

# Characterization test
finalParameterSet   = {}
for nymgInd in np.arange(0,1,1):
    nymgStats       = statsForSyntheticNymgs.iloc[nymgInd]
    nymgName        = nymgStats.NYMG
    npop            = int(nymgStats.realN)
    ageCluster      = nymgStats.age
    uvwstd          = [nymgStats.stdU,nymgStats.stdV,nymgStats.stdW]
    xyzstd          = [nymgStats.stdX,nymgStats.stdY,nymgStats.stdZ]
    Rlim            = Rlim0
    Vlim            = Vlim0
    
    age = isochroneAges[np.abs(isochroneAges.age-ageCluster*1e6) == np.min(np.abs(isochroneAges.age-ageCluster*1e6))].iloc[0][0]

    nvrClusterFinalCoord = pd.read_csv(f'dataframes_output/Ndmethod/nvrClusterFinalCoord_{sample}_{age}.csv')
    
    with open(f'dataframes_output/Ndmethod/nvrClusterFinal_{sample}_{age}.pickle', 'rb') as handle:
        nvrClusterFinal = pickle.load(handle)
        
    parametersList  = Parallel(n_jobs=Nprocessors)(delayed(groupFieldCharacterizer)\
                                                  (nymgName                     = nymgName,
                                                    NstarsInPop                 = npop,
                                                    ageIso                      = ageCluster*1e6,
                                                    agePop                      = ageCluster*1e6,
                                                    dG                          = dG,
                                                    sample_class                = sample_class,
                                                    dr3maps                     = dr3sf,
                                                    rvmaps                      = rvssf,
                                                    allErrorFunctions           = allErrorDistributions,
                                                    GLimToFilter                = GLimToFilter,
                                                    lim1                        = Vlim,
                                                    lim2                        = Rlim,
                                                    std1                        = uvwstd,
                                                    std2                        = xyzstd,
                                                    plim                        = plim,
                                                    pcolumn                     = 'p',
                                                    field                       = gogRVFinalField,
                                                    dxBoxLim                    = [20,20,20,3,3,3],
                                                    centers                     = centersUVW+[0,0,0],
                                                    deltaData                   = nvrClusterFinalCoord,
                                                    nH0Values                   = nvrClusterFinal,
                                                    alpha                       = alphaV,
                                                    nControl                    = n) for n in np.arange(0,nCluster,1))
        
    finalParameterSet.update({nymgName:parametersList})
    
    # with open(f'dataframes_output/Ndmethod/synth_groups_and_field/synthGroupsAndFieldPurityCompletness_{sample}_{nymgName}.pickle', 'wb') as handle:
    #     pickle.dump(finalParameterSet, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%%


# Here is a quick test to check everything work

# nymgInd         = 1
# nymgStats       = statsForSyntheticNymgs.iloc[nymgInd]
# nymgName        = nymgStats.NYMG
# npop            = int(nymgStats.realN)
# ageCluster      = nymgStats.age
# uvwstd          = [nymgStats.stdU,nymgStats.stdV,nymgStats.stdW]
# xyzstd          = [nymgStats.stdX,nymgStats.stdY,nymgStats.stdZ]
# Rlim            = Rlim0
# Vlim            = Vlim0

# print(nymgName)
# print(ageCluster)
# age = isochroneAges[np.abs(isochroneAges.age-ageCluster*1e6) == np.min(np.abs(isochroneAges.age-ageCluster*1e6))].iloc[0][0]

# nvrClusterFinalCoord = pd.read_csv(f'dataframes_output/Ndmethod/nvrClusterFinalCoord_{sample}_{age}.csv')

# with open(f'dataframes_output/Ndmethod/nvrClusterFinal_{sample}_{age}.pickle', 'rb') as handle:
#     nvrClusterFinal = pickle.load(handle)
    
# observedCluster,realCluster = GST.randomObservedGroup(N                             = npop,
#                                                       Age                           = ageCluster*1e6,
#                                                       dG                            = dG,
#                                                       sample_class                  = sample_class,
#                                                       dr3maps                       = dr3sf,
#                                                       rvmaps                        = rvssf,
#                                                       allErrorFunctions             = allErrorDistributions,
#                                                       GLimToFilter                  = GLimToFilter,
#                                                       lim1                          = Vlim,
#                                                       lim2                          = Rlim,
#                                                       std1                          = uvwstd,
#                                                       std2                          = xyzstd,
#                                                       plim                          = plim,
#                                                       pcolumn                       = 'p',
#                                                       field                         = gogRVFinalField,
#                                                       dxBoxLim                      = [20,20,20,3,3,3],
#                                                       centers                       = centersUVW+[0,0,0])



# parametersToAppend         = runDetectionAlgorithm(clusterField       = observedCluster,
#                                                     trueCluster        = realCluster,
#                                                     ageIso             = ageCluster*1e6,
#                                                     deltaData          = nvrClusterFinalCoord,
#                                                     nH0Values          = nvrClusterFinal,
#                                                     sample_class       = sample_class,
#                                                     alpha              = alphaV,
#                                                     nymgName           = nymgName)


# parametersToAppend = groupFieldCharacterizer\
#                                                   (nymgName                     = nymgName,
#                                                     NstarsInPop                 = npop,
#                                                     ageIso                      = ageCluster*1e6,
#                                                     agePop                      = ageCluster*1e6,
#                                                     dG                          = dG,
#                                                     sample_class                = '6d',
#                                                     dr3maps                     = dr3sf,
#                                                     rvmaps                      = rvssf,
#                                                     allErrorFunctions  = allErrorDistributions,
#                                                     GLimToFilter                = GLimToFilter,
#                                                     lim1                        = Vlim,
#                                                     lim2                        = Rlim,
#                                                     std1                        = uvwstd,
#                                                     std2                        = xyzstd,
#                                                     plim                        = plim,
#                                                     pcolumn                     = 'p',
#                                                     field                       = gogRVFinalField,
#                                                     dxBoxLim                    = [20,20,20,3,3,3],
#                                                     centers                     = centersUVW+[0,0,0],
#                                                     deltaData                   = nvrClusterFinalCoord,
#                                                     nH0Values                   = nvrClusterFinal,
#                                                     alpha                       = alphaV,
#                                                     nControl                    = 0)


# Nmin = int(parametersToAppend.Nmin.iloc[0])
# veps = parametersToAppend.veps.iloc[0]
# reps = parametersToAppend.reps.iloc[0]

# dbinfo,clusters = db.runFullPhotometricDEA(data           = observedCluster,
#                                            columnList     = ['U','V','W','X','Y','Z'],
#                                            weightList     = [veps,
#                                                              veps,
#                                                              veps,
#                                                              reps,
#                                                              reps,
#                                                              reps],
#                                            Nmin           = Nmin,
#                                            alpha          = alphaV,
#                                            nH0            = nvrClusterFinal[Nmin][veps][reps],
#                                            nMax           = 1e3,
#                                            Njobs          = 1,
#                                            age            = AGE,
#                                            wantClusters   = 'yes',
#                                            magMax         = 0.5)



# observedRealCluster = observedCluster[observedCluster.source_id<0]
# recoveredCluster = clusters[clusters.source_id<0]

# print(recoveredCluster.shape[0]/observedRealCluster.shape[0]-parametersToAppend.observedRecovery)

# plt.scatter(observedCluster.X,observedCluster.Y,color='blue')
# plt.scatter(observedCluster[observedCluster.source_id<0].X,observedCluster[observedCluster.source_id<0].Y,color='red')
# plt.scatter(clusters.X,clusters.Y,color='orange')
# plt.scatter(clusters[clusters.source_id<0].X,clusters[clusters.source_id<0].Y,color='black')


# plt.scatter(observedCluster.U,observedCluster.V)
# plt.scatter(observedCluster[observedCluster.source_id<0].U,observedCluster[observedCluster.source_id<0].V)
# plt.scatter(clusters.U,clusters.V)
# plt.scatter(clusters[clusters.source_id<0].U,clusters[clusters.source_id<0].V)



# for i in np.arange(0,statsForSyntheticNymgs.shape[0],1):
#     stats = statsForSyntheticNymgs.iloc[i]
#     age = stats.age
#     nymgName = stats.NYMG
#     print(nymgName)
#     print(age)
#     age = isochroneAges[np.abs(isochroneAges.age-ageCluster*1e6) == np.min(np.abs(isochroneAges.age-ageCluster*1e6))].iloc[0][0]
#     finalParameters = pd.read_pickle(f'dataframes_output/Ndmethod/synth_groups_and_field/synthGroupsAndFieldPurityCompletness_RVs_{age}.pickle')
#     params = finalParameters[nymgName]
#     for i in np.arange(0,len(params),1):
#         param = params[i]        
#         if param.shape[0]>0:
#             print(f'{nymgName} ({age}): (Nmin,veps,reps)=({param.Nmin.iloc[0]},{param.veps.iloc[0]},{param.reps.iloc[0]})')


#%%


# kinematicErrors = allErrorDistributions['parallax_error']
# keys = list(kinematicErrors.keys())
# for i in np.arange(0,len(kinematicErrors),1):
#     key =  keys[i]
#     kinematicError = kinematicErrors[key]
#     dmg = np.arange(kinematicError.x.min(),kinematicError.x.max(),0.01)
#     plt.scatter(dmg,kinematicError(dmg),s=1,c=key*np.ones(len(dmg)),cmap='rainbow',vmin=keys[0],vmax=keys[len(keys)-1])
# plt.colorbar()
# plt.show()





# photoErrors = allErrorDistributions['G_err']
# keys = list(photoErrors.keys())
# for i in np.arange(0,len(photoErrors),1):
#     key =  keys[i]
#     photoError = photoErrors[key]
#     dmg = np.arange(photoError.x.min(),photoError.x.max(),0.01)
#     plt.scatter(dmg,photoError(dmg),s=1,c=key*np.ones(len(dmg)),cmap='rainbow',vmin=keys[0],vmax=keys[len(keys)-1])
# plt.colorbar()
# plt.show()

