import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


#%%

# This function receives as input a dataframe clusters that contains the info of detected clusters (IDs and labels) and 
# a dataframe that consists on the match between clusters and a real known cluster of interest. The function returns 
# all the clusters from clusters that contains at least one member of match.
def groupsMatchBuilder(clusters,match):
    clusterLabels = match.cluster_labels.unique()
    groupMatch    = pd.DataFrame(columns=clusters.columns)
    for i in np.arange(0,len(clusterLabels),1):
        label      = clusterLabels[i]
        cluster    = clusters[clusters.cluster_labels == label]
        groupMatch = pd.concat([groupMatch,cluster],axis=0).reset_index(drop=True)
    return groupMatch


#%%


# This function receives as an input a dataframe with the info of detected clusters (IDs and cluster labels), the a real
# knwon cluster of interest, its observed counterpart from the data, its name and the combination of parameters associated
# to the set of detected clusters. 
# The function returns a dataframe with the real and observed recovery rates, the parameters combination, the fragmentation
#(number of clusters from clusters that contains at least one member from observedRealCluster) and the groups purity
#(number of members from observedRealCluster in clusters divided by the number of members from the clusters that contains
# members from observedRealCluster).
def purityRecoveryEtcMeasurer(clusters,
                              observedRealCluster,
                              realCluster,
                              realClusterLabel,
                              parametersCombination):
    
    columns = ['clusterLabel',
               'observedRecovery',
               'fragmentation',
               'realRecovery',
               'groupsPurity',
               'reps',
               'veps',
               'Nmin']
    
    analysisResults                     = pd.DataFrame(np.zeros([1,len(columns)]),columns=columns)

    realCluster                         = realCluster.set_index(realCluster.source_id)
    clusters                            = clusters.set_index(clusters.input_index)

    match                               = pd.concat([clusters.cluster_labels,realCluster],axis=1,join='inner')
    
    if match.shape[0] > 0:
        groupMatch                          = groupsMatchBuilder(clusters,match)
    
        
        analysisResults.clusterLabel        = realClusterLabel
        analysisResults.observedRecovery    = match.shape[0]/observedRealCluster.shape[0]
        analysisResults.fragmentation       = match.cluster_labels.unique().shape[0]
        analysisResults.realRecovery        = match.shape[0]/realCluster.shape[0]
        analysisResults.groupsPurity        = match.shape[0]/groupMatch.shape[0]
        analysisResults.Nmin                = parametersCombination[0]
        analysisResults.veps                = parametersCombination[1]
        analysisResults.reps                = parametersCombination[2]
    
    return analysisResults


#%%


# This function receives as input a dict setOfClusters with dataframes that contains info (IDs and labels) of
# detected clusters for different parameter combinations from parameterCombination, a real known cluster of interest 
# (realCluster, whose label is realClusterLabel) and its observed counterpart (the cluster as observed in the data). 
# The function loops purityRecoveryEtcMeasurer on the diferent clusters and returns a dataframe where each row corresponds
# to an output of purityRecoveryEtcMeasurer for each parameter combination results.

def setOfClustersToAnalysisResults(setOfClusters,
                                   observedRealCluster,
                                   realCluster,
                                   realClusterLabel,
                                   parameterCombinations):
    columns = ['clusterLabel',
               'observedRecovery',
               'fragmentation',
               'realRecovery',
               'groupsPurity',
               'reps',
               'veps',
               'Nmin']
    
    allresults       = pd.DataFrame(columns=columns)
    
    for i in np.arange(0,parameterCombinations.shape[0],1):
        Nmin = parameterCombinations.Nmin.iloc[i]
        reps = parameterCombinations.reps.iloc[i]
        veps = parameterCombinations.veps.iloc[i]
        
        clusters = setOfClusters[Nmin][veps][reps]
        clusters = clusters.set_index(clusters.input_index)
        
        match               = pd.concat([clusters.cluster_labels,realCluster],axis=1,join='inner')
        condition           = match.shape[0] > 0
        
        if condition:
            analysisResults = purityRecoveryEtcMeasurer(clusters                = clusters,
                                                        observedRealCluster     = observedRealCluster,
                                                        realCluster             = realCluster,
                                                        realClusterLabel        = realClusterLabel,
                                                        parametersCombination   = [Nmin,veps,reps])
            
            allresults                          = pd.concat([allresults,analysisResults],axis=0).reset_index(drop=True)
            
    return allresults


#%%


# This function receives a dataframe that contains the information of applying purityRecoveryEtcMeasurer to multiple cases
# (similar to the output of setOfClustersToAnalysisResults) and returns the best case as the one that maximizes the observed
# recovery and groups purity and minimizes the fragmentation. To do so, the function remove all cases where recovery differs
# from purity by a fraction bigger than similarity_tolerance.

def bestCaseFinder(allResults,
                   similarity_tolerance,
                   criteria,
                   sample_class):
    rexp = 3
    vexp = 1
    if sample_class == '5d':
        vexp = 2
    elif sample_class == '6d':
        vexp = 3
    
    if criteria == 'minMax':
        column1 = 'min_purity_recovery'
        column2 = 'max_purity_recovery'
    elif criteria == 'recovery':
        column1 = 'min_purity_recovery'
        column2 = 'observedRecovery'
    elif criteria == 'purity':
        column1 = 'min_purity_recovery'
        column2 = 'groupsPurity'
    
    if allResults.shape[0] > 0:
        allResults    = pd.concat([allResults.reset_index(drop=True),
                                   pd.DataFrame({'density':allResults.Nmin/(allResults.reps**rexp*allResults.veps**vexp),
                                                 'min_purity_recovery':allResults[['observedRecovery','groupsPurity']].min(axis=1),
                                                 'max_purity_recovery':allResults[['observedRecovery','groupsPurity']].max(axis=1)},
                                                index=allResults.index)],axis=1)
        minFragmentation = np.max([5,allResults.fragmentation.min()])
        allResults       = allResults[allResults.fragmentation < minFragmentation]
        
        bestCase1 = allResults[np.abs(allResults[column1]-allResults[column1].max()) <= similarity_tolerance]
        bestCase1 = bestCase1[bestCase1[column2] == bestCase1[column2].max()]
        bestCase1 = bestCase1[bestCase1.fragmentation == bestCase1.fragmentation.min()]        
        bestCase1 = bestCase1[bestCase1.density == bestCase1.density.min()]
        
        bestCase2 = allResults[np.abs(allResults[column1]-allResults[column1].max()) == 0]
        bestCase2 = bestCase2[bestCase2[column2] == bestCase2[column2].max()]
        bestCase2 = bestCase2[bestCase2.fragmentation == bestCase2.fragmentation.min()]  
        bestCase2 = bestCase2[bestCase2.density == bestCase2.density.min()]
        
        if np.abs(bestCase1[column2].iloc[0]-bestCase2[column2].iloc[0]) > similarity_tolerance/2:
            bestCase = bestCase1
        else:
            bestCase = bestCase2
            
    else:
        bestCase = pd.DataFrame(np.zeros([1,allResults.shape[1]+2]),columns=list(allResults.columns)+['density',
                                                                                                      'min_purity_recovery'
                                                                                                      'max_purity_recovery'])
        
    return bestCase.reset_index(drop=True)













