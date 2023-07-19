# This code contains useful functions that use the dbscan function for astronomical purpose.
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

#%%


# dbscan():

# This function receives as input a dataframe data with at least the columns defined by colnames and the DBSCAN parameters
# epsilon and Nmin. This function apply the DBSCAN algorithm to the space of the data defined by colnames using the 
# input parameters and returns a 2-columns dataframe with the following information of the detected clusters: the 
# input indexes of the data and the detected clusters labels from DBSCAN.

def dbscan(data,
           colnames,
           epsilon,
           N_min,
           Njobs):
# We first select the columns on which will be applied DBSCAN:
    dbscan_input = data[colnames]
# Here we store the information of the input index:
    input_index                         = pd.DataFrame({'input_index':dbscan_input.index.values}).reset_index(drop=True)
                
# Here starts the DBSCAN process:
# We apply the DBSCAN to dbscan_input (finally, exciting right?):
    db                                  = DBSCAN(eps=epsilon, min_samples=N_min,n_jobs=Njobs).fit(dbscan_input)

# output_labels contains the DBSCAN labels for each star:
    output_labels                       = pd.DataFrame({'DBSCAN_labels':db.labels_})
    
# We join the input index information with output_labels:
    input_index_output_labels           = pd.concat([input_index,output_labels],axis=1,join="inner")
    
    return input_index_output_labels
    
#%%


# H0Filter():

# This function receives as input a dataframe with detected clusters and test H0 on these clusters by removing
# clusters with a number of clusters smaller than nH0. An optional upper limit can be fixed with nMax.

def H0Filter(clusters,labelsColumn,nH0,nMax):
    if clusters.shape[0] >= nH0:
        labels = clusters[labelsColumn].unique()
        for i in np.arange(0,len(labels),1):
            label = labels[i]
            cluster = clusters[clusters[labelsColumn] == label]
            if (cluster.shape[0] < nH0) | (cluster.shape[0] > nMax):
                clusters = clusters[clusters[labelsColumn] != label]
    else:
        clusters = pd.DataFrame(columns = clusters.columns)
    return clusters

#%%


# NDdbscan():

# The DBSCAN Elliptic Algorithm (DEA) receives as input a dataframe data that at least contains columnList columns,
# the DBSCAN parameters Nmin and epsilon, weights (weightList). This function first scales the axes defined by 
# columnList before applying the DBSCAN algorithm to the scaled spaced defined by these columns and returns a 
# 2-columns dataframe that contains the following information of the detected clusters: the input indexes of the
# data and the detected clusters labels from DBSCAN. H0 is tested on the detected clusters using H0Filter.

def DEA(data,
        columnList,
        weightList,
        Nmin,
        alpha,
        nH0,
        nMax,
        Njobs):
    
    data                                = data[columnList]
    for i in np.arange(0,len(columnList),1):
        data[columnList[i]]             = data[columnList[i]]/weightList[i]
        
    # Here we store the information of the input index:
    input_index                         = pd.DataFrame({'input_index':data.index.values}).reset_index(drop=True)
    
    # We apply the DBSCAN to dbscan_input (finally, exciting right?):
    db                                  = DBSCAN(eps=alpha*np.sqrt(2), min_samples=Nmin,n_jobs=Njobs).fit(data)

    # output_labels contains the DBSCAN labels for each star:
    output_labels                       = pd.DataFrame({"cluster_labels":db.labels_})
    output_labels                       = output_labels[output_labels["cluster_labels"] != -1]
    
    if nH0 > 0:
        
        output_labels                   = H0Filter(output_labels,
                                                   'cluster_labels',
                                                   nH0,
                                                   nMax)
    
    # We join the input index information with output_labels:
    input_index_output_labels           = pd.concat([input_index,output_labels],axis=1,join="inner")
   
    input_index_output_labels.columns   = ["input_index","cluster_labels"]
    
    return input_index_output_labels


#%%


# This function runs the full detection algorithm that consists in the photometric selection+the DEA application
# (for Gaia data format only) on the space defined by columnList columns using weightList weights and Nmin as DEA
# parameters and assuming an age age for the isochrone selection and nH0 for the H0 testing. If wantClusters is set 
# to yes then the function returns the output of the DEA function+its match with the input data (so you get the full
# initial information of the detected clusters), otherwise if set to no, it will only return the DEA output.

def runFullPhotometricDEA(data,
                          columnList,
                          weightList,
                          Nmin,
                          alpha,
                          nH0,
                          nMax,
                          Njobs,
                          age,
                          wantClusters,
                          magMax):
    
    import mass_photometry_tools as massPhoto
    
    dataPhoto = massPhoto.isochroneSelector(data            = data,
                                            age             = age,
                                            deltaMag        = [1e6,magMax],
                                            dataMagName     = 'MG',
                                            dataColorNames  = ['phot_bp_mean_mag','phot_rp_mean_mag'])
    
    dataPhoto = dataPhoto.set_index(dataPhoto.source_id)
    
    input_index_output_labels = DEA(dataPhoto,
                                    columnList,
                                    weightList,
                                    Nmin,
                                    alpha,
                                    nH0,
                                    nMax,
                                    Njobs)
    
    if wantClusters == 'yes':
        data                        = data.set_index(data.source_id)
        input_index_output_labels   = input_index_output_labels.set_index(input_index_output_labels.input_index)
        clusters                    = pd.concat([input_index_output_labels,data],axis=1,join='inner')
        return input_index_output_labels,clusters
    elif wantClusters == 'no':
        return input_index_output_labels
    else:
        print('The argument wantClusters must be equal to yes or no.')








