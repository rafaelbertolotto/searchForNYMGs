# We set the path of the (so nicely nice) gaiaunlimited stuff BEFORE importing gaiaunlimited stuff
import os
os.environ['GAIAUNLIMITED_DATADIR'] = '/home/rafaelbertolotto/gaiaunlimited_files'

# We import the rest of the cool stuff
import numpy as np
import pandas as pd
import warnings
from scipy import interpolate
import mass_photometry_tools as massPhoto
from joblib import Parallel, delayed

# Warnings are anoying... we do not like warnings...
warnings.filterwarnings("ignore")


#%%


# ages isochrones generation (SP locus generation is based on 100Myr isochrone, which is for ages[0]):
BR_to_MG_interps          = []
MG_to_BR_interps          = []
ages = np.arange(1e7,11e7,1e7).tolist()
for i in np.arange(0,len(ages),1):
    age                              = ages[i]
    BR_to_MG_interp, MG_to_BR_interp = massPhoto.iso_interpolation_generator(age)
    
    BR_to_MG_interps                 = BR_to_MG_interps+[BR_to_MG_interp]
    MG_to_BR_interps                 = MG_to_BR_interps+[MG_to_BR_interp]
    
MS_BR_to_MG_interp                   = BR_to_MG_interps[ages.index(100e6)]
MS_MG_to_BR_interp                   = MG_to_BR_interps[ages.index(100e6)]
BR_interval                      = np.arange(MS_BR_to_MG_interp.x.min(),MS_BR_to_MG_interp.x.max()-0.001,0.001)

MS_isochrone = pd.DataFrame({'BR':BR_interval,'MG':MS_BR_to_MG_interp(BR_interval)})


#%%


common_path = 'builded_or_modified_cat/'

gogRVFinalField = pd.read_csv(common_path+'gogRVFinalField.csv')

gogClean  = pd.read_csv(common_path+'gog_XYZUVW.csv')

gdr3MS = pd.read_csv(common_path+'gdr3MS.csv')


#%%


# Here we define a function that generate n random uniformly distributed point within a sphere (or disc) of radius R
def randomUniformSpherical(n,Rmax,columns):
# In case the user desire a 3-d distribution
    if len(columns) == 3:
        R     = Rmax*(np.random.uniform(0,1,n))**(1/3)
        theta = np.arccos(1-2*np.random.uniform(0,1,n))
        phi   = np.random.uniform(0,2*np.pi,n)
        data  = pd.DataFrame({columns[0]:R*np.cos(phi)*np.sin(theta),columns[1]:R*np.sin(phi)*np.sin(theta),columns[2]:R*np.cos(theta)})

# In case the user desire a 2-d distribution
    if len(columns) == 2:
        R     = Rmax*(np.random.uniform(0,1,n))**(1/2)
        phi   = np.random.uniform(0,2*np.pi,n)
        data  = pd.DataFrame({columns[0]:R*np.cos(phi),columns[1]:R*np.sin(phi)})
    return data

# Here we define a function designed to generate a set of stars that have positions, propper motions and photometry similar to GOG sample
def randomStarGenerator(data,data1,data2,r,kinematicrR,nProcessors,MGLims,kinematicColumns):
    stars     = pd.DataFrame()
    Ndiff     = data2.shape[0]-data1.shape[0]
    data      = data[data.pmra**2+data.pmdec**2 <= 250**2] 
    
        
    MGminForPhoto     = np.max([MGLims[0],MS_MG_to_BR_interp.x.min()])
    MGmaxForPhoto     = np.min([MGLims[1],MS_MG_to_BR_interp.x.max()])
    
    if (np.max([data1.shape[0],data2.shape[0]]) < 100) & (MGLims[0] <= MGmaxForPhoto) & (MGminForPhoto <= MGLims[1]):
        dataForPhoto      = pd.DataFrame({'MG':np.random.uniform(MGminForPhoto,
                                                                 MGmaxForPhoto,
                                                                 500),
                                          'BR':np.linspace(MS_MG_to_BR_interp(MGmaxForPhoto),
                                                         MS_MG_to_BR_interp(MGminForPhoto),
                                                         500)})
    
    elif data1.shape[0] < data2.shape[0]:
        dataForPhoto      = pd.DataFrame({'MG':data2.MG,'BR':data2.BR})
    else:
        dataForPhoto      = pd.DataFrame({'MG':data1.MG,'BR':data1.BR})
    
    randomToMG = interpolate.interp1d(np.arange(dataForPhoto.shape[0]) / (dataForPhoto.shape[0] - 1),np.sort(dataForPhoto.MG))
    randomToBR = interpolate.interp1d(np.arange(dataForPhoto.shape[0]) / (dataForPhoto.shape[0] - 1),np.sort(dataForPhoto.BR))
    FakePhoto  = pd.DataFrame({'BR':randomToBR(np.random.random(Ndiff)),'MG':randomToMG(np.random.random(Ndiff))})
    
    def NeumanPositions(n,kinematicColumns):
        subStars  = pd.DataFrame()
        condition = True  
        if len(kinematicColumns) == 3:
            kinematicLimit = 100
            fractionLimit  = 1e-4
        else:
            kinematicLimit = 170
            fractionLimit  = 1e-3
        while condition:
            star           = pd.concat([randomUniformSpherical(1,200,['X','Y','Z']),randomUniformSpherical(1,kinematicLimit,kinematicColumns)],axis=1)
            randomFraction = np.random.uniform(0,fractionLimit)
            
            kinematicSum = 0
            for column in kinematicColumns:
                kinematicSum += (data[column]-star[column][0])**2
            
            trueFraction      = data[((data.X-star.X[0])**2 + (data.Y-star.Y[0])**2 + (data.Z-star.Z[0])**2 <= r**2) &
                                  (kinematicSum <= kinematicrR**2)].shape[0]/data.shape[0]
            
            condition1        = (trueFraction == 0)
            condition2        = (randomFraction > trueFraction)
            condition         = condition1 | condition2
        subStars = pd.concat([subStars,star],axis=0).reset_index(drop=True)
        return subStars
    
    listStars = Parallel(n_jobs=nProcessors)(delayed(NeumanPositions)(j,kinematicColumns) for j in np.arange(0,Ndiff,1))
    for i in np.arange(0,len(listStars),1):
        stars = pd.concat([stars,listStars[i]],axis=0).reset_index(drop=True)
    stars = pd.concat([stars,FakePhoto],axis=1)
    
    return stars,listStars


#%%


# We discard stars below the MS locus
gogMSPMS = gogClean[gogClean["MG"] < 3*(gogClean["BR"])+5.5]

# We discard post-MS stars and get the MS + PMS
gogMSPMS = gogMSPMS[((gogMSPMS["BR"] > 6/5) & (gogMSPMS["MG"] > -0.5*(gogMSPMS["BR"])+4)) |
                    ((gogMSPMS["BR"] <= 6/5) & (gogMSPMS["MG"] > 5*(gogMSPMS["BR"])-3)) |
                    (gogMSPMS["BR"] < 0.7)]

# With a certain tolerance we discard the PMS to keep only the MS
gogMS = massPhoto.isochroneSelector(data            = gogMSPMS,
                                    age             = 100e6,
                                    deltaMag        = [1,2.5],
                                    dataMagName     = 'MG',
                                    dataColorNames  = 'BR')

# We define from the previous sub-samples new sub-samples with RVs
gogRVMSPMS = gogMSPMS[(gogMSPMS.radial_velocity_error.isnull().values == False) & (np.abs(gogMSPMS.radial_velocity_error/gogMSPMS.radial_velocity) <= 0.1)]
gogRVMS    = gogMS[(gogMS.radial_velocity_error.isnull().values == False) & (np.abs(gogMS.radial_velocity_error/gogMS.radial_velocity) <= 0.1)]


#%%


# We define the basice objects needed to build the final synthetic fields
gogFinalField   = pd.DataFrame(columns = gogMS.columns)
dMGbins         = 0.2
MGbins          = np.arange(gdr3MS.MG.min()-gdr3MS.MG.min()%dMGbins+dMGbins,gdr3MS.MG.max()-gdr3MS.MG.max()%dMGbins+dMGbins,dMGbins)
maxProcessors   = 20
minID           = np.max([gogMS.source_id.max(),gogRVFinalField.source_id.max()])

# We go MG bin by bin, building the MS field in such a way it reproduce the Gaia DR3 LF
for mg in np.arange(0,len(MGbins)-1,1):
    MGmin       = round(MGbins[mg],3)
    MGmax       = round(MGbins[mg+1],3)
    print(f'-------------{MGmin}-----------------------')
    
    # We separate in each bin stars with RVs from stars without RVs
    gogMG       = gogMS[(gogMS.MG >= MGmin) &
                        (gogMS.MG < MGmax) &
                        (gogMS.radial_velocity_error.isnull().values == True)]
    
    gdr3MG      = gdr3MS[(gdr3MS.MG >= MGmin) &
                         (gdr3MS.MG < MGmax)]
    
    gogRVchosen = gogRVFinalField[(gogRVFinalField.MG >= MGmin) &
                                  (gogRVFinalField.MG < MGmax)]
            
    # We want to keep the stars selected previously in the field for propper motion space, so if there are more stars in GOG 
    # than in Gaia in this bin we join it with the best n stars in this bin that doesn´t have RVs where n is the number of 
    # stars in Gaia in this bin (here we use the fractional parallax error).
    NdiffRV = gdr3MG.shape[0]-gogRVchosen.shape[0]  
    
    # Objects in gogMG that are not in gogRVchosen
    NgogNotInChosen  = np.max([0,gogMG.shape[0]-NdiffRV])
    gogMGnotInChosen = gogMG.sort_values('error_over_parallax').head(NgogNotInChosen)
        
    gogChosen = pd.concat([gogRVchosen,gogMGnotInChosen.sort_values('error_over_parallax').tail(NdiffRV)],axis=0).reset_index(drop=True)
    
    Ndiff = gdr3MG.shape[0]-gogChosen.shape[0]  
    
    if Ndiff > 0:
        newStars,listStars = randomStarGenerator(gogMS,gogChosen,gdr3MG,35,20,18,[MGmin,MGmax],['pmra','pmdec'])
        newStars           = pd.concat([pd.DataFrame({'source_id':np.arange(minID+1,newStars.shape[0]+minID+1,1)}),newStars],axis=1)
        minID              = newStars.source_id.max()
        
        gogChosen = pd.concat([gogChosen,newStars],axis=0).reset_index(drop=True)
        
    # We add the information to both fields respectively
    gogFinalField = pd.concat([gogFinalField,gogChosen],axis=0).reset_index(drop=True)
    
    print(gogFinalField[(gogFinalField.set_index(gogFinalField.source_id).index.duplicated()) & (gogFinalField.source_id>0)].MG.min())
    
    # This is GOG final field with RVs
    gogFinalField.to_csv(common_path+'gogFinalField.csv',index=False)





















