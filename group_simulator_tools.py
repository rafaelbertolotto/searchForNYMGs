# We set the path of the (so nicely nice) gaiaunlimited stuff BEFORE importing gaiaunlimited stuff
import os
os.environ['GAIAUNLIMITED_DATADIR'] = '/home/rafaelbertolotto/gaiaunlimited_files'

import numpy as np
import pandas as pd
import warnings
import mass_photometry_tools as massPhoto
from scipy import interpolate
from astropy import coordinates
from astropy.coordinates import ICRS
from astropy.coordinates import Galactic
from astropy.coordinates import LSR
import astropy.units as u
from astropy.coordinates import (CartesianRepresentation,CartesianDifferential)
from astropy.coordinates import SkyCoord

warnings.filterwarnings("ignore")


#%%


# data reading:

# This is the numerical IMF for NYMG generation
numerical_IMF       = pd.read_csv("input_cat/numerical_IMF.csv") 


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


#%%


def pdfToCumulativePdf(pdf,columnX,columnY):
    pdf = pdf.sort_values(columnX)
    normalization = pdf[columnY].sum()
    numericCumulative = pd.DataFrame(columns=[columnX,'cumulative'])
    sumPdf = 0
    for i in np.arange(0,pdf.shape[0],1):
        sumPdf += pdf[columnY].iloc[i]/normalization
        numericCumulative = pd.concat([numericCumulative,pd.DataFrame({columnX:[pdf[columnX].iloc[i]],'cumulative':[sumPdf]})],axis=0).reset_index(drop=True)
    return numericCumulative


#%%


def accumulativeDistributionInterpolator1D(dataToInterpolate,
                                           dBin,
                                           columnBin,
                                           columnY):
    
    bins = np.arange(dataToInterpolate[columnBin].min()-0.5*dBin,
                     dataToInterpolate[columnBin].max()+0.5*dBin,
                     dBin)
    
    allAccumulativeDistributions = {}
    for binind in np.arange(0,len(bins)-1,1):
        binMin                          = round(bins[binind],2)
        binMax                          = round(bins[binind+1],2)
        
        localDataToInterpolate          = dataToInterpolate[(dataToInterpolate[columnBin] >= binMin) &
                                                            (dataToInterpolate[columnBin]  < binMax)]
        
        accumulativeDistribution        = interpolate.interp1d(np.arange(localDataToInterpolate.shape[0]) /\
                                                               (localDataToInterpolate.shape[0] - 1),
                                                               np.sort(localDataToInterpolate[columnY]))
            
        allAccumulativeDistributions.update({binMin: accumulativeDistribution})

    return allAccumulativeDistributions


#%%


def errorDistributor(allAccumulativeDistributions,
                     dBin,
                     dataToTransform,
                     columnBin,
                     columnToTransform,
                     nError):
    
    dataTransformed = pd.DataFrame(columns=dataToTransform.columns)
    for i in np.arange(0,len(allAccumulativeDistributions)-1,1):
        binMin = list(allAccumulativeDistributions.keys())[i]
        binMax = list(allAccumulativeDistributions.keys())[i+1]
        
        dataTransformedBin          =  dataToTransform[(dataToTransform[columnBin] >= binMin) &
                                                       (dataToTransform[columnBin] < binMax)]
        
        accumulativeDistribution    = allAccumulativeDistributions[binMin]
        
        if dataTransformedBin.shape[0] > 0:
            error                   = nError*accumulativeDistribution(np.random.random(dataTransformedBin.shape[0]))
            errors                  = pd.DataFrame(error)
            if errors[errors[0].isnull().values].shape[0] == 0:
                dataTransformedBin[columnToTransform]   =  dataTransformedBin[columnToTransform]+\
                                                           np.random.choice([-1,1],error.shape[0])*error
                                                
        dataTransformed = pd.concat([dataTransformed,dataTransformedBin],axis=0).reset_index(drop=True).astype('float')
        
    return dataTransformed


#%%


def probabilitySFCalculator(data,
                            dr3maps,
                            rvmaps):
    # We apply the RVSSF to calculate probabilities of belonging to the 5d/6d sample_class
    data.pdr3sf = pd.DataFrame(dr3maps.query(SkyCoord(ra     = data.ra.to_numpy()*u.degree, 
                                                      dec    = data.dec.to_numpy()*u.degree, 
                                                      frame  = 'icrs'),
                                             gmag   = data.phot_g_mean_mag))
    
    data.prvssf = pd.DataFrame(rvmaps.query(SkyCoord(ra      = data.ra.to_numpy()*u.degree, 
                                                     dec     = data.dec.to_numpy()*u.degree, 
                                                     frame   = 'icrs'),
                                            g       = data.phot_g_mean_mag,
                                            c       = data.phot_g_mean_mag-data.phot_rp_mean_mag))
    data.p      = pd.DataFrame(data.pdr3sf*data.prvssf)
    
    return data


#%%


# This function build a synthetic NYMG with N stars, standard deviation in cartesian axes stdArray, mean position in phase-
# space meanArray and age Age for the 5d sample_class as well as the 6d sample_class. Kinematic and photometric errors 
# interpolated from the real data are applied in order to return both the real and the observed NYMG.
def groupGenerator(N,
                   stdArray,
                   meanArray,
                   Age,
                   dG,
                   dr3maps,
                   rvmaps,
                   allErrorFunctions,
                   GLimToFilter):
   
    kinematicColumns            = ['X','Y','Z','U','V','W']
    observablesColumns          = ['parallax','radial_velocity','pmra','pmdec']
    
    # We build the real cluster
    trueColumns = ['source_id',
                   'mass',
                   'MG',
                   'BP',
                   'RP',
                   'phot_g_mean_mag',
                   'phot_bp_mean_mag',
                   'phot_rp_mean_mag',
                   'X',
                   'Y',
                   'Z',
                   'U',
                   'V',
                   'W',
                   'parallax',
                   'ra',
                   'dec',
                   'radial_velocity',
                   'pmra',
                   'pmdec',
                   'pdr3sf',
                   'prvssf',
                   'p']
    
    trueCluster = pd.DataFrame(np.zeros([N,len(trueColumns)]),columns=trueColumns)
    
    trueCluster.source_id   = pd.DataFrame({'source_id':(-1*np.arange(1,N+1,1)).astype('int64')})
    trueCluster.mass        = numerical_IMF.sample(N).sort_values(by=['mass'],ascending=False).reset_index(drop=True)
    
    GBR                     = massPhoto.massToPhoto_baraffe_PARSEC(trueCluster,
                                                                   'mass',
                                                                   Age).sort_values(by=['G']).reset_index(drop=True).astype('float')
    
    trueCluster.MG          = GBR.G
    trueCluster.BP          = GBR.G_BP
    trueCluster.RP          = GBR.G_RP
    
    # Real positions and velocities
    for i in np.arange(0,len(kinematicColumns),1):
        trueCluster[kinematicColumns[i]]  = np.random.normal(meanArray[i],stdArray[i],N)
      
    # Real kinematic observables
    Galactic_coord              = coordinates.Galactic(u = trueCluster.X.to_numpy()*u.pc,
                                                       v = trueCluster.Y.to_numpy()*u.pc, 
                                                       w = trueCluster.Z.to_numpy()*u.pc,
                                                       U = trueCluster.U.to_numpy()*u.km/u.s,
                                                       V = trueCluster.V.to_numpy()*u.km/u.s,
                                                       W = trueCluster.W.to_numpy()*u.km/u.s,
                                                       representation_type=CartesianRepresentation,
                                                       differential_type=CartesianDifferential)  
            
    icrs                        = Galactic_coord.transform_to(ICRS)
    trueCluster.parallax        = pd.DataFrame(1000/icrs.distance)
    trueCluster.ra              = pd.DataFrame(icrs.ra)
    trueCluster.dec             = pd.DataFrame(icrs.dec)
    trueCluster.radial_velocity = pd.DataFrame(icrs.radial_velocity)
    trueCluster.pmra            = pd.DataFrame(icrs.pm_ra_cosdec)
    trueCluster.pmdec           = pd.DataFrame(icrs.pm_dec)
    
    lsr                         = icrs.transform_to(LSR())
    trueCluster['pmraLSR']      = pd.DataFrame(lsr.pm_ra_cosdec)
    trueCluster['pmdecLSR']     = pd.DataFrame(lsr.pm_dec)
    
    # Real apparent magnitudes
    magNames = ['phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','G','G_BP','G_RP']
    
    for i in [0,1,2]:
        trueCluster[magNames[i]] = pd.DataFrame({magNames[i]:GBR[magNames[i+3]]-5+5*np.log10(1000/trueCluster.parallax)})
    
    
    # Here starts the building of the observed cluster (real cluster with errors in observables)
    # We start by limiting the cluster to Gaia DR3 interpolated error limits (as a function of G)
    Gmin                    = GLimToFilter[0]+dG
    Gmax                    = GLimToFilter[1]-dG
    
    observedCluster = trueCluster[(trueCluster.phot_g_mean_mag > Gmin) &
                                          (trueCluster.phot_g_mean_mag < Gmax)].reset_index(drop=True)
    
    # We apply the errors to the kniematic observables
    for i in np.arange(0,len(observablesColumns),1):
        observedCluster = errorDistributor(allAccumulativeDistributions    = allErrorFunctions[observablesColumns[i]+'_error'],
                                           dBin                            = dG,
                                           dataToTransform                 = observedCluster,
                                           columnBin                       = 'phot_g_mean_mag',
                                           columnToTransform               = observablesColumns[i],
                                           nError                          = 1) 
        
    # We apply the errors to the photometric observables
    magErrorNames = ['G_err','BP_err','RP_err']
    for i in [0,1,2]:
        if magErrorNames[i] == 'G_err':
            nError = 50
        else:
            nError = 10
        observedCluster = errorDistributor(allAccumulativeDistributions    = allErrorFunctions[magErrorNames[i]],
                                           dBin                            = dG,
                                           dataToTransform                 = observedCluster,
                                           columnBin                       = 'phot_g_mean_mag',
                                           columnToTransform               = magNames[i],
                                           nError                          = nError) 
    
    # We then calculate the observed MG using the observed parallax and observed G
    observedCluster.MG  = observedCluster.phot_g_mean_mag+5-5*np.log10(1000/observedCluster.parallax)
    
    icrs                = coordinates.ICRS(ra              = observedCluster.ra.to_numpy()*u.degree,
                                           dec             = observedCluster.dec.to_numpy()*u.degree,
                                           distance        = (1000/observedCluster.parallax).to_numpy()*u.pc,
                                           pm_ra_cosdec    = observedCluster.pmra.to_numpy()*u.mas/u.yr,
                                           pm_dec          = observedCluster.pmdec.to_numpy()*u.mas/u.yr,
                                           radial_velocity = observedCluster.radial_velocity.to_numpy()*u.km/u.s)

    XYZUVW                                        = icrs.transform_to(Galactic())     
    observedCluster["X"]                          = pd.DataFrame(XYZUVW.cartesian.x)
    observedCluster["Y"]                          = pd.DataFrame(XYZUVW.cartesian.y)
    observedCluster["Z"]                          = pd.DataFrame(XYZUVW.cartesian.z)
    observedCluster["U"]                          = pd.DataFrame(XYZUVW.velocity.d_x)
    observedCluster["V"]                          = pd.DataFrame(XYZUVW.velocity.d_y)
    observedCluster["W"]                          = pd.DataFrame(XYZUVW.velocity.d_z)
    
    lsr                                           = icrs.transform_to(LSR())
    observedCluster['pmraLSR']                    = pd.DataFrame(lsr.pm_ra_cosdec)
    observedCluster['pmdecLSR']                   = pd.DataFrame(lsr.pm_dec)
            
    # We calculate the RVSSF probabilities of belonging to the 5d/6d sample_class for the real and observed cluster
    trueCluster     = probabilitySFCalculator(data          = trueCluster,
                                              dr3maps       = dr3maps,
                                              rvmaps        = rvmaps)
    observedCluster = probabilitySFCalculator(data          = observedCluster,
                                              dr3maps       = dr3maps,
                                              rvmaps        = rvmaps)
        
        
    return {'real':trueCluster.astype('float'),'observed':observedCluster.astype('float')}
    

#%%


def randomObservedGroup(N,
                        Age,
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
                        centers):
    
    
    
    
    
    
    def rvssfSelector(data,fieldData,pcolumn,dG,plim):
        newData = pd.DataFrame(columns=data.columns)
        Gbins   = np.arange(data.phot_g_mean_mag.min()-0.5*dG,data.phot_g_mean_mag.max()+0.5*dG,dG)
        for i in np.arange(0,len(Gbins)-1,1):
            Gmin        = Gbins[i]
            Gmax        = Gbins[i+1]
            
            fieldpmin   = fieldData[(fieldData.phot_g_mean_mag > Gmin) &
                                    (fieldData.phot_g_mean_mag <= Gmax)][pcolumn].min()
            
            dataG       = data[(data.phot_g_mean_mag > Gmin) &
                               (data.phot_g_mean_mag <= Gmax) &
                               (data[pcolumn] > np.max([plim,fieldpmin]))]
                
            newData     = pd.concat([newData,dataG],axis=0).reset_index(drop=True)
        return newData    
    
    
    
    
    
    
    def boxSelector(data,colNames,dx):
        nymg = data[data.source_id < 0]
        for i in np.arange(0,len(colNames),1):
            xmin    = nymg[colNames[i]].min()-dx[i]
            xmax    = nymg[colNames[i]].max()+dx[i]
            data    = data[(data[colNames[i]] >= xmin) &
                           (data[colNames[i]] <= xmax)]
        return data
    
    
    
    
    
    
    col1                = ['U','V','W']
    col2                = ['X','Y','Z']
    
    mean1               = randomUniformSpherical(n       = 1,
                                                 Rmax    = lim1, 
                                                 columns = col1)
    
    for i in np.arange(0,len(col1),1):
        mean1[col1[i]] = mean1[col1[i]]+centers[i]
        
    
    mean2               = randomUniformSpherical(n       = 1,
                                                 Rmax    = lim2, 
                                                 columns = col2)
    
    for i in np.arange(0,len(col2),1):
        mean2[col2[i]] = mean2[col2[i]]+centers[i+len(col1)]
        
    
    realObservedNymg    = groupGenerator(N                          = N,
                                        stdArray                    = std2+std1,
                                        meanArray                   = list(mean2.loc[0])+list(mean1.loc[0]),
                                        Age                         = Age,
                                        dG                          = dG,
                                        dr3maps                     = dr3maps,
                                        rvmaps                      = rvmaps,
                                        allErrorFunctions           = allErrorFunctions,
                                        GLimToFilter                = GLimToFilter)
    
    
    trueNymg            = realObservedNymg['real']
    Nymg                = realObservedNymg['observed']
    
    Nymg                = Nymg.drop('mass',axis=1)
    
    Nymg                = rvssfSelector(data        = Nymg,
                                        fieldData   = field, 
                                        pcolumn     = pcolumn, 
                                        dG          = dG, 
                                        plim        = plim)
    
    Nymg.source_id      = Nymg.source_id.astype(field.source_id.dtype)
    fieldNymg           = pd.concat([field,Nymg],axis=0).reset_index(drop=True)
    
    if sample_class == '6d':
        colNames = col2+col1
    elif sample_class == '5d':
        colNames = col2+['pmra','pmdec']
        
    fieldNymg           = boxSelector(data      = fieldNymg,
                                      colNames  = colNames,
                                      dx        = dxBoxLim)
        
    
    fieldNymg           = fieldNymg.reset_index(drop=True)
    return fieldNymg,trueNymg


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


