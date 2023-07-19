import pandas as pd
import math
import numpy as np
import warnings
from scipy import interpolate as interp

warnings.filterwarnings("ignore")


#%%


# This function return the probability value from the analytic function of the IMF defined by Chabrier model for masses <= 1 and by 
# Salpeter for masses > 1.
def analytic_imf0(Mmean,sc,x):
    if x <= 0:
      result = 1.03*(1/(math.sqrt(2*math.pi*sc**2)))*math.exp(-((x-math.log(Mmean,10))**2)/(2*sc**2))
      return result
    else:
      result = 1.03*(1/(math.sqrt(2*math.pi*sc**2)))*(1/10**(1.35*x))*math.exp(-(math.log(Mmean,10)**2)/(2*sc**2))
      return result


#%%


# This function uses analytic_imf0 to build a numerical IMF between xmin and xmax (with x=log(M)).
def numerical_imf(N,n,dx,xmin,xmax):
    analytic_imf = np.vectorize(analytic_imf0)
    x_chabrier   = np.arange(xmin,0,dx)
    x_salpeter   = np.arange(0,xmax,dx)
    n_chabrier   = np.round(N*analytic_imf(0.3,0.57,x_chabrier)*dx)
    n_salpeter   = np.round(N*analytic_imf(0.3,0.57,x_salpeter)*dx)
    numeric_imf  = np.ones(int(n_chabrier[0]))*x_chabrier[0]
    
    for i in range(1,len(x_chabrier)):
        numericIMF  = np.ones(int(n_chabrier[i]))*x_chabrier[i]
        numeric_imf = np.concatenate((numeric_imf,numericIMF))
    for i in range(1,len(x_salpeter)):
        numericIMF  = np.ones(int(n_salpeter[i]))*x_salpeter[i]
        numeric_imf = np.concatenate((numeric_imf,numericIMF))
    
    imf_result = pd.DataFrame(10**np.random.choice(numeric_imf,n,replace=False))
    imf_result.columns = ["mass"]
    
    return imf_result


#%%


def imfNormFinder(data,logMassMin,logMassMax,dm,analytic_imf):
    massRange       = np.arange(logMassMin,logMassMax,dm)
    
    return data.shape[0]/(np.sum(analytic_imf(0.3,0.57,massRange))*dm)

        
#%%


# The following function join Baraffe and PARSEC data, remove the post Main Sequence data and remove the faintest part of the DCM
# and finally returns a numerical isochrone of age age based on PARSEC and Baraffe models for the Main Sequence and pre Main Sequence
# stages.

def baraffe_PARSEC_MSPMS_data(age):
    # basic limits definition for isochrone building
    masslim         = 0.75                      # this mass defines the limit between PARSEC and Baraffe models
    color_limits    = [-2,6]                    # This color limits define the range of color of the isochrones
    mag_max         = 17                        # This magnitude define the maximum magnitude of the isochrones
    
    # columns name definition (this is no done for age columns since different scales are used, so this must be change manually)
    massName1       = 'Mini'                    # mass column name of first isochrone table
    massName2       = 'Mass'                    # mass column name of second isochrone table
    magName1        = 'Gmag'                    # magnitude column name of first isochrone table
    magName2        = 'G'                       # magnitude column name of second isochrone table
    colorName11     = 'G_BPmag'                 # first color column name of first isochrone table
    colorName12     = 'G_RPmag'                 # second color column name of first isochrone table
    colorName21     = 'G_BP'                    # first color column name of second isochrone table
    colorName22     = 'G_RP'                    # second color column name of second isochrone table
    
    # Input catalogue paths:
    name_isoc1      = "input_cat/PARSEC.csv"    # Isochrone tables from PARSEC-COLIBRI
    name_isoc2      = "input_cat/baraffe.csv"   # Isochrone tables from Baraffe
    
    # Input catalogues reading:
    cat_isoc1 = pd.read_csv(name_isoc1,delimiter=",")
    cat_isoc2 = pd.read_csv(name_isoc2,delimiter=",")

    iso1 = cat_isoc1[(10**cat_isoc1["logAge"] >= age-0.05e6) & 
                     (10**cat_isoc1["logAge"] <= age+0.05e6) & 
                     (cat_isoc1[massName1] >= masslim) &
                     (cat_isoc1[colorName11]-cat_isoc1[colorName12] >= color_limits[0]) &
                     (cat_isoc1[colorName11]-cat_isoc1[colorName12] <= color_limits[1])]
    
    iso2 = cat_isoc2[(np.abs(cat_isoc2["Age"]*10**6-age) == min(np.abs(cat_isoc2["Age"]*10**6-age))) & 
                     (cat_isoc2[massName2] <= masslim) &
                     (cat_isoc2[colorName21]-cat_isoc2[colorName22] >= color_limits[0]) &
                     (cat_isoc2[colorName21]-cat_isoc2[colorName22] <= color_limits[1])]
    
    # We eliminate the post Main Sequence data from the isochrones by finding the color for which the signe of the derivative of the isochrone
    # trajectory changes from + to -. However, some isochrones (like the 10Myr one) can see their derivative sign change before entering the
    # Main Sequence, so we ignore these particular cases by starting our search from G=1 to lower values of G.
    iso1                = iso1.sort_values(magName1)
    iso1                = iso1[::-1]
    BP_RP1              = (iso1[colorName11].iloc[1]-iso1[colorName12].iloc[1])-(iso1[colorName11].iloc[0]-iso1[colorName12].iloc[0])
    rate_of_change_1    = (iso1[magName1].iloc[1]-iso1[magName1].iloc[0])/BP_RP1
    
    upperIso1 = iso1[iso1.Gmag <= 1]
    i = 0
    while rate_of_change_1 > 0:
        i = i+1
        BP_RP1              = upperIso1[colorName11].iloc[i+1]-upperIso1[colorName11].iloc[i]+upperIso1[colorName12].iloc[i]-upperIso1[colorName12].iloc[i+1]
        rate_of_change_1    = (upperIso1[magName1].iloc[i+1]-upperIso1[magName1].iloc[i])/BP_RP1
    mag_min                 = (upperIso1[magName1].iloc[i+1]+upperIso1[magName1].iloc[i])/2
    
    iso1 = iso1[(iso1[magName1] >= mag_min) &
                (iso1[magName1] <= mag_max)]
    
    iso2 = iso2[(iso2[magName2] >= mag_min) &
                (iso2[magName2] <= mag_max)]
    
    iso1            = iso1[[massName1,magName1,colorName11,colorName12]]
    iso2            = iso2[[massName2,magName2,colorName21,colorName22]]
    iso1.columns    = iso2.columns
    
    iso  = pd.concat([iso1,iso2],axis=0).reset_index(drop=True)

    return iso

#%%


# This function build the mass-luminosity interpolated relationship between a set of given masses.
def massToPhoto_baraffe_PARSEC(masses,inputMassColumn,age):
    # We build the numerical isochrones
    iso     = baraffe_PARSEC_MSPMS_data(age)
    
    # We define the name of the columns (must match the names of the ones from iso)
    massName    = 'Mass'
    colorNames  = ['G_BP','G_RP']
    magName     = 'G'
    
    # Mass-Luminosity Interpolation:
    iso     = iso.sort_values(massName)
    ML      = pd.DataFrame({magName:np.interp(masses[inputMassColumn],iso[massName],iso[magName]), 
                            colorNames[0]:np.interp(masses[inputMassColumn],iso[massName],iso[colorNames[0]]), 
                            colorNames[1]:np.interp(masses[inputMassColumn],iso[massName],iso[colorNames[1]])})
    return ML


#%%
   

# This function build the interpolated functions magnitude(color) and color(magnitude) of an isochrone of age age using
# the PARSEC and Baraffe stellar models through baraffe_PARSEC_MSPMS_data function.
     
def iso_interpolation_generator(age):
    # We build the numerical isochrones
    iso     = baraffe_PARSEC_MSPMS_data(age)
    
    # We define the name of the columns (must match the names of the ones from iso)
    colorNames  = ['G_BP','G_RP']
    magName     = 'G'
    
    colorToMagnitude = interp.interp1d(iso[colorNames[0]]-iso[colorNames[1]],iso[magName])
    magnitudeToColor = interp.interp1d(iso[magName],iso[colorNames[0]]-iso[colorNames[1]])
    
    return colorToMagnitude,magnitudeToColor


#%%


# This function receives a dataframe with data of interest with at least color columns from dataColorNames and a magnitude
# column from dataMagName and select only the data deltaMag close to the isochrone of age age based on the function 
# iso_interpolation_generator
def isochroneSelector(data,age,deltaMag,dataMagName,dataColorNames):
    # We build the numerical isochrones
    colorToMagnitude,magnitudeToColor     = iso_interpolation_generator(age)
    
    # We set the minimum color from which interpolating
    colorLimit = np.max([colorToMagnitude.x.min(),-0.5])
    
    if type(dataColorNames) == str:
        dataToFilter = data[(data[dataColorNames] >= colorLimit) &
                            (data[dataColorNames] <= colorToMagnitude.x.max())]
        dataNoFilter = data[data[dataColorNames] < colorLimit]
        
        dataToFilter = dataToFilter[(dataToFilter[dataMagName] > colorToMagnitude(dataToFilter[dataColorNames].astype('float')) - deltaMag[0]) &
                                    (dataToFilter[dataMagName] < colorToMagnitude(dataToFilter[dataColorNames].astype('float')) + deltaMag[1])]
    
    elif type(dataColorNames) == list:
        dataToFilter = data[(data[dataColorNames[0]]-data[dataColorNames[1]] >= colorLimit) &
                            (data[dataColorNames[0]]-data[dataColorNames[1]] <= colorToMagnitude.x.max())]
        
        dataNoFilter = data[data[dataColorNames[0]]-data[dataColorNames[1]] < colorLimit]
        
        dataToFilter = dataToFilter[(dataToFilter[dataMagName] > colorToMagnitude(dataToFilter[dataColorNames[0]].astype('float')-\
                                                                                  dataToFilter[dataColorNames[1]].astype('float')) -\
                                     deltaMag[0]) &
                                    (dataToFilter[dataMagName] < colorToMagnitude(dataToFilter[dataColorNames[0]].astype('float')-\
                                                                                  dataToFilter[dataColorNames[1]].astype('float')) +\
                                     deltaMag[1])]
            
    finalData = pd.concat([dataToFilter,dataNoFilter],axis=0).reset_index(drop=True)
    
    return finalData


#%%


# This function receive
def curveInterpolator(inputData,columns,s,dCurveParameter):
    inputData = inputData.drop_duplicates(subset=columns,keep='last').reset_index(drop=True)
    dataTointerp = []
    for column in columns:
        dataTointerp += [inputData[column]]
    tck, u = interp.splprep(dataTointerp,s=s)
    
    curveParameterRange = np.arange(0,1+dCurveParameter,dCurveParameter)
    
    interpdCurve = pd.DataFrame(np.stack(interp.splev(curveParameterRange,tck)).T,columns=columns)
    curveParameterRange = pd.DataFrame({'curveParameter':curveParameterRange})
    interpdCurve = pd.concat([curveParameterRange,interpdCurve],axis=1,join='inner')
    return interpdCurve,tck


#%%


# This function estimate the mass from the CMD just by minimizing the distance between the star CMD position and the isochrone
def curveParameterValueFinder(data,columns,curve,curveParameterColumn):
    curveParameterValues = pd.DataFrame({'curveParameter':np.zeros(data.shape[0])})
    for i in np.arange(0,data.shape[0],1):
        distance = (curve[columns[0]]-data[columns[0]].iloc[i])**2
        for j in np.arange(1,len(columns),1):
            distance = distance+(curve[columns[j]]-data[columns[j]].iloc[i])**2
        curveParameterValues.curveParameter.iloc[i] = curve[distance == distance.min()][curveParameterColumn].iloc[0]
    
    return curveParameterValues


#%%


def DCMtoMassRelation(data,colorColumns,magColumn,massColumn,age):
    if type(colorColumns) == list:
        data            = pd.concat([data,pd.DataFrame({'BR':data[colorColumns[0]]-data[colorColumns[1]]})],
                                    axis=1,
                                    join='inner')
        colorColumns    = 'BR'
    
    inputColumns        = ['BR','G','mass']
    inputData           = pd.concat([data[colorColumns],data[magColumn],data[massColumn]],axis=1,join='inner')
    inputData.columns   = inputColumns
    
    
    N                   = int(1e4)
    minMass             = 0.002
    maxMass             = 10
    dMass               = (maxMass-minMass)/N
    s                   = 1e-4
    dCurveParameter     = 1e-5
    syntheticData       = pd.DataFrame({'mass':np.arange(minMass,maxMass,dMass)}).sort_values(by=['mass'],ascending=False).reset_index(drop=True)
    syntheticData       = pd.concat([syntheticData,massToPhoto_baraffe_PARSEC(syntheticData,'mass',age).reset_index(drop=True).astype('float')],axis=1,join='inner')
    syntheticData['BR'] = syntheticData.G_BP-syntheticData.G_RP
    
    curve,tck = curveInterpolator(inputData         = syntheticData,
                                  columns           = inputColumns,
                                  s                 = s,
                                  dCurveParameter   = dCurveParameter)
    
    parameterValues = curveParameterValueFinder(data                 = inputData,
                                                columns              = inputColumns,
                                                curve                = curve,
                                                curveParameterColumn = 'curveParameter')
    
    BR,G,masses = interp.splev(parameterValues,tck)
    masses      = np.concatenate(masses)
    
    interpdMass = pd.DataFrame({'interpolatedMass':np.array(masses)})
    
    return interpdMass


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


# this function computes log(Likelihood)
def GBRmassLikelihood(photoData,photoIsochrone,photoError,nError):
    pGBRm = (((photoData[0]-photoIsochrone[0])**2)/(2*(nError[0]*photoError[0](photoData[1]))**2))+\
            (((photoData[1]-photoIsochrone[1])**2)/(2*(nError[1]*photoError[1](photoData[1]))**2))+\
            np.log(2*np.pi*np.sqrt(photoError[0](photoData[1])*photoError[1](photoData[1]))*nError[0]*nError[1])
    
    return -pGBRm


#%%

# This function computes log(Posterior)
def massGBRposterior(photoData,isochrone,photoError,nError):
    analytic_imf               = np.vectorize(analytic_imf0)
    allLogPriors               = pd.DataFrame({'logp':np.log(analytic_imf(0.3,0.57,np.log10(isochrone.mass)))}) #(isochrone.mass.max()-isochrone.mass.min())/isochrone.shape[0]*np.ones(isochrone.shape[0])
    allLogLikelihoods          = GBRmassLikelihood(photoData,[isochrone.BR,isochrone.G],photoError,nError)
    allLogLiPi                 = allLogPriors.logp+allLogLikelihoods
    
    # To compute log(N)=log(sum{LiPi}) we use logSum()=log(exp(x)+exp(y)) by operating as follow:
    # We define Sk=L0P0+...+LkPk and li=log(LiPi)=log(Li)+log(Pi), then Sk=Sum{exp(li)}. We initialize 
    # log(S1)=logaddexp(l0,l1) and then we iterate by computing log(Sk+1)=logaddexp(log(Sk),lk+1)
    logNormalization           = np.logaddexp(allLogLiPi.iloc[0],allLogLiPi.iloc[1])
    for i in np.arange(2,allLogLiPi.shape[0],1):
        logNormalization       = np.logaddexp(logNormalization,allLogLiPi.iloc[i])
        
    numericPosterior        = pd.DataFrame({'mass':isochrone.mass,'logp':allLogLikelihoods+allLogPriors.logp-logNormalization})
    analyticPosterior       = interp.interp1d(numericPosterior.mass,numericPosterior.logp)
    return numericPosterior,analyticPosterior


#%%


def allMassToGBRposteriorPDF(allPhotoData,age,photoError,dataMGcolumn,dataBRcolumn,nError,idColumn):
    N               = int(1e4)
    minMass         = np.log10(0.002)
    maxMass         = np.log10(10)
    dMass           = (maxMass-minMass)/N
    isochrone       = pd.DataFrame({'mass':10**np.arange(minMass,maxMass,dMass)}).sort_values(by=['mass'],ascending=False).reset_index(drop=True)
    isochrone       = pd.concat([isochrone,
                                 massToPhoto_baraffe_PARSEC(isochrone,'mass',age).reset_index(drop=True).astype('float')],axis=1,join='inner')
    isochrone['BR'] = isochrone.G_BP-isochrone.G_RP
    
    allAnalyticalPosteriorPDF   = {}
    allNumericalPosteriorPDF    = {}
    for i in np.arange(0,allPhotoData.shape[0],1):
        photoData                           = allPhotoData.iloc[i]
        numericPosterior,analyticPosterior  = massGBRposterior(photoData     = [photoData[dataBRcolumn],photoData[dataMGcolumn]], 
                                                               isochrone     = isochrone, 
                                                               photoError    = photoError,
                                                               nError        = nError)
        
        allAnalyticalPosteriorPDF.update({photoData[idColumn]:analyticPosterior})
        allNumericalPosteriorPDF.update({photoData[idColumn]:numericPosterior})
    
    return allNumericalPosteriorPDF,allAnalyticalPosteriorPDF


#%%


def IMFbayesInferer(data,age,photoErrors,MGcolumn,BRcolumn,nError,idColumn,nSampling):
    allNumericMassPDF,allAnalyticMassPDF = allMassToGBRposteriorPDF(allPhotoData = data,
                                                                    age          = age,
                                                                    photoError   = photoErrors,
                                                                    dataMGcolumn = MGcolumn,
                                                                    dataBRcolumn = BRcolumn,
                                                                    nError       = nError,
                                                                    idColumn     = idColumn)
    
    massPDFkeys = list(allAnalyticMassPDF.keys())
    
    infMass = pd.DataFrame(columns=['inferedMass','massErrorMinus','massErrorPlus'])
    
    massErrorMinus = 9999
    massErrorPlus  = 9999
    
    for i in np.arange(0,len(massPDFkeys),1):
        numericMassPDF      = allNumericMassPDF[massPDFkeys[i]]
        numericMassPDF      = numericMassPDF[numericMassPDF.logp.isnull().values == False]
        
        mass                = numericMassPDF[numericMassPDF.logp == numericMassPDF.logp.max()].mass.iloc[0]
        
        if nSampling > 0:
            numericMassPDF.logp = np.exp(numericMassPDF.logp)
            cumulativeMassPDF   = pdfToCumulativePdf(numericMassPDF,
                                                     'mass',
                                                     'logp')
            cumulativeInterp    = interp.interp1d(cumulativeMassPDF.cumulative,
                                                  cumulativeMassPDF.mass)
            interpMasses        = pd.DataFrame({'mass':cumulativeInterp(np.random.uniform(0,1,nSampling))})
            massErrorMinus      = np.mean(mass-interpMasses[interpMasses.mass < mass].mass)
            massErrorPlus       = np.mean(interpMasses[interpMasses.mass > mass].mass-mass)
            
        infMass = pd.concat([infMass,pd.DataFrame({'inferedMass':[mass],
                                                   'massErrorMinus':[massErrorMinus],
                                                   'massErrorPlus':[massErrorPlus]})]).reset_index(drop=True)
    
    return infMass,allNumericMassPDF,allAnalyticMassPDF

    
#%%


# import matplotlib as mpl
# import seaborn as sns
# from matplotlib import pyplot as plt

# sns.set_context("talk")
# mpl.style.use("seaborn")
# sns.set_context("paper",font_scale=1.5)
# sns.set_style("whitegrid")

# N                   = int(1e5)
# minMass             = np.log10(0.002)
# maxMass             = np.log10(10)
# dMass               = (maxMass-minMass)/N
# s                   = 1e-4
# dCurveParameter     = 1e-5
# age                 = 1e7
# iso                 = pd.DataFrame({'mass':10**np.arange(minMass,maxMass,dMass)}).sort_values(by=['mass'],ascending=False).reset_index(drop=True)
# iso                 = pd.concat([iso,
#                                   massToPhoto_baraffe_PARSEC(iso,'mass',age).reset_index(drop=True).astype('float')],axis=1,join='inner')
# iso['BR']           = iso.G_BP-iso.G_RP
# i = 70000
# data = [iso.BR.iloc[i]+0.1,iso.G.iloc[i]+0.5]

# plt.scatter(iso.BR,iso.G,s=0.1)
# plt.scatter(data[0],data[1])
# plt.ylim(15,-2)


# gdr3 = pd.read_csv('builded_or_modified_cat/gdr3MSPMS.csv',usecols=['MG',
#                                                                     'BP_err',
#                                                                     'RP_err',
#                                                                     'MG_err'])

# BR_err = np.sqrt(gdr3.BP_err**2+gdr3.RP_err**2)
# gdr3['BR_err'] = np.sqrt(gdr3.BP_err**2+gdr3.RP_err**2)
# plt.scatter(gdr3.MG,BR_err,s=0.01)


# Gmin = gdr3.MG.min()
# Gmax = gdr3.MG.max()
# N = 30
# MGbins = np.arange(Gmin,Gmax,(Gmax-Gmin)/N)

# MGtoErrToInterpolate = pd.DataFrame()
# for i in np.arange(0,len(MGbins)-1,1):
#     binMin  = MGbins[i]
#     binMax  = MGbins[i+1]
#     gdr3Bin = gdr3[(gdr3.MG >= binMin) & (gdr3.MG < binMax)]
#     gdr3Bin = pd.DataFrame({'MG':[(binMin+binMax)/2],
#                             'MG_err':[gdr3Bin.MG_err.mean()+2*gdr3Bin.MG_err.std()],
#                             'BR_err':[gdr3Bin.BR_err.mean()+2*gdr3Bin.BR_err.std()]})
#     MGtoErrToInterpolate = pd.concat([MGtoErrToInterpolate,gdr3Bin],axis=0).reset_index(drop=True)

# MGtoMGerr = interp.interp1d(MGtoErrToInterpolate.MG,MGtoErrToInterpolate.MG_err)
# MGtoBRerr = interp.interp1d(MGtoErrToInterpolate.MG,MGtoErrToInterpolate.BR_err)

# allMG = np.arange(-1,15,0.01)

# plt.scatter(gdr3.MG,gdr3.MG_err,s=1)
# plt.scatter(gdr3.MG,BR_err,s=0.01)
# plt.scatter(allMG,MGtoMGerr(allMG))
# plt.scatter(allMG,MGtoBRerr(allMG))
# plt.scatter(MGtoErrToInterpolate.MG,MGtoErrToInterpolate.BR_err)
# plt.yscale('log')
# plt.show()

# for i in np.arange(0,100000,1000):
#     dx = 0.1
#     dy = -0.2
#     data = [iso.BR.iloc[i]+dx,iso.G.iloc[i]+dy]
    
    
#     a,b = massGBRposterior(photoData    = data, 
#                             isochrone    = iso, 
#                             photoError   = [MGtoBRerr,MGtoMGerr],
#                             nError = 10)
    
#     plt.scatter(iso.BR,iso.G,s=10,c=np.log10(iso.mass),cmap='rainbow',vmin=np.log10(iso.mass.min()),vmax=np.log10(iso.mass.max()))
#     plt.colorbar()
#     plt.scatter(data[0],data[1],s=100,color='black')
#     plt.arrow(iso.BR.iloc[i],iso.G.iloc[i],dx,dy,color='black')
#     plt.ylim(20,-2)
#     plt.title((a[a.p==a.p.max()].mass.iloc[0]-iso.mass.iloc[i])/iso.mass.iloc[i])
#     plt.show()
#     epsilon = 0.05
    
#     # plt.scatter(a.mass,a.p)
#     # plt.axvline(iso[np.abs(iso.G-data[1]) == np.min(np.abs(iso.G-data[1]))].mass.iloc[0])
#     # plt.xlim(a[a.p==a.p.max()].mass.iloc[0]*(1-epsilon),a[a.p==a.p.max()].mass.iloc[0]*(1+epsilon))
#     # plt.show()

    
