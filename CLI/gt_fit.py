import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.misc import logsumexp
import os

x = []
y = []

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def Gt_MMM(time, params):
    #Variable frequencies
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    #Fixed frequencies
    #lambdaArr=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(tfinal))
    #gArr = params/np.sum(params)
    return np.dot(np.exp(-time/lambdaArr), gArr)

def log_Gt_MMM(time,params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    #lambdaArr=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(tfinal))
    #gArr = params/np.sum(params)
    return logsumexp(-time/lambdaArr, b=gArr)

#Vectorize function fdt and log_fdt
Gt_MMM_vec=np.vectorize(Gt_MMM, excluded=['params'])
log_Gt_MMM_vec=np.vectorize(log_Gt_MMM, excluded=['params'])

def Gp_MMM(omega, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])

    return np.sum((gArr * lambdaArr**2 * omega**2)/(1 + lambdaArr**2 * omega**2))

def Gdp_MMM(omega, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])

    return np.sum((gArr * lambdaArr * omega)/(1 + lambdaArr**2 * omega**2))

#Vectorize function Gp and Gdp
Gp_MMM_vec=np.vectorize(Gp_MMM, excluded=['params'])
Gdp_MMM_vec=np.vectorize(Gdp_MMM, excluded=['params'])

def gt_fit():
    global x
    global y

    with open('gt_aver.dat') as f:
        lines = f.readlines()
        x = np.array([float(line.split()[0]) for line in lines])
        y = np.array([float(line.split()[1]) for line in lines])
    GN0=y[0]
    cutoff_list = np.array([np.argmax(y[1:]-y[:-1]>0), np.argmax(y<0), find_nearest(y,0.01*GN0), np.size(x)])

    cutoff = min(cutoff_list[cutoff_list>0])
    x=x[:cutoff]
    y=y[:cutoff]

    tfinal=x[-1]
    tstart=x[1]

    #Save calculated G(t) to file
    gt_result=zip(x, y)
    file = open("gt_result.dat","w")
    for i in gt_result:
        file.write(str(i[0])+'\t'+str(i[1])+'\n')
    file.close()

    #Define residuals
    def residuals_Gt_MMM(param):
        return Gt_MMM_vec(time=x, params=param)*GN0-y

    #Define log-residuals
    def residuals_log_Gt_MMM(param):
        if np.any(Gt_MMM_vec(time=x[:-1], params=param) < 0):
            return np.full(x[:-1].shape,1e8) #Penalty for negative f_d(t)
        else:
            return log_Gt_MMM_vec(time=x, params=param)+np.log(GN0)-np.log(y)

    #Define Mean-Squared Error
    def MSE_MMM(param):
        return np.dot(residuals_Gt_MMM(param),residuals_Gt_MMM(param))/np.size(x)

    def log_MSE_MMM(param):
        return np.dot(residuals_log_Gt_MMM(param),residuals_log_Gt_MMM(param))/np.size(x)

    fits_1 = [] #output of fitting function for all tested numbers of modes
    successful_fits_1 = [] #number of modes for successful fits

    lambdaArrInit=np.e**(np.log(tstart)+(np.array(range(2), float))/(np.log(tfinal)-np.log(tstart)))
    fit = np.linalg.lstsq(np.exp(-np.outer(x,1.0/lambdaArrInit)), y)[0]
    fits_1.append(fit)
    min_log_SME = log_MSE_MMM(np.append(lambdaArrInit, fit))
    best_fit = 2
    print(2, fit, MSE_MMM(np.append(lambdaArrInit, fit)))

    for nmodes in range(3, 15):
        lambdaArrInit=np.e**(np.log(tstart)+(np.array(range(nmodes), float))/(nmodes-1)*(np.log(tfinal)-np.log(tstart)))
        fit = np.linalg.lstsq(np.exp(-np.outer(x,1.0/lambdaArrInit)), y)[0]
        fits_1.append(fit)
        print(nmodes, fit, MSE_MMM(np.append(lambdaArrInit, fit)), log_MSE_MMM(np.append(lambdaArrInit, fit)))

        #if not np.any(Gt_MMM_vec(time=x, params=np.append(lambdaArrInit, fit)) < 0):
        if not np.any(fit < 0) and log_MSE_MMM(np.append(lambdaArrInit, fit))<min_log_SME:
            min_log_SME = log_MSE_MMM(np.append(lambdaArrInit, fit))
            best_fit = nmodes


    fit = fits_1[best_fit-2]
    li=np.e**(np.log(tstart)+(np.array(range(best_fit), float))/(best_fit-1)*(np.log(tfinal)-np.log(tstart)))
    gi=fit
    result=zip(li, gi)
    f = open('gt_MMM_fit.dat','w')
    f.write(str(best_fit))
    for i in result:
        f.write('\n'+str(i[0])+'\t'+str(i[1]))
    f.close()

    return result

if __name__== "__main__":
    gt_fit()
