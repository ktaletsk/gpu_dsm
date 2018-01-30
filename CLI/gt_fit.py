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
    return np.dot(np.exp(-time/lambdaArr), gArr)*GN0

def log_Gt_MMM(time,params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    #lambdaArr=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(tfinal))
    #gArr = params/np.sum(params)
    return logsumexp(-time/lambdaArr, b=gArr*GN0)

#Vectorize function fdt and log_fdt
Gt_MMM_vec=np.vectorize(Gt_MMM, excluded=['params'])
log_Gt_MMM_vec=np.vectorize(log_Gt_MMM, excluded=['params'])

def Gp_MMM(omega, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])

    return np.sum((gArr * lambdaArr**2 * omega**2)/(1 + lambdaArr**2 * omega**2))*GN0

def Gdp_MMM(omega, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])

    return np.sum((gArr * lambdaArr * omega)/(1 + lambdaArr**2 * omega**2))*GN0

#Vectorize function Gp and Gdp
Gp_MMM_vec=np.vectorize(Gp_MMM, excluded=['params'])
Gdp_MMM_vec=np.vectorize(Gdp_MMM, excluded=['params'])

def gt_fit():
    global x
    global y
    global GN0

    with open('G.dat') as f:
        lines = f.readlines()
        x = np.array([float(line.split()[0]) for line in lines])
        y = np.array([float(line.split()[1]) for line in lines])
    GN0=y[0]
    cutoff = min(np.argmax(y[1:]-y[:-1]>0), np.argmax(y<0), find_nearest(y,0.01*GN0))

    x=x[:cutoff]
    y=y[:cutoff]
    tfinal=x[-1]
    tstart=x[1]

    #Define residuals
    def residuals_Gt_MMM(param):
        return Gt_MMM_vec(time=x, params=param)-y

    #Define log-residuals
    def residuals_log_Gt_MMM(param):
        if np.any(Gt_MMM_vec(time=x[:-1], params=param) < 0):
            return np.full(x[:-1].shape,1e8) #Penalty for negative f_d(t)
        else:
            return log_Gt_MMM_vec(time=x, params=param)-np.log(y)

    #Define Mean-Squared Error
    def MSE_MMM(param):
        return np.dot(residuals_Gt_MMM(param),residuals_Gt_MMM(param))/np.size(x)

    def log_MSE_MMM(param):
        return np.dot(residuals_log_Gt_MMM(param),residuals_log_Gt_MMM(param))/np.size(x)

    fits_1 = [] #output of fitting function for all tested numbers of modes
    successful_fits_1 = [] #number of modes for successful fits
    for nmodes in range(2, 15):
        lambdaArrInit=np.e**(np.log(tstart)+(np.array(range(nmodes), float))/(nmodes-1)*(np.log(tfinal)-np.log(tstart)))
        fit = np.linalg.lstsq(np.exp(-np.outer(x,1.0/lambdaArrInit)), y)[0]
        fits_1.append(fit)
        #if not np.any(Gt_MMM_vec(time=x, params=np.append(lambdaArrInit, fit)) < 0):
        if not np.any(fit < 0):
            successful_fits_1.append(nmodes)
            print nmodes, fit, MSE_MMM(np.append(lambdaArrInit, fit))
    print(successful_fits_1)

    fits_2 = [] #output of fitting function for all tested numbers of modes
    min_log_SME = log_MSE_MMM(fits_1[successful_fits_1[0]-2])
    best_nmodes = successful_fits_1[0]

    for i in successful_fits_1:
        fit = fits_1[i-2]
        nmodes = i
        lambdaArrInit=np.e**(np.log(tstart)+(np.array(range(nmodes), float))/(nmodes-1)*(np.log(tfinal)-np.log(tstart)))
        print('nmodes\t{0}'.format(i))

        fit2 = least_squares(residuals_log_Gt_MMM, np.append(lambdaArrInit, fit), xtol=1e-14, ftol=1e-14)
        fits_2.append(fit2)

        print('First fit log-MSE\t{0}'.format(log_MSE_MMM(np.append(lambdaArrInit, fit))))

        if fit2.success:
            weights = np.split(fit2.x, 2)[1]/np.sum(np.split(fit2.x, 2)[1])
            if log_MSE_MMM(fit2.x)<min_log_SME and not np.any(weights<0):
                min_log_SME = log_MSE_MMM(fit2.x)
                best_fit = fit2
                best_nmodes = i
            print('Second fit log-MSE\t{0}'.format(log_MSE_MMM(fit2.x)))
            print(fit2.message)
            print('Weights\t{0}'.format(weights))

        print(' ')

    fit2 = best_fit
    li=np.split(fit2.x,2)[0]
    gi=np.split(fit2.x,2)[1]/np.sum(np.split(fit2.x,2)[1])
    result=zip(li, gi)
    f = open('gt_MMM_fit.dat','w')
    f.write(str(best_nmodes))
    for i in result:
        f.write('\n'+str(i[0])+'\t'+str(i[1]))
    f.close()

    return result

if __name__== "__main__":
    gt_fit()