import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.misc import logsumexp

x = []
y = []

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def fdt(time, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    return np.dot(np.exp(-time/lambdaArr), gArr)

def log_fdt(time,params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    return logsumexp(-time/lambdaArr, b=gArr)

#Vectorize function fdt and log_fdt
fdtvec=np.vectorize(fdt, excluded=['params'])
logfdtvec=np.vectorize(log_fdt, excluded=['params'])

#Define residuals
def residuals_fdt(param):
    global x
    global y
    return fdtvec(time=x, params=param)-y

def residuals_log_fdt(param):
    #print(logfdtvec(time=x[:-1], params=param)-np.log(y[:-1]))
    if np.any(fdtvec(time=x[:-1], params=param) < 0):
        return np.full(x[:-1].shape,1e8) #Penalty for negative f_d(t)
    else:
        return logfdtvec(time=x[:-1], params=param)-np.log(y[:-1])

#Define Mean-Squared Error
def MSE(param):
    return np.dot(residuals_fdt(param),residuals_fdt(param))/np.size(x)

def log_MSE(param):
    return np.dot(residuals_log_fdt(param),residuals_log_fdt(param))/np.size(x)

def fdt_fit():
    global x
    global y
    with open('fdt.dat') as f:
        lines = f.readlines()
        x = np.array([float(line.split()[0]) for line in lines])
        y = np.array([float(line.split()[1]) for line in lines])

    tfinal=x[-1]

    #Remove all zeros from data
    mask = y!=0
    x=x[mask]
    y=y[mask]

    #Cut data points on the left where they dont change much
    cutoff=find_nearest(x, 1e-2)
    x=x[cutoff:]
    y=y[cutoff:]
    #Subsample data
    x=x[0::10]
    y=y[0::10]

    # First, optimize using standard residuals $y_i-f(x_i)$
    fits_1 = [] #output of fitting function for all tested numbers of modes
    successful_fits_1 = [] #number of modes for successful fits
    for nmodes in range(1, 10):
        lambdaArrInit=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(tfinal))
        gArrInit=np.full(nmodes, 1.0/nmodes)

        fit = least_squares(residuals_fdt, np.append(lambdaArrInit, gArrInit), xtol=1e-15)
        fits_1.append(fit)
        if fit.success and not np.any(fdtvec(time=x, params=fit.x) < 0):
            successful_fits_1.append(nmodes)

    print(successful_fits_1)
    
    fits_2 = [] #output of fitting function for all tested numbers of modes
    min_log_SME = log_MSE(fits_1[successful_fits_1[0]-1].x)
    best_nmodes = successful_fits_1[0]
    for i in successful_fits_1:
        fit = fits_1[i-1]
        print('nmodes\t{0}'.format(i))
        print(fit.message)
        print('Initial guess MSE\t{0}'.format(MSE(np.append(lambdaArrInit, gArrInit))))
        print('Fit MSE\t\t\t{0}'.format(MSE(fit.x)))

        fit2 = least_squares(residuals_log_fdt, fit.x, xtol=1e-14, ftol=1e-14)
        fits_2.append(fit2)

        if fit2.success:
            if log_MSE(fit2.x)<min_log_SME:
                min_log_SME = log_MSE(fit2.x)
                best_fit = fit2
                best_nmodes = i
            print(fit2.message)
            print('First fit log-MSE\t{0}'.format(log_MSE(fit.x)))
            print('Second fit log-MSE\t{0}'.format(log_MSE(fit2.x)))

        print(' ')

    result=np.reshape(best_fit.x,(2,best_nmodes)).T
    print(result)
    return result