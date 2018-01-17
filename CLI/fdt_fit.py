import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.misc import logsumexp

x = []
y = []

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_line(xdata, ydata):
    A = np.vstack([xdata, np.ones(len(xdata))]).T
    return np.linalg.lstsq(A, np.log(ydata))[0]

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

def fdt_fit():
    global x
    global y

    with open('fdt.dat') as f:
        lines = f.readlines()
        xx = np.array([float(line.split()[0]) for line in lines])
        yy = np.array([float(line.split()[1]) for line in lines])

    x = 10**(xx/1000.0-10.0)
    y = 1.0-np.cumsum(yy)/np.sum(yy)

    mask = y>0
    x=x[mask]
    y=y[mask]

    tstart=x[0]
    tfinal=x[-1]

    #Fit line in log-linear coordinates
    x_subset=x[find_nearest(x, 0.05*x[-1]):find_nearest(x, 0.65*x[-1])]
    y_subset=y[find_nearest(x, 0.05*x[-1]):find_nearest(x, 0.65*x[-1])]
    m, c = find_line(x_subset, y_subset)
    yline=np.exp(m*x+c)
    gn=np.exp(c)
    taun=-1/m

    # Trim data to $t>0.01 \tau_c $ and subsample it $\times 10$ times to speed-up fitting
    cutoff=find_nearest(x, 1e-2)
    x=x[cutoff:]
    y=y[cutoff:]
    #Subsample data
    x=x[0::10]
    y=y[0::10]

    #Define residuals
    def residuals_fdt(param):
        return fdtvec(time=x, params=param)-y

    #Define log-residuals with penalty
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

    # First, optimize using standard residuals $y_i-f(x_i)$
    fits_1 = [] #output of fitting function for all tested numbers of modes
    successful_fits_1 = [] #number of modes for successful fits
    for nmodes in range(2, 10):
        lambdaArrInit=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(taun))
        #gArrInit=np.full(nmodes, 1.0/nmodes)
        gArrInit = np.append(np.full(nmodes-1, (1.0-gn)/(nmodes-1)),gn)

        print('nmodes\t{0}'.format(nmodes))
        fit = least_squares(residuals_fdt, np.append(lambdaArrInit, gArrInit), xtol=1e-15)
        fits_1.append(fit)

        if fit.success:
            weights = np.split(fit.x, 2)[1]/np.sum(np.split(fit.x, 2)[1])
            if not np.any(fdtvec(time=x, params=fit.x) < 0) and not np.any(weights<0):
                successful_fits_1.append(nmodes)
            print(fit.message)
            print('Initial guess MSE\t{0}'.format(MSE(np.append(lambdaArrInit, gArrInit))))
            print('Fit MSE\t\t\t{0}'.format(MSE(fit.x)))

    print(successful_fits_1)

    fits_2 = [] #output of fitting function for all tested numbers of modes
    min_log_SME = log_MSE(fits_1[successful_fits_1[0]-1].x)
    best_nmodes = successful_fits_1[0]
    for i in successful_fits_1:
        fit = fits_1[i-2]
        print('nmodes\t{0}'.format(i))

        fit2 = least_squares(residuals_log_fdt, fit.x, xtol=1e-14, ftol=1e-14)
        fits_2.append(fit2)

        if fit2.success:
            weights = np.split(fit2.x, 2)[1]/np.sum(np.split(fit2.x, 2)[1])
            if log_MSE(fit2.x)<min_log_SME and not np.any(weights<0):
                min_log_SME = log_MSE(fit2.x)
                best_fit = fit2
                best_nmodes = i
            print(fit2.message)
            print('Weights\t{0}'.format(weights))
            print('First fit log-MSE\t{0}'.format(log_MSE(fit.x)))
            print('Second fit log-MSE\t{0}'.format(log_MSE(fit2.x)))

        print(' ')

    fit2 = best_fit
    li=np.split(fit2.x,2)[0]
    gi=np.split(fit2.x,2)[1]/np.sum(np.split(fit2.x,2)[1])
    result=zip(li, gi)
    f = open('fdt_MMM_fit.dat','w')
    f.write(str(best_nmodes))
    for i in result:
        f.write('\n'+str(i[0])+'\t'+str(i[1]))
    f.close()

    return result

if __name__== "__main__":
    fdt_fit()