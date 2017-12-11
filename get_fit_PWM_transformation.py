import pandas
from scipy.interpolate import interp1d
import numpy as np

def get_transformations():
	logKD2PWM = lambda x:x
	PWM2logKD = lambda x:x
	return logKD2PWM, PWM2logKD


def get_spline_transformations():
	fit = pandas.read_csv('./CDR_KD_spline.csv', header=0, index_col=0)

	x = np.array(fit['x']).flatten()
	y = np.array(fit['y']).flatten()
	usethis = np.isfinite(x)
	x = x[usethis]
	y = y[usethis]

	x,ind = np.unique(x,return_index=True)
	y = y[ind]
    
	logKD2PWM = interp1d(np.hstack((-1e16, x, 1e16)),np.hstack((y[0], y, y[-1])))
	PWM2logKD = interp1d(np.hstack((-1e16, y, 1e16)),np.hstack((x[0], x, x[-1])))
	return logKD2PWM, PWM2logKD