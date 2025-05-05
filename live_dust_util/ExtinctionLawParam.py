import warnings
import numpy as np

def cardelli(wave, tau_v=1, R_v=3.1, **kwargs): #TODO this is not a good practice?
	""" Cardelli, Clayton, and Mathis 1998 Milky Way extinction curve,
    with an update in the near-UV from O'Donnell 1994

    :param wave:
		wavelength in micron
    :param tau_v: (default: 1)
        The optical depth at 5500\AA, used to normalize the
        attenuation curve.

    :param R_v: (default: 3.1)
        The ratio of total selective extinction, parameterizing the
        slope of the attenuation curve.  A_v = R_v * E(B-V)

    :returns tau:
        The optical depth at each wavelength.
	"""
	if (wave < 1e-1).any():
		warnings.warn('Cardelli: extinction not defined (set to zero) below 1000AA')
	mic = wave
	x_sup, x_inf = 10.0, 0.3
	x = 1.0 / mic
	a = np.zeros_like(x)
	b = np.zeros_like(x)
	
	w1 = (x >= 1.1) & (x <= 3.3)  # Optical 0.303 to 0.909 micron
	w2 = (x >= x_inf) & (x < 1.1)  # NIR 0.909 to 3.3 micron
	w3 = (x > 3.3) & (x <= 8)  # UV 0.125 - 0.303 micron
	w4 = (x > 8.0) & (x <= x_sup)  # XUV, 1000 -1250AA
	wsh = x > x_sup
	wlg = x < x_inf
	
	y = x[w1] - 1.82
	a[w1] = (1 + 0.17699 * y - 0.50447 * y**2. - 0.02427 * y**3. +
	         0.72085 * y**4. + 0.01979 * y**5. - 0.77530 * y**6. +
	         0.32999 * y**7.0)
	b[w1] = (1.41338 * y + 2.28305 * y**2. + 1.07233 * y**3. -
	         5.38434 * y**4. - 0.62251 * y**5. + 5.30260 * y**6. -
	         2.09008 * y**7.)
	
	y = x[w2]**1.61
	a[w2] = 0.574 * y
	b[w2] = -0.527 * y
	
	fa = x[w3] * 0.
	fb = x[w3] * 0.
	ou = (x[w3] > 5.9)
	# print(type(ou),ou[0], type(w3))
	
	if ou.any():
		y = x[w3][ou] - 5.9
		fa[ou] = -0.04473 * y**2. - 0.009779 * y**3.
		fb[ou] = 0.2130 * y**2. + 0.1207 * y**3.
	a[w3] = 1.752 - 0.316 * x[w3] - 0.104 / ((x[w3] - 4.67)**2. + 0.341) + fa
	b[w3] = -3.090 + 1.825 * x[w3] + 1.206 / ((x[w3] - 4.62)**2. + 0.263) + fb
	
	y = x[w4] - 8.
	a[w4] = -1.073 - 0.628 * y + 0.137 * y**2. - 0.070 * y**3.
	b[w4] = 13.670 + 4.257 * y - 0.420 * y**2. + 0.374 * y**3.
	
	tau = a + b / R_v
	return tau_v * tau

def smc(wave, tau_v=1, **kwargs):
    """Pei 1992 SMC extinction curve.

    :param wave:
		wavelength in micron
    :param tau_v: (default: 1)
        The optical depth at 5500\AA, used to normalize the
        attenuation curve.

    :returns tau:
        The optical depth at each wavelength.
    """
    if (wave < 1e-1).any():
        warnings.warn('SMC: extinction extrapolation below 1000AA is poor')
    mic = wave
    aa = [185.,  27.,  0.005, 0.010, 0.012, 0.030]
    ll = [0.042, 0.08, 0.22,  9.7,   18.,   25.]
    bb = [90.,   5.50, -1.95, -1.95, -1.80, 0.00]
    nn = [2.0,   4.0,  2.0,   2.0,   2.0,   2.0]

    abs_ab = np.zeros_like(mic)
    norm_v = 0  # hack to go from tau_b to tau_v
    mic_5500 = 5500 * 1e-4

    for i, a in enumerate(aa):
        norm_v += aa[i] / ((mic_5500 / ll[i])**nn[i] +
                           (ll[i] / mic_5500)**nn[i] + bb[i])
        abs_ab += aa[i] / ((mic / ll[i])**nn[i] + (ll[i] / mic)**nn[i] + bb[i])

    return tau_v * (abs_ab / norm_v)



def lmc(wave, tau_v=1, **kwargs):
    """ Pei 1992 LMC extinction curve.

    :param wave:
        The wavelengths at which optical depth estimates are desired.

    :param tau_v: (default: 1)
        The optical depth at 5500\AA, used to normalize the
        attenuation curve.

    :returns tau:
        The optical depth at each wavelength.
    """
    if (wave < 1e3).any():
        warnings.warn('LMC: extinction extrapolation below 1000AA is poor')
    mic = wave
    aa = [175.,  19.,  0.023, 0.005, 0.006, 0.020]
    ll = [0.046, 0.08, 0.22,  9.7,   18.,   25.]
    bb = [90.,   5.50, -1.95, -1.95, -1.80, 0.00]
    nn = [2.0,   4.5,  2.0,   2.0,   2.0,   2.0]

    abs_ab = np.zeros_like(mic)
    #mic * 0.
    norm_v = 0  # hack to go from tau_b to tau_v
    mic_5500 = 5500 * 1e-4

    for i, a in enumerate(aa):
        norm_v += aa[i] / ((mic_5500 / ll[i])**nn[i] +
                           (ll[i] / mic_5500)**nn[i] + bb[i])
        abs_ab += aa[i] / ((mic / ll[i])**nn[i] + (ll[i] / mic)**nn[i] + bb[i])

    return tau_v * (abs_ab / norm_v)

