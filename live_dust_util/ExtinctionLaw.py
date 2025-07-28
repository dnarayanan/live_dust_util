import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from .GrainSizeDistribution import GrainSizeDistribution

class ExtinctionLaw(object):
	"""
	Contains the extinction law as a function of wavelengths,
	derived from grain size distributions of various species

	parameters:
	gsd:      GrainSizeDistribution object
	wave:     <ndarray> wavelengths in micron
	[op_a]:	  <str> filename of grain radii used by optical property tables
	[op_gra]: <str> directory and prefix of optical proprerty table for graphites
	[op_sil]: <str> directory and prefix of optical property table for silicates
	"""
	_x0 = []
	_op = []
	def __init__(self, gsd, wave, op_a=None, op_gra=None, op_sil=None):
		# LEARNING NOTE: __init__ does not return, __new__ i.e. the constructor return
        # I only want users to modify wavelengths by reset_wavelength 
		# so that the extinction law is automatically updated.
		# And I seriously do not want users to modify gsd of an instance.
		# The current strategy may look ugly but I will make it better as I learn more.
		self.__gsd = gsd
		if (len(ExtinctionLaw._x0) == 0) or (len(ExtinctionLaw._op) == 0):
			ExtinctionLaw.set_optical_properties(wave, op_a, op_gra, op_sil)
		self.reset_wavelength(wave)

	@classmethod
	def _Qext_load(cls, op): #this one smells
	# load the optical property table and transform it to log10(1/wavelengths) x log10(Q)
		ntab = len(cls._x0)
		Q_tab = [[] for i in range(ntab)]
		for i in range(ntab):
			num = str(i).zfill(2)
			dat = np.loadtxt(op + num)
			Q_tab[i] = dat[:, (0,1)]
			Q_tab[i][:,1] += dat[:,2]
			Q_tab[i][:,0] = np.log10(1 / Q_tab[i][:,0])
			Q_tab[i][:,1] = np.log10(Q_tab[i][:,1])

		return Q_tab
		
	@classmethod
	def set_optical_properties(cls, wave, op_a, op_gra, op_sil):
		"""
		This method will set up the tables of optical properties as class variables 
		and compute extinction coefficient Qext afterwards

		parameters:
    
		wave:     <ndarray> wavelengths in micron
		[op_a]:   <str> filename of grain radii used by optical property tables
		[op_gra]: <str> filename of optical proprerty table for graphites
		[op_sil]: <str> filename of optical property table for silicates
		"""
		cls._x0 = np.log10(np.loadtxt(op_a))
		cls._op = dict.fromkeys(GrainSizeDistribution.species_keys, None)
		# read original tables into arrays
		cls._op['Aliphatic C']  = cls._Qext_load(op_gra)
		cls._op['PAH'] = cls._Qext_load(op_gra)
		cls._op['Silicate'] = cls._Qext_load(op_sil)


	def reset_wavelength(self, wave): #TODO: only allow change wavelength by this method
		self._wave = wave
		self._Qext = self._Qext_set(self._wave)
		#TODO: consider band pass + transfer function
		self._Qext_V = self._Qext_set(np.array([0.551]))
		self._Qext_B = self._Qext_set(np.array([0.445]))
		self._Qext_bump = self._Qext_set(np.array([0.2175]))
		self._Qext_1500 = self._Qext_set(np.array([0.1500]))
		self._Qext_3000 = self._Qext_set(np.array([0.3000]))
		self._compute_extinction_law()

	def _compute_extinction_law(self):
		self.extinction = np.zeros(len(self._wave))
		self.AV   = 0.
		self.AB   = 0.
		self.A1500 = 0.
		self.A3000 = 0.
		self.Abump = 0.
		self.E_BV = 0.
		self.RV   = 0.
		K = 2.5 * np.log10(np.e) * np.pi
		for key in GrainSizeDistribution.species_keys:
			DNSF = self.__gsd.DNSF[key]
			self.AV += K * np.sum(self.__gsd.a**2*self._Qext_V[key] * DNSF)
			self.AB += K * np.sum(self.__gsd.a**2*self._Qext_B[key] * DNSF)
			self.A1500 += K * np.sum(self.__gsd.a**2*self._Qext_1500[key] * DNSF)
			self.A3000 += K * np.sum(self.__gsd.a**2*self._Qext_3000[key] * DNSF)
			self.Abump += K * np.sum(self.__gsd.a**2*self._Qext_bump[key] * DNSF)
			for i in range(len(self._wave)):
				self.extinction[i] += K * np.sum(self.__gsd.a**2*self._Qext[key][:,i]*DNSF)
		self.extinction /= self.AV
		self.E_BV = self.AB - self.AV
		self.RV = self.AV / self.E_BV
		self.slope_uo = self.A1500 / self.AV
		self.bump = (self.Abump  - 0.33 * self.A1500 - 0.67 * self.A3000) / self.Abump
		self.Abump = self.Abump - 0.33 * self.A1500 - 0.67 * self.A3000

	def get_wavelength(self):
		return self._wave
	
	def get_extinction_law(self):
		return (self._wave, self.extinction)

	def _Q_interp_wl(self,wlen,Q0): #TODO: this looks bad
		Q = interpolate.interp1d(Q0[:,0],Q0[:,1],fill_value='extrapolate')
		Q_wl = Q(wlen)
		return Q_wl

	def _Qext_set(self, wave):
		#wlen: actually 1/wavelength [1/micron] to plot extinction curve
		#x0: array of log10(grain_radii) of the optical property table
		x = np.log10(self.__gsd.a)
		wlen = np.log10(1 / wave)
		ntab = len(ExtinctionLaw._x0)
		dim = (len(x),len(wlen)) # dim: wavelength x grain size
		keys = GrainSizeDistribution.species_keys
		Qext = dict.fromkeys(keys, None)
	
		# extended in wlen
		Q_wl = np.zeros((ntab,dim[1]))
	
		# fully extended table (additionally extended in grain size)
		#Q_2D = dict.fromkeys(keys, np.zeros(dim)) # don't do this! the values of different keys are THE reference of the SAME ndarray
		Q_2D = np.zeros(dim)
	
		# align arrays across different grain sizes by linear interpolation
		for key in keys:
			for i in range(ntab):
				Q_wl[i,:] = self._Q_interp_wl(wlen,ExtinctionLaw._op[key][i])
	
		# bilinear interpolate to obtain the extended table
			for i in range(dim[1]):
				Qx = interpolate.interp1d(ExtinctionLaw._x0,Q_wl[:,i],fill_value='extrapolate')
				Q_2D[:,i]  = Qx(x)
			if(dim[1] == 1):
				Qext[key] = 10**Q_2D.transpose()
			else:
				Qext[key] = 10**Q_2D

		return Qext
