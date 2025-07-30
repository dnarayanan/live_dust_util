import h5py
import numpy as np

class SnapshotContainer(object):
	"""
	Container of some important fields extracted from Arepo_dust simulation snapshots
	To show list of fields:
		use show_field_list()

	Parameter:
		snap_no: <int> No. of the input snapshot or <str> to load the sub-snapshot
		snap_dir: <str> directory name of the input snapshot; default = '.'
		snap_pref: <str> prefix of the input snapshot; default = 'snapshot'
	"""
	# URGENT TODO: ad hoc loading // Think about filtered container or full container // Memory-speed tradeoff
    # I seriously don't want users to change those fields
	# TODO: support written-in-parallel snapshot files (individual or combined)
	__field_list = [
					"PartType0/Density",
	                "PartType0/Masses",
					"PartType0/Coordinates",
					"PartType0/GFM_Metallicity",
					"PartType0/GFM_Metals",
					"PartType0/MolecularHFrac",
					"PartType0/StarFormationRate",
		            "PartType3/Dust_NumGrains",
		            "PartType3/Dust_DustDensity",
					"PartType3/Masses",
					"PartType3/Coordinates",
					"PartType3/Dust_MetalFractions",
					"PartType4/Masses",
					"PartType4/Coordinates",
					"PartType4/SNIaNumber",
					"PartType4/SNIINumber"
				   ]
	_part_types = ['PartType0', 'PartType3', 'PartType4']

	def __init__(self, snap_no, snap_dir = '.', snap_pref = 'snapshot'):
		self.dataset = dict.fromkeys(SnapshotContainer.__field_list, None)
		snap_no = str(snap_no).zfill(3)
		snap = h5py.File(f"{snap_dir}/{snap_pref}_{snap_no}.hdf5", 'r')
		for key in self.dataset.keys():
			self.dataset[key] = snap[key][()]
		# used to compute filters
		self.p_c, self.r_s, self.r_e, self.lz = np.array([np.nan,np.nan,np.nan]), np.nan, np.nan, np.array([0.,0.,1.])
		self.filt = dict.fromkeys(SnapshotContainer._part_types, None)
			
	def show_field_list(self):
		print(list(self.dataset.keys()))

	def compute_filter(self, p_c, r_s, r_e, lz):
		"""
		compute filters to select the 
		Parameters:
			p_c: <ndarray[3], dtype = float64> center to compute radii
			r_s: <float32> lower bound of radius interval
			r_e: <float32> upper bound of radius interval
			lz: <ndarray[3], dtype = float64> direction of angular momentum, default [0, 0, 1]
		return: 
			dict('PartType{i}': filter_PartType{i})
		"""
		if (len(p_c) == 0) or (r_s == None) or (r_e == None):
			# essentially no filter applied
			for part_type in SnapshotContainer._part_types:
				self.filt[part_type] = np.where(self.dataset[part_type + '/Masses'] > 0)
			return self.filt
		if ( np.all((np.allclose(self.p_c, p_c), np.isclose(self.r_s, r_s), np.isclose(self.r_e, r_e))) ): 
			# no need to repeat the computation
			# TODO: nan != nan, so the risk of invalid input that crashes the script is very high! input examination
			return self.filt

		self.p_c, self.r_s, self.r_e = p_c, r_s, r_e
		for part_type in SnapshotContainer._part_types:
			x = self.dataset[part_type + "/Coordinates"][:,0] - p_c[0]
			y = self.dataset[part_type + "/Coordinates"][:,1] - p_c[1]
			z = self.dataset[part_type + "/Coordinates"][:,2] - p_c[2]
			r_s2 = r_s**2
			r_e2 = r_e**2
			r2 = x**2 + y**2 + z**2
			self.filt[part_type] = np.where((r2 >= r_s2) & (r2 < r_e2))
		return self.filt
