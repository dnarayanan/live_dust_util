import copy
import numpy as np
from .SnapshotContainer import SnapshotContainer
from .utils import IndexUtil
import pdb

class GrainSizeDistribution(object):
    '''
    Contains the grain number count or mass as a function of grain sizes in one galaxy.
    Parameters:
    snap: <SnapshotContainer>
    [a]: <ndarray[], dtype = float64> centers of grain size bins in micron
    [p_c]: <ndarray[3], dtype = float64> center to compute radii in code units
    [r_s]: <float32> lower bound of radius interval in code units
    [r_e]: <float32> upper bound of radius interval in code units
    [lz] :  <ndarray[3], dtype = float64> direction of angular momentum, default [0, 0, 1]
    '''
    
    #class variables
    species_rho = {"Aliphatic C":2.2, "PAH": 2.2, "Silicate":3.3}
    species_rho_ext = {"Aliphatic C":2.2,"PAH":2.2,"Silicate":3.3,"Carbonaceous":3.3}
    species_keys = species_rho.keys()
    species_keys_ext = species_rho_ext.keys()

    def __init__(self,snap,a=10.**np.linspace(-3.4,0,16), p_c = [], r_s = None, r_e = None, lz = np.array([0.,0.,1.])):
        self.set_grain_size_distribution(snap,a,p_c,r_s,r_e,lz)

    def set_grain_size_distribution(self,snap,a,p_c,r_s,r_e,lz):
        '''
        Computes properties related to grain size distributions from a snapshot
        
        Parameters:
        snap: <SnapshotContainer>
        a: <ndarray[], dtype = float64> centers of grain size bins in micron
        p_c: <ndarray[3], dtype = float64> center to compute radii in code units
        r_s:  <float32> lower bound of radius interval in code units
        r_e: <float32> upper bound of radius interval in code units
        lz:  <ndarray[3], dtype = float64> direction of angular momentum, default [0, 0, 1]
        '''


        self.a = a
        if len(self.a > 1):
            self.dloga = np.log10(self.a[1]/self.a[0])

        self.DNSF = dict.fromkeys(GrainSizeDistribution.species_keys_ext,None)
        self.DMSF = copy.deepcopy(self.DNSF)

        nbins = len(self.a)

        filt = snap.compute_filter(p_c,r_s,r_e,lz)['PartType3']
        if (snap.dataset["PartType3/Dust_NumGrains"].shape[1]) >= 3 * nbins:

            self.DNSF['Carbonaceous'] = np.sum(snap.dataset["PartType3/Dust_NumGrains"][filt][:,nbins: 2 * nbins], axis=0)
            self.DNSF['Silicate'] = np.sum(snap.dataset["PartType3/Dust_NumGrains"][filt][:, 0:nbins], axis=0)
            self.DNSF['Aromatic Fraction'] = snap.dataset["PartType3/Dust_NumGrains"][filt][:,2 * nbins::]
            self.DNSF['PAH'] = np.sum(self.DNSF['Aromatic Fraction']*snap.dataset["PartType3/Dust_NumGrains"][filt][:,nbins: 2 * nbins],axis=0)
            self.DNSF['Aliphatic C'] = self.DNSF['Carbonaceous']-self.DNSF['PAH']

            if self.DNSF['Aliphatic C'][0] < 0: pdb.set_trace()
            
            #f_PAH = snap.dataset["PartType3/Dust_NumGrains"][filt][:,-nbins:]
            #self.DNSF["Aliphatic C"] = np.sum((1.0 - f_PAH) * snap.dataset["PartType3/Dust_NumGrains"][filt][:,nbins: 2 * nbins], axis=0)
            #self.DNSF["PAH"] = np.sum(f_PAH * snap.dataset["PartType3/Dust_NumGrains"][filt][:,nbins: 2 * nbins], axis=0)
            #self.DNSF["Carbonaceous"] = np.sum(snap.dataset["PartType3/Dust_NumGrains"][filt][:,nbins: 2 * nbins], axis=0)
            #self.DNSF["Silicate"] = np.sum(snap.dataset["PartType3/Dust_NumGrains"][filt][:, : nbins], axis=0)
        else:
            f_C = snap.dataset["PartType3/Dust_MetalFractions"][filt][: ,IndexUtil.elem_i_C]
            self.DNSF["Aliphatic C"] = np.sum(np.dot(f_C, snap.dataset["PartType3/Dust_NumGrains"][filt][:, :]), axis=0)
            self.DNSF["Silicate"] = np.sum(np.dot((1.0 - f_C), snap.dataset["PartType3/Dust_NumGrains"][filt][:, :]), axis=0)
            self.DNSF["PAH"] = np.sum(0.0 * snap.dataset["PartType3/Dust_NumGrains"][filt][:, :], axis=0)
            self.DNSF["Carbonaceous"] = self.DNSF["Aliphatic C"] # no need to deep copy since they are actually the same
        for key in GrainSizeDistribution.species_keys_ext:
            self.DMSF[key] =  self._from_n_to_m(self.DNSF[key],key)


    def get_grain_size_distribution(self, spe, qtype = "mass"):
        '''
        return the grain number count or massa as a function of grain sizes.

        Parameters:
        spe: <str> species "Aliphatic C", "PAH" or "Silicate"
        qtype: <str> "mass" or "num", default = "mass"
        '''

        
        if spe in GrainSizeDistribution.species_keys_ext:
            if qtype in ["Mass","mass"]:
                return self.DMSF[spe]
            elif qtype in ["Num","num"]:
                return self.DNSF[spe]
            else:
                print("Field type not found! Please make sure:")
                print("field type keyword in", ["mass", "num"])

        else:
            print("Species not found! Please make sure:")
            print("species keyword in",list(GrainSizeDistribution.species_keys_ext))
            return np.array([])



    def compute_small_to_large_ratio(self,size=6e-2,large_is_all_grains = False):
        '''
        Compute small-to-large-grain mass ratio for different grain species
        return: <dict> small-to-large mass ratio
        
        #if large_is_all_grains is set to True, then we compute the large as the mass of all grains

        '''
        i = 0
        stl = np.zeros(len(GrainSizeDistribution.species_keys_ext))
        for key in GrainSizeDistribution.species_keys_ext:
            filt_small = np.where(self.a <= size) # Aoyama+2020

            if large_is_all_grains == False:
                filt_large = np.where(self.a > size)
            else:
                filt_large = np.where(self.a > 0)
            m_small = np.sum(self.DMSF[key][filt_small])
            m_large =  np.sum(self.DMSF[key][filt_large])
            stl[i] = m_small/m_large
            i += 1
            

        return dict(zip(GrainSizeDistribution.species_keys_ext, stl))


    def compute_q_pah(self,size=1.5e-3):
        '''
        Compute q_PAH explicitly (which is really just doing a
        small_to_large_ratio calculation with the PAH species only,
        and ensuring that 'Large' is actually all grains and not just
        large ones)
        '''


        filt_small = np.where(self.a <= size) # Aoyama+2020
        filt_large = np.where(self.a > 0)

        m_small = np.sum(self.DMSF['PAH'][filt_small])


        for counter,key in enumerate(GrainSizeDistribution.species_keys_ext):
            if counter == 0:
                m_large =  np.sum(self.DMSF[key][filt_large])
            else:
                m_large += np.sum(self.DMSF[key][filt_large])
        '''
        if len(filt_small) > 1:
            m_small = np.trapz(self.DMSF['PAH'][filt_small]/self.a[filt_small],self.a[filt_small])
        else:
            m_small = np.sum(self.DMSF['PAH'][filt_small])

            
        for counter,key in enumerate(GrainSizeDistribution.species_keys_ext):
            if counter == 0:
                m_large = np.trapz(self.DMSF[key][filt_large]/self.a[filt_large],self.a[filt_large])
            else:
                m_large += np.trapz(self.DMSF[key][filt_large]/self.a[filt_large],self.a[filt_large])
        '''

        qpah = m_small/m_large
        return qpah,m_small,m_large


        
        

    def compute_abundances(self):

        '''
        compute abundances of different grain species
        return <dict> abundances
        '''


        m_spe =  np.zeros(len(GrainSizeDistribution.species_keys_ext))
        i = 0
        for key in GrainSizeDistribution.species_keys_ext:
            m_spe[i] = np.sum(self.DMSF[key])
            i+=1
        return dict(zip(GrainSizeDistribution.species_keys_ext, m_spe / np.sum(m_spe[:-1])))


    def _from_n_to_m(self, arr, key):
        return arr * 4 * np.pi / 3 * self.a**3 * GrainSizeDistribution.species_rho_ext[key] # cgs
