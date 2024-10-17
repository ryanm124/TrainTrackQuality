import numpy as np
import awkward as ak
from fxpmath import Fxp

class dataType:
    '''
    Put data into proper format.
    '''
    def __init__(self, data_arrays):
 
        self.init_features = ["trkExtEmuFixed_pt","trkExtEmuFixed_eta","trkExtEmuFixed_phi","trkExtEmuFixed_d0","trkExtEmuFixed_z0","trkExtEmuFixed_chi2rz","trkExtEmuFixed_bendchi2","trkExtEmuFixed_MVA","dvEmuFixed_d_T","dvEmuFixed_R_T","dvEmuFixed_cos_T","dvEmuFixed_del_Z"]
        self.verts = verts(data_arrays, self.init_features, False)
       
    def summarize(self):
 
        print('Vertex info:')
        print('\t',len(self.verts.y),'total vertices')
        print('\t',len(self.verts.y[self.verts.y==0]),'fake vertices (',len(self.verts.y[self.verts.y==0])
              /len(self.verts.y),')')
        print('\t',len(self.verts.y[self.verts.y==1]),'real vertices (',len(self.verts.y[self.verts.y==1])
              /len(self.verts.y),')')
              
        return


class verts:
    
    def __init__(self, data_arrays, feats, match_bool):
        
        # data for ML
        self.tp_match = match_bool
        self.isReal = ak.flatten(data_arrays['dvEmu_isReal'])
        self.R_T = ak.flatten(data_arrays['dvEmu_R_T'])
        self.X_feats = None
        self.X = self.setX(data_arrays, feats)
        self.y = self.setY(data_arrays)
#         self.removeBad()

    def rounder(self, layout, **kwargs):
        if layout.is_numpy:
            return ak.contents.NumpyArray(
                Fxp(layout.data,True,12,4,overflow="saturate",rounding="around")
            )

    def setX(self, arrays, feats):
        
        self.X_feats = []        
        tmp_X = [None]*((len(feats)*2)-4) # add additional features from original ones (nlaymissbeamline)
        
        # loop through features and extract them from array
        jj = 0;
        for ii in range(len(feats)):
            feat = feats[ii]
            if "trkExtEmu" in feat:
                tmp_X[jj] = ak.flatten(arrays[feat][arrays['dvEmu_firstIndexTrk']])
                if "d0" in feat:
                    tmp_X[jj] = tmp_X[jj]*-1
                #tmp_X[jj] = ak.transform(self.rounder,tmp_X[jj])
                self.X_feats.append(feat+"_firstTrk")
                jj+=1;
                tmp_X[jj] = ak.flatten(arrays[feat][arrays['dvEmu_secondIndexTrk']])
                if "d0" in feat:
                    tmp_X[jj] = tmp_X[jj]*-1
            elif "chi2rzdofSum" in feat:
                tmp_X[jj] = ak.flatten(arrays["trkExtEmu_chi2rz"][arrays['dvEmu_firstIndexTrk']] + arrays["trkExtEmu_chi2rz"][arrays['dvEmu_secondIndexTrk']])
            elif "numStubsSum" in feat:
                tmp_X[jj] = ak.flatten(arrays["trkExtEmu_nstub"][arrays['dvEmu_firstIndexTrk']] + arrays["trkExtEmu_nstub"][arrays['dvEmu_secondIndexTrk']])
            elif "chi2rphidofSum" in feat:
                tmp_X[jj] = ak.flatten(arrays["trkExtEmu_chi2rphi"][arrays['dvEmu_firstIndexTrk']] + arrays["trkExtEmu_chi2rphi"][arrays['dvEmu_secondIndexTrk']])
            elif "minD0" in feat:
                tmp_X[jj] = ak.flatten(np.minimum(arrays["trkExtEmu_d0"][arrays['dvEmu_firstIndexTrk']],arrays["trkExtEmu_d0"][arrays['dvEmu_secondIndexTrk']]))
            elif "sumPt" in feat:
                tmp_X[jj] = ak.flatten(arrays["trkExtEmu_pt"][arrays['dvEmu_firstIndexTrk']] + arrays["trkExtEmu_pt"][arrays['dvEmu_secondIndexTrk']])
            else:
                tmp_X[jj] = ak.flatten(arrays[feat])
            #tmp_X[jj] = ak.transform(self.rounder,tmp_X[jj])
            self.X_feats.append(feat)
            jj+=1
            
        # put features in proper format
        X = tmp_X[0]
        for ii in range(1,len(tmp_X)):
            X = np.column_stack((X,tmp_X[ii]))        
        
        return X
    
    def setY(self, arrays):
        
        
        y = ak.flatten(arrays['dvEmu_isReal'])

        # both hard (1) and soft interactions (2) labeled as 1
        #y[y==2] = 1
        
        return y
    
    def set_nlaymissinterior(self, hitpat):
        '''
        Extract number of missed interior layers from hitpattern.
        '''

        nlaymiss = np.zeros(len(hitpat))
        for i in range(len(hitpat)):
            bin_hitpat = np.binary_repr(hitpat[i]) #can set this to fixed width with "width=n"
            bin_hitpat = bin_hitpat.strip('0') #take out all 0 at beginning and end
            nlaymiss[i] = bin_hitpat.count('0') 
    
        return nlaymiss
    
    def set_nlaymissbeamline(self, hitpat):
        '''
        Extract number of missed layers from beamline to last hit layer from hitpattern.
        '''

        nlaymiss = np.zeros(len(hitpat))
        for i in range(len(hitpat)):
            bin_hitpat = np.binary_repr(hitpat[i]) #can set this to fixed width with "width=n"
            bin_hitpat = bin_hitpat.lstrip('0') #take out all leading 0
            nlaymiss[i] = bin_hitpat.count('0') 
    
        return nlaymiss

    def set_nlaymiss(self, hitpat, eta):
        '''
        Extract number of all missed layers from hitpattern.
        '''

        nlaymiss = np.zeros(len(hitpat))
        for i in range(len(hitpat)):
            if eta[i]>1.26 and eta[i]<=1.68:
                w = 7
            else:
                w = 6
            bin_hitpat = np.binary_repr(hitpat[i],width=w) #can set this to fixed width with "width=n"
            nlaymiss[i] = bin_hitpat.count('0') 
    
        return nlaymiss

    def removeBad(self):
        '''
        Take out instances of tracks with nan as a feature.
        '''
        
        bad_idx = np.argwhere(np.isnan(self.X))[0]

        while np.isnan(self.X).any():
            bad_i = np.argwhere(np.isnan(self.X))[0][0]
            self.X = np.delete(self.X,bad_i,0)
            self.y = np.delete(self.y,bad_i,0)
            self.pdgid = np.delete(self.pdgid,bad_i,0)
            self.pt = np.delete(self.pt,bad_i,0)
            self.eta = np.delete(self.eta,bad_i,0)
        
        return    
