import numpy as np

class dataType:
    '''
    Put data into proper format.
    '''
    def __init__(self, data_arrays):
 
        self.init_features = ['trk_phi','trk_eta','trk_z0',
                              'trk_bendchi2','trk_nstub','trk_hitpattern',
                              'trk_chi2rphi','trk_chi2rz']
        self.trks = trks(data_arrays, self.init_features, False)
       
    def summarize(self):
 
        print('L1 tracks info:')
        print('\t',len(self.trks.y),'total tracks')
        print('\t',len(self.trks.y[self.trks.y==0]),'fake tracks (',len(self.trks.y[self.trks.y==0])
              /len(self.trks.y),')')
        print('\t',len(self.trks.y[self.trks.y==1]),'real tracks (',len(self.trks.y[self.trks.y==1])
              /len(self.trks.y),')')
              
        return


class trks:
    
    def __init__(self, data_arrays, feats, match_bool):
        
        # data for ML
        self.tp_match = match_bool
        self.X_feats = None
        self.pt = data_arrays['trk_pt'].flatten()
        self.eta = data_arrays['trk_eta'].flatten()
        self.chi2pdof = data_arrays['trk_chi2'].flatten()/(2*data_arrays['trk_nstub'].flatten()-4)
        self.X = self.setX(data_arrays, feats)
        self.y = self.setY(data_arrays)
        self.pdgid = abs(data_arrays['trk_matchtp_pdgid'].flatten())
#         self.removeBad()
        
    def setX(self, arrays, feats):
        
        self.X_feats = []        
        tmp_X = [None]*(len(feats)) # add additional features from original ones (nlaymissbeamline)
        
        # loop through features and extract them from array
        jj = 0;
        for ii in range(len(feats)):
            feat = feats[ii]
            if 'hitpattern' in feat:
                tmp_X[jj] = self.set_nlaymissinterior(arrays[feat].flatten())
                self.X_feats.append('trk_nlaymissinterior')
                jj+=1
            else:
                tmp_X[jj] = arrays[feat].flatten()
                self.X_feats.append(feat)
                jj+=1
            
        # put features in proper format
        X = tmp_X[0]
        for ii in range(1,len(tmp_X)):
            X = np.column_stack((X,tmp_X[ii]))        
        
        return X
    
    def setY(self, arrays):
        
        if self.tp_match:
            y = np.ones(len(self.X[:,0]))
        else:
            y = arrays['trk_fake'].flatten()

        # both hard (1) and soft interactions (2) labeled as 1
        y[y==2] = 1
        
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