import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def _match_within_group(x1,x2,metric='mahalanobis',VI=None,p=None):
    """
    returns the indexto match x2 to x1 and some other stuff
    """
    pairwise = cdist(x1,x2,metric=metric,VI=None,p=None)
    idx, dist = get_nearest(pairwise)
    #idx = idx[dist<cutoff] #for doing cutoffs
    
    return idx,dist,pairwise

def match_within_group(indep,dep,matching_cols,metric='mahalanobis',VI=None,p=None):
    
    """
    do 1 to 1 matching with with the given metric
    if you want a VI for Mahal dist to the be same across groups then precalculate adn give
    as an argument to this function
    if indep and dep have different sizes this will result in the lower number of samples
    
    
    could modify to return both of the matchign columns but not implementing now
    
    returns
    
    """
    i_arr = indep[matching_cols].values
    i_dat = indep.drop(matching_cols,axis=1)
    
    d_arr = dep[matching_cols].values
    d_dat = dep.drop(matching_cols,axis=1)
        
    if i_arr.shape[0]<d_arr.shape[0]:
        idx,dist,_ = _match_within_group(i_arr,d_arr,metric=metric,VI=VI,p=p)
        new_cols = np.setdiff1d(d_dat.columns,i_dat.columns)
        
        for i in new_cols:
            i_dat[new_cols]=d_dat[i].values[idx]
        return i_dat
    else:
        idx,dist,_ = _match_within_group(d_arr,i_arr,metric=metric,VI=VI,p=p)
        new_cols = np.setdiff1d(i_dat.columns,d_dat.columns)
        for i,v in enumerate(new_cols):
            d_dat[new_cols[i]]=i_dat[v].values[idx]
        return d_dat
