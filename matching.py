import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
def get_nearest(pairwise):
    nearest_idx = pairwise.argmin(axis=1)
    dists = pairwise[np.arange(pairwise.shape[0]),nearest_idx]
    return nearest_idx,dists
def _match_within_group(x1,x2,metric='mahalanobis',VI=None,p=None):
    """
    returns the indexto match x2 to x1 and some other stuff
    """
    pairwise = cdist(x1,x2,metric=metric,VI=None,p=None)
    idx, dist = get_nearest(pairwise)
    #idx = idx[dist<cutoff] #for doing cutoffs
    
    return idx,dist,pairwise

# def _match_within_group(onto,source,metric='mahalanobis',VI=None,p=None):
#     """
#     matches the entries in source onto the entries in onto. so the final size is never bigger than the onto dataframe
    
#     """
#     onto = onto.copy()
#     onto_arr = onto.values
#     source_arr = source.values
#     pairwise = cdist(onto_arr,source_arr,metric=metric,VI=None,p=None)
#     idx, dist = get_nearest(pairwise)
    
    
#     new_cols = np.setdiff1d(source.columns,onto.columns)
#     for i,v in enumerate(new_cols):
#         onto[new_cols[i]]=source[v].values[idx]
#     onto['dist']=dist

#     return onto
    

def match_within_group_old(indep,dep,matching_cols,metric='mahalanobis',VI=None,p=None):
    
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
        for i,v in enumerate(new_cols):
            
            i_dat[new_cols[i]]=d_dat[v].values[idx]
        i_dat['dist']=dist

        return i_dat
    else:
        idx,dist,_ = _match_within_group(d_arr,i_arr,metric=metric,VI=VI,p=p)
        new_cols = np.setdiff1d(i_dat.columns,d_dat.columns)
        for i,v in enumerate(new_cols):
            d_dat[new_cols[i]]=i_dat[v].values[idx]
        
        d_dat['dist']=dist
        return d_dat
def gen_fake_data(N=100,n_match=4,countries=[0,1,2],noise_fraction=.2):
    """
    currently assumes gender as a binary as that is what most of maleah's data does
    but i feel bad about baking that in :(
    """
    matching = np.random.randn(N,n_match)
    matching_cols = [f'match{i}' for i in range(n_match)]


    gender = np.random.choice([0,1],N,replace=True).astype(np.bool)
    country = np.random.choice(countries,N,replace=True)

    exact_cols = ['country','gender']
    exacts = np.hstack([country[:,None],gender[:,None]])


    X = np.random.randn(N)

    m = 5
    sigma = 2
    slope_men  =10
    slope_women = 3

#     noise_fraction = .2
    n_men = gender.sum()
    Y = np.zeros_like(X)
    
    Y[gender] = slope_men*X[gender] + np.random.randn(n_men)*sigma
    Y[~gender] = slope_women*X[~gender] + np.random.randn(N-n_men)*sigma

    matching_noise = np.random.randn(N,n_match)*noise_fraction
    indep = pd.DataFrame(np.hstack([matching+matching_noise,exacts,X[:,None]])) #,identity[:,None]
    indep.columns =matching_cols+exact_cols+['indep']#,'identity']


    dep = pd.DataFrame(np.hstack([matching,exacts,Y[:,None]])) #,identity[:,None]
    dep.columns = matching_cols+exact_cols+['dep']#,'identity']


    #shuffle indep

    indep = shuffle(indep)
    
    true_df = pd.DataFrame(np.hstack([matching,exacts,X[:,None],Y[:,None]])  )
    true_df.columns = matching_cols+exact_cols+['indep','dep']
    true_df[exact_cols+['indep','dep']].groupby(exact_cols).mean()
    return indep,dep,true_df,matching_cols,exact_cols
def match_within_group(onto,source,matching_cols,metric='mahalanobis',VI=None,p=None):
    """
    matches the data from source and appends into the onto dataframe
    """
    onto_arr = onto[matching_cols].values
    onto_dat = onto.drop(matching_cols,axis=1)

    source_arr = source[matching_cols].values
    source_dat = source.drop(matching_cols,axis=1)
    
    pairwise = cdist(onto_arr,source_arr,metric=metric,VI=None,p=None)
    idx, dist = get_nearest(pairwise)
    
    new_cols = np.setdiff1d(source_dat.columns,onto_dat.columns)
    for i,v in enumerate(new_cols):
        onto_dat[new_cols[i]]=source_dat[v].values[idx]
    onto_dat['dist']=dist
    return onto_dat
def matching(onto,source,matching_cols,exact_cols=None,metric = 'mahalanobis'):
    if exact_cols is None:
        print("not doing exact matching")
        return match_within_group(onto,source,matching_cols,metric=metric)
    else:
        out = []
        onto_group = indep.groupby(exact_cols)
        source_group = source.groupby(exact_cols)
        for key in source_group.groups.keys():
            source_dat = source.loc[source_group.groups[key],:]
            onto_dat = onto.loc[onto_group.groups[key],:]
            out.append(match_within_group(onto_dat,source_dat,matching_cols,metric=metric))
        
        #https://stackoverflow.com/questions/50501787/python-pandas-user-warning-sorting-because-non-concatenation-axis-is-not-aligne
        return pd.concat(out,sort=False)