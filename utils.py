import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def poopsie_diff_cols(groups, variables, df, genders=None):
    """
    Default: subtracts 'Men' - 'Women' 
    inputs
    ------
    genders : (optional) tuple
        How gender is encoded in the data. e.g. ('men','women')
        default: ('Male','Female')
    outputs:
    """
    grouped = df.groupby(groups)[variables].mean().reset_index()
    if genders is None:
        genders = ('Male','Female')

    men = grouped[grouped['gender']==genders[0]] # WATCH OUT YO' FOOOOOOL
    women = grouped[grouped['gender']==genders[1]]

    del men['gender']
    del women['gender']

    groups.remove('gender')
    d = pd.merge(men,women,on=groups, suffixes=('_m','_w'),how='outer')

    for v in variables:
        d[f'diff_{v}'] = d[v+'_m']-d[v+'_w']
    return d



def scale(df,columns,scaler):
    """
    modifies the columns of df inplace with scaler.fit_transform
    
    example usage:
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    df = #some data you make
    cols = ['column1','var2','poop']
    scale(df,cols,sc)
    """
    df[columns] = pd.DataFrame(scaler.fit_transform(df[columns]),index=df.index,columns=columns)
def inverse_scale(df,columns,scaler):
    '''
    inverts the scale fxn defined above
    '''
    df[columns] = pd.DataFrame(scaler.inverse_transform(df[columns]),index=df.index,columns=columns)
def get_country(df,country):
    """
    input
    -----
    df : dataframe
        should have either a 'country' or 'country_code' column
    country : int or str
        either a string of the country or the country code
    """
    if isinstance(country,str):
        return df[df['country']==country]
    else:
        return df[df['country_code']==country]

def impute_smam(sub,smam,plot=False,title=None):
    """
    impute the smam for the values missing from the subtracted dataframe.
    Performs a linear fit of the smam and then fills in the missing values in sub 
    that don't have an overlapping year in the smam data.
    
    
    inputs
    ------
    sub : dataframe
        lol get rekt nerd (only loser read the docs)
    smam : dataframe
        Data frame with columns: 'year' and 'diff_smam'. this will have dropna called on it
    
    plot : boolean (optional)
        whether to plot the fit
    """
    smam = smam.copy().dropna()
    smam_years = smam['year'].values
    sub_years = sub['year'].values
    
    if smam_years.size==0 or sub_years.size==0:
        raise ValueError('Country has inadequate data!!!')
    overlap = np.intersect1d(sub_years,smam_years)
    predict_years = np.setdiff1d(sub_years,overlap)
    
    # fill in acutal values that overlap
    idx = sub['year'].isin(overlap).values
    sub.loc[idx,'diff_smam'] = smam.loc[smam['year'].isin(overlap),'diff_smam'].values

    # predict and fill in remaining values
    if len(predict_years)>0:
        # sometimes there is already a smam for every year
        reg = LinearRegression()
        reg.fit(smam_years[:,None],smam['diff_smam'])
        pred = reg.predict(predict_years[:,None])
        x = np.linspace(smam_years.min(),smam_years.max())
        if plot:
            if title is not None:
                plt.title(title)
            plt.plot(x,reg.predict(x[:,None]))
            plt.scatter(smam_years,smam['diff_smam'],label='SMAM')
            plt.scatter(predict_years,pred,label='Predicted')
            plt.xlabel('Year')
            plt.ylabel('diff smam')
            plt.legend()
            plt.show()
        idx = sub['year'].isin(predict_years).values
        sub.loc[idx,'diff_smam'] = pred
    return sub

