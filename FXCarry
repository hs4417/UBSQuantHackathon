import numpy as np
import pandas as pd

def FXCarrySelect(features, n=60, leverage=0.9, signalboost=0.01,alpha=1):
    #Generates two signals Based on returns volatility, and carry/carry volatility
    #alpha is the ratio of the two signals  carry signal/returns signal = alpha
    #signalboost adjusts the leverage of the weighting on inverse volatility
    
    #load features if need be
    if features.pxs is None: features.load()
        
    #Compute momentum
    df = features.subset(fields="bb_live")
    mom = df.pxs / df.pxs.shift(n).values - 1
  
    #Find Returns Conditional Volatility
    logDyReturns=np.log(df.pxs)-np.log(df.pxs.shift(1))
    ewmvol=logDyReturns.ewm(alpha=0.92,adjust=False,ignore_na=False).var(bias=True)
    days=logDyReturns.index.tolist()
 
    
    #Carry Signal
    dfc = features.subset(fields="carry12")
    carryval=dfc.pxs
    carrychange=dfc.pxs.pct_change(1)
    carryvol=carrychange.ewm(alpha=0.94,adjust=False,ignore_na=False).var(bias=True)
    CARRYsignal=carryval.rolling(50,min_periods=10).mean(ignore_na=True)/carryvol #Carry Signal
    carrysignalSum= np.abs(CARRYsignal).apply(lambda x: x.sum() if x.isna().sum()==0.0 else np.nan, axis=1)[:,None]
    carryweight= (CARRYsignal / carrysignalSum).fillna(method='pad') #Normalising signal, so that weights add up to one 
    carryweight=carryweight.clip(lower=0)  #Carry Signal is long only, remove negative weights
    
    # Calculating Returns Signal
    signWithNa = lambda x: np.nan if np.isnan(x) else np.sign(x)
    momSign = mom.applymap(signWithNa)
    carrySign = dfc.pxs.applymap(signWithNa)

    signal=pow((ewmvol*(momSign + carrySign.values))/(2),1) #Returns Signal
   
    
    sumOfSignals = np.abs(signal).apply(lambda x: x.sum() if x.isna().sum()==0.0 else np.nan, axis=1)[:,None] #explicitly remove days with some NaN momentums.
    weights = (signal /sumOfSignals).fillna(method='pad') #Normalising Returns Signal, so that weights add up to one
    
    weights=((alpha*carryweight.values)+weights)/(1+alpha) #Combining Carry and Returns Signal
    
    
    #Signal Booster (Volatility adjusted Leverage)
    sumOfSignals=signal.abs().sum(axis=1) 
    sigmean=sumOfSignals.rolling(60,min_periods=10).mean(skipna=True) #average rolling vol (signal)
    sigstd=sumOfSignals.rolling(60,min_periods=10).std(skipna=True) #std devation of vol
    
    new=(sumOfSignals-sigmean)/sigstd #Finding how far each day volatility is from previous mean volatility
    new=1/new  #Origninal Signal is volatity, we need to use inverse volatility
    ratio=leverage*signalboost #scaling paramaters
    new=(ratio*new)+leverage #Increasing/decreaseing base leverage (daily)
    print(new.mean())
    print(new.std()) #Print stats for daily leverage
    print(new.max())
    print(new.min())
    new=new.clip(lower=0) #Make sure minimum leverage is zero (Dont want to get negative leverage)
    
    weights=weights.multiply((new),axis=0) #Apply New leverage 
    weights = weights.fillna(0)
    weights=weights.clip(lower=-1.0,upper=1.0) #Make sure weights are between 1 and -1
 
    return weights
