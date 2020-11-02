def sectorMomentumSP500(features, fieldWanted='bb_live',sectorN=30,stockN=60):  #sectorN and stockN is window size for the rolling average of monthly returns of the sectors and the stocks. Defult values are approximatly/crudely optimized 
    
    px = features.subset(fields=fieldWanted, asDataFeatures=True) 
    
    # Selection Inclusion
    inclusionMatrix = getTickersSP500(ticker=features.tickers, startDate=features.startDate, endDate=features.endDate, asMatrix=True)
    
     # Rebalancing at the end of each month
    selection = px.changeFreq('monthly').copy()
    selection.columns = features.tickers
    selection.iloc[:, :] = 0.0 #initialise
    rebalDates = selection.index.tolist()
    
    #Initializing Sector Dictionaries, Keys represent each of the 11 GICS Secotrs
    
    def sectorlist():
        gics = getGICSDescription()
        fins = gics[gics.sector=='Financials'].industry.unique().tolist()
        comms= gics[gics.sector=='Communication Services'].industry.unique().tolist()
        disc= gics[gics.sector=='Consumer Discretionary'].industry.unique().tolist()
        stapl= gics[gics.sector=='Consumer Staples'].industry.unique().tolist()
        eng= gics[gics.sector=='Energy'].industry.unique().tolist()
        health= gics[gics.sector=='Health Care'].industry.unique().tolist()
        indust= gics[gics.sector=='Industrials'].industry.unique().tolist()
        IT= gics[gics.sector=='Information Technology'].industry.unique().tolist()
        matrl= gics[gics.sector=='Materials'].industry.unique().tolist()
        util= gics[gics.sector=='Utilities'].industry.unique().tolist()
        real= gics[gics.sector=='Real Estate'].industry.unique().tolist()
        return dict(zip(['fins','comms','disc','stapl','eng','health','indust','IT','matrl','util','real'],[fins,comms,disc,stapl,eng,health,indust,IT,matrl,util,real]))
    
    sectors=sectorlist() #Dictionary of industries for each sector (for gettickersSP500)
    sectordata=dict(zip(['fins','comms','disc','stapl','eng','health','indust','IT','matrl','util','real'],[0,0,0,0,0,0,0,0,0,0,0])) #Price Dataframe subset by sector (For Sector Momentum)
    stockdata=dict(zip(['fins','comms','disc','stapl','eng','health','indust','IT','matrl','util','real'],[0,0,0,0,0,0,0,0,0,0,0])) #Price Dataframe subset by sector  (For Stock Momentum)
    
    
    sectormomentum=dict(zip(['fins','comms','disc','stapl','eng','health','indust','IT','matrl','util','real'],[0,0,0,0,0,0,0,0,0,0,0])) #Momentum of each Sector (Geometric Average Returns)
    sectoralloc=dict(zip(['fins','comms','disc','stapl','eng','health','indust','IT','matrl','util','real'],[0,0,0,0,0,0,0,0,0,0,0])) #Stock Allocation for each sector (How many stocks invested in per sector)
    sectorinclusion=dict(zip(['fins','comms','disc','stapl','eng','health','indust','IT','matrl','util','real'],[0,0,0,0,0,0,0,0,0,0,0])) #Inclusion Matrix for each Sector
    
    
    allocratio=[9,8,7,6,5,5,4,3,2,1,0] #Allocation Ratio (TODO: Make proportional to difference in returns) <--
    
    
    universeStockdata=px.pxs.pct_change(30).rolling(sectorN).mean().replace(np.nan,0)
    print(type(universeStockdata))
    for key in sectors.keys():
        sectordata[key]=px.subset(tickers=getTickersSP500(industry=sectors[key]).ticker.unique().tolist(),asDataFeatures=True) #Price Dataframe subset by sector
        stockdata[key]=sectordata[key]
        sectorinclusion[key]=inclusionMatrix.loc[:,getTickersSP500(industry=sectors[key]).ticker.unique().tolist()] #Inclusion Matrix for each Sector
        sectordata[key]=sectordata[key].pxs.pct_change(30).rolling(sectorN).mean().replace(np.nan,0)  #Converting to moving average of monthly returns
        stockdata[key]=stockdata[key].pxs.pct_change(30).rolling(stockN).mean().replace(np.nan,0)
        
        
    def tkrsget(inclMatrix,date): #Function to get available tickers for a given inclusion matrix and date
        includedAtDate = (inclMatrix.loc[date,:]>0).values
        tkrsAtDate = [tk for tk, incl in zip(inclMatrix.loc[date,:].index.tolist(), includedAtDate) if incl]
        return tkrsAtDate
    
    rebalLength=len(rebalDates)
    i=0
    while i<rebalLength:
        
        date=rebalDates[i]
        includedAtDate = (inclusionMatrix.loc[date,:]>0.0).values
        tkrsAtDate = [tk for tk, incl in zip(inclusionMatrix.loc[date,:].index.tolist(), includedAtDate) if incl] 
        for key in sectors.keys(): #Finding Momentum of Each sector
            temp=sectordata[key].reindex(columns=tkrsget(sectorinclusion[key],date)).iloc[sectordata[key].index.get_loc(date,method='ffill')] #Filtering out un-availbe tickers, if rebalancing date is on weekend, look at data from nearest previous date   
            sectormomentum[key]=scipy.stats.gmean([x+1 for x in temp]) #finidng geometric mean returns of all stocks for each sector 
            sectormomentumsort = sorted(sectormomentum.items(), key=lambda x: x[1], reverse=True) #Returning sorted list of each sector and its mean return 
        for j in range(len(sectormomentumsort)):
            sectoralloc[sectormomentumsort[j][0]]=allocratio[j] #Allocating number of stocks to pick for each sector
        
        for key in sectoralloc.keys():
            temp=stockdata[key].reindex(columns=tkrsget(sectorinclusion[key],date)).iloc[stockdata[key].index.get_loc(date,method='ffill')]
            selected=temp.sort_values(ascending=False).iloc[:sectoralloc[key]].index.tolist() #Chooses best momentum stocks (Number of stocks chosen dependent on allocation)
            selection.loc[date,selected]=0.02 #Asigns equal weight
        
        stocksum=selection.loc[date,:].sum()
        #print(stocksum)
        while stocksum<0.99:  #If stocks are missing for some reason: Add required amount of stocks to selected to get to 50 stocks
            if i==0: break
            stocksum=selection.loc[date,:].sum()
            extrastock=int(round((1.0-stocksum)/0.02))
            temp=universeStockdata.reindex(columns=tkrsget(inclusionMatrix,date)).iloc[universeStockdata.index.get_loc(date,method='ffill')]
            tiks=tkrsget(selection,date)   #Gets Selected tickers
            temp.drop(tiks,inplace=True)   #drops Them
            extraselected=temp.sort_values(ascending=False).iloc[:extrastock].index.tolist()
            selection.loc[date,extraselected]=0.02
            stocksum=selection.loc[date,:].sum()
            print(i)
            
        
        if i+1<len(rebalDates): #If not last reblancing month
            tiks=tkrsget(selection,date)#Gets Selected Tickers
            tempselect=selection.loc[date,tiks].index.tolist()
            temp=inclusionMatrix.loc[date:rebalDates[i+1],tempselect] #inclusion matrix of selected stocks inbetween rebalancing months
            temp=temp[:-1] #(Dropping next months rebalancing day)
            
            
            if temp.eq(0.0).any().any(): #If any zeros found in montly inclusion matrix
                for index,row in temp.iterrows(): #Find the row/date with the zero
                    if row.eq(0.0).any():
                        break
                rebalDates.insert(i+1,index) #Insert new rebalancing day into the list (above rebalacing code will occur on this date in next for loop itteration )
                selection.loc[index]=0.0 #adding new selection row at new rebalacing date at bottom of dataframe
                selection.sort_index(inplace=True) #Re-sorting the index by date
                #print(index)<---UnComment this to see In-Between Rebalancing Dates
                #After this date Loop, the next date in the loop will be on the day the stock drops out of the index  
        i+=1
        rebalLength=len(rebalDates)
       
        
    
    
    
    #remove all zeros
    notAllZeros = (selection.sum(axis=1)!=0.0).values
    idxToDrop = [rebalDates[i] for i in range(len(notAllZeros)) if notAllZeros[i]==False]
    selection.drop(idxToDrop, axis=0, inplace=True)
        
    
    return selection
