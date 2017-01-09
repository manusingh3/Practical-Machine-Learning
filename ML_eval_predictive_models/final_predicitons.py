#!/usr/bin/env python

#=======================================================
# COMS W4721
# Final Project - Submission Code
# Team: DAMAGE
# Team Members: Daniel Nachajon, Manu Singh, Gerardo Sierra
#
# NOTE: This code may not necessarily be futurized to Python 3
#=======================================================
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sys import argv, exit, stderr

#------------------------------------------------------------------
# Part 1. Create the Prior Distributions
#
# To help handle categorical data with many potential categories,
# our team cached the prior probabilites. For a given attribute, f, 
# taking values v in 1,...,Nf we compute the probability:
#
# Pr(label = +1 | Attribute f == v) = #{label = +1, Attribute f == v}/#{Attribute == f}
#
# For all the attributes save the explicitly numerical ones.
#------------------------------------------------------------------
def CreatePriors(data):
    label = data.label
    priors = data.drop('label', axis=1)

    P = pd.melt(priors)
    P['dummy'] = 1
    P = P.groupby(['variable','value']).count().reset_index()
    P['Freq'] = P.dummy*1.0/(len(priors))

    P['prior'] = None
    for ind in P.index:
        f = P.ix[ind,'variable']
        v = P.ix[ind,'value']
        if f not in ['v59','v60']:
            pi_fv = np.sum(data[(data[f]==v)]['label'] > 0)*1.0/len(data[(data[f]==v)])
            P.ix[ind,'prior'] = pi_fv

    #Store to overall prior probability        
    dfT = pd.Series({'variable':'TOTAL',
                     'value':'TOTAL',
                     'dummy':len(label),
                     'Freq':1.0,
                     'prior':np.sum(label > 0)*1.0/len(label)})

    P = P.append(dfT,ignore_index=True)
    
    # We apply an expert judgement correction for those fields with too few observations
    for i in P.index:
        if P.ix[i, 'dummy'] <= 100:
            P.ix[i, 'prior'] = np.sum(label > 0)*1.0/len(label)
        
    return P

#------------------------------------------------------------------
# Part 2. Apply the Prior Distributions to transform the data
#
# Having created the prior probabilities for each observed value of
# each categorical we can apply these priors to replace the "hard" label
# with a "soft" assignment. We chose to apply these priors on the basis
# of two criteria:
#     i. Relative Homogeneity of the Attribute: If one value accounts
#        for more than X% of the observations we deem the attribute
#        to have insufficient variation in relation to the label and is removed
#     ii. Maximum Cardinality of the Attribute: If the attribute has more
#        than C distinct values then we replace the values with their priors
#        and convert to a numerical value.
#
# We refer to transforms of the data as follows data_98_4 refers to data
# which has had all columns that are 98% or more homogeneous removed and
# Any column which cardinality greater than 4 replaced with their priors
# Similarly data_100_1000 removes columns which only contain a single value
# and any column with more than 1000 distinct categories replaced with priors
#------------------------------------------------------------------
def FeatureMap(data, priors, fillValue, threshHom=0.98, maxCard=4, numericFields=['v59','v60']):
    #Remove Features which are deemed Homogeneous
    dump0 =list(priors[priors.Freq >= threshHom]['variable'].drop_duplicates())
    processedData = data[list(set(data.columns) - set(dump0))]
    
    #Keep from for Fields which are categorical
    priors = priors[[True if x not in numericFields else False for x in priors.variable]]
    
    #Identify Variables which violate maximum cardinality rule
    cardinality = priors[['variable','value']].groupby('variable').count().reset_index()
    changeVars = list(cardinality[cardinality.value > maxCard]['variable'])
    
    #Apply the Prior Distribution
    for v in changeVars: 
        mapPrior = priors[priors.variable == v][['value','prior']] 
        mapPrior = mapPrior.set_index('value')['prior'].to_dict() 
        processedData.replace({v:mapPrior}, inplace=True)
        
    #Identify Variables for which we need dummies
    catVars = list(cardinality[cardinality.value <= maxCard]['variable'])
    catVars = list(set(catVars).intersection(set(processedData.columns)))
    
    dfDummy = pd.DataFrame()
    for c in catVars:
        dum = pd.get_dummies(processedData[c])
        if (len(dum.columns) == 2) and ('1.0' in dum.columns):
            dum = pd.DataFrame(dum['1.0'], columns=c)
            dfDummy = pd.concat([dfDummy, dum], axis=1)
        else:
            dum = dum.ix[:, dum.columns != dum.columns[-1]]
            dum.columns=["{0}_{1}".format(c, d) for d in dum.columns]
            dfDummy = pd.concat([dfDummy, dum], axis=1)
            
    #Remove the Categorical Variables we are going to replace
    processedData.drop(catVars, axis=1, inplace=True)
    processedData = pd.concat([processedData, dfDummy], axis=1)
    
    #Fill Null Values with the overall mean
    for col in processedData.columns:
        if col != 'label':
            processedData[col].fillna(fillValue, inplace=True)
    
    #Change dtypes for safety
    processedData = processedData.convert_objects(convert_numeric=True)
    return processedData

#------------------------------------------------------------------
# Part 3. Apply our Prior mapping and FeatureMap to create a voting prediction
#------------------------------------------------------------------	
def team_DAMAGE_predictions(DATAFILE, QUIZFILE, OUTPUTFILE):
    """We tacitly assume that the three inputs are full file paths with names and extensions included"""
    data = pd.read_csv(DATAFILE)
    data.columns = ["v{0}".format(c) if c != 'label' else 'label' for c in data.columns]
    quiz = pd.read_csv(QUIZFILE)
    quiz.columns = ["v{0}".format(c) if c != 'label' else 'label' for c in quiz.columns]
    print("Loaded the Data")

    #------------------------------------------------------------------
    # A. Create the Processed Data
    #------------------------------------------------------------------
    priors = CreatePriors(data)
    print("Created Prior Distributions")

    fillValue = priors.ix[priors.variable=='TOTAL','prior'].iloc[0]
    print(fillValue)

    data_98_4 = FeatureMap(data, priors, fillValue, threshHom=0.98, maxCard=4, numericFields=['v59','v60'])
    quiz_98_4 = FeatureMap(quiz, priors, fillValue, threshHom=0.98, maxCard=4, numericFields=['v59','v60'])
    quiz_98_4.fillna(fillValue, inplace=True)
    
    data_98_4 = data_98_4[sorted(data_98_4.columns)]
    quiz_98_4 = quiz_98_4[sorted(quiz_98_4.columns)]
    print("Processed Data: 98, 4")

    data_100_1000 = FeatureMap(data, priors, fillValue, threshHom=1.0, maxCard=1000, numericFields=['v59','v60'])
    quiz_100_1000 = FeatureMap(quiz, priors, fillValue, threshHom=1.0, maxCard=1000, numericFields=['v59','v60'])
    quiz_100_1000.fillna(fillValue, inplace=True)

    #Ensure we have column agreement in this broader case
    outColsQ = list(set(data_100_1000.columns).intersection(set(quiz_100_1000.columns)))
    outColsD = list(outColsQ)
    outColsD.append('label')
    data_100_1000 = data_100_1000[outColsD]
    quiz_100_1000 = quiz_100_1000[outColsQ]

    data_100_1000 = data_100_1000[sorted(data_100_1000.columns)]
    quiz_100_1000 = quiz_100_1000[sorted(quiz_100_1000.columns)]
    print("Processed Data: 100, 1000")
    
    #------------------------------------------------------------------
    # B. Voting Member 1:
    #          Adaboost - 200 Estimates
    #          Weak Learner - Decision Tree(max depth 500)
    #          Data - data_98_4 using RFE to go down to 25 variables from 44
    #------------------------------------------------------------------
    # First we have to select the features
    y = data_98_4.label.values
    X = data_98_4.ix[:,1:].values
    Xt = quiz_98_4.values

    X_train = data_98_4.ix[:(0.7*X.shape[0]),:]
    y_train = X_train.label.values
    X_train = X_train.ix[:,1:]

    # Use RFE to select the best 25 features on a limited version of the model
    modelRFE  = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30), algorithm="SAMME.R", n_estimators=2)
    rfe_25 = RFE(modelRFE,25)
    rfe_25 = rfe_25.fit(X_train, y_train)

    # Apply the support
    X = X[:,rfe_25.support_]
    Xt = Xt[:,rfe_25.support_]

    # Train our model
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=500), algorithm="SAMME.R", n_estimators=200)
    model.fit(X,y)
    v1 = model.predict(Xt)
    print("Created First Voter")

    #------------------------------------------------------------------
    # C. Voting Member 2:
    #          Random Forest - 500 Trees, full depth
    #          Data - data_98_4 using RFE to go down to 30 variables from 44
    #------------------------------------------------------------------
    # First we have to select the features
    y = data_98_4.label.values
    X = data_98_4.ix[:,1:].values
    Xt = quiz_98_4.values

    X_train = data_98_4.ix[:(0.7*X.shape[0]),:]
    y_train = X_train.label.values
    X_train = X_train.ix[:,1:]

    # Use RFE to select the best 25 features on a limited version of the model
    modelRFE  = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30), algorithm="SAMME.R", n_estimators=2)
    rfe_30 = RFE(modelRFE,30)
    rfe_30 = rfe_30.fit(X_train, y_train)

    # Apply the support
    X = X[:,rfe_30.support_]
    Xt = Xt[:,rfe_30.support_]

    # Train our model
    model = RandomForestClassifier(n_estimators = 500)
    model.fit(X,y)
    v2 = model.predict(Xt)
    print("Created Second Voter")

    #------------------------------------------------------------------
    # D. Voting Member 3:
    #          Random Forest - 500 Trees, max depth 200
    #          Data - data_100_1000
    #------------------------------------------------------------------
    # Get the data
    y = data_100_1000.label.values
    X = data_100_1000.ix[:,1:].values
    Xt = quiz_100_1000.values

    # Train our model
    model = RandomForestClassifier(n_estimators = 500, max_depth=200, max_features=1)
    model.fit(X,y)
    v3 = model.predict(Xt)
    print("Created Third Voter")

    #------------------------------------------------------------------
    # E. Let Democracy Reign
    #
    # Our approach voted over three methods that were somewhat linked
    # and all applied trees. The first voter focused on deeply understanding
    # the most reduced set of data and therefore boosted to pay attention
    # to the most important data points. The second voter applied a forest of
    # deep trees over a reduced data set to sample the space. Lastly, the 
    # third voter also used random forests over the broadest data set but
    # chose to limit the depth of each tree to limit the search in another way
    #------------------------------------------------------------------
    v = np.sign(v1 + v2 + v3)
    predictions = pd.DataFrame(data={'Prediction':v,'Id':range(1,len(v)+1)})
    predictions[['Id','Prediction']].to_csv(OUTPUTFILE, sep=',', index=False)
    print("Cached the Predictions")
    return None

if __name__ == '__main__':
    if len(argv) < 4:
        print("Usage: python %s DATAFILE QUIZFILE OUTPUTFILE" % argv[0])
        exit(1)
    team_DAMAGE_predictions(argv[1], argv[2], argv[3])

#DATAFILE = r"C:\Users\Daniel\Documents\Columbia\2016 Spring\COMS W4721\Final Project\00_Inputs\data.csv"
#QUIZFILE = r"C:\Users\Daniel\Documents\Columbia\2016 Spring\COMS W4721\Final Project\00_Inputs\quiz.csv"
#OUTPUTFILE = r"C:\Users\Daniel\Desktop\DAMAGE_predictions.csv"
