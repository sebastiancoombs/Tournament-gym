
from sklearn.impute import  SimpleImputer,KNNImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures,MinMaxScaler
from scipy.special import softmax


# pipline
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

#classification model
# import xgboost as xgb

#metrics
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,classification_report

#grid search
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold

# optional: pipeline visualiza
# tion
from sklearn import set_config
set_config(display='diagram')
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import time
import requests
import json
import ast
from datetime import datetime
tqdm.pandas()
'''
These functions are to clean the data into the format we want
'''

def drop_columns(data,keep=[],verbose=False):
    drop=['Game','Date','Id','Unnamed','Name','HomeOrAway','PregameOdds','Conf_','Team_','Spread','Wse','Lse','Win','Lose','Payout','Book','book','Score','Created','Updated','Status','Type','Number']
    col_groups=[data.filter (like=c).columns for c in drop]
    for cols in col_groups:
        for  c in cols:
            if c in data.columns:
                if c not in keep:
                    if verbose:
                        print('Drop:', c)
                    data.drop(c,axis=1,inplace=True)
    return data

def moneyline_to_odds(line):
    if line>=0:
        odd=(line/100)+1

    else: 
        odd=(100/np.abs(line))+1
    return odd

def convert_lines(data):
    data['Home_Odds']=data['HomeMoneyLine'].apply(moneyline_to_odds)
    data['Away_Odds']=data['AwayMoneyLine'].apply(moneyline_to_odds)
    return data

def fix_date(data):
    data['Date']=pd.to_datetime(data['Day'])
    data=data.drop('Day',axis=1)
    return data

def get_odds_results(data,result_column='_W'):
    cols=data.filter(like=result_column,axis=1).columns
    print(cols)
    results=data[cols].astype(int)
    odds=data.drop(cols,axis=1)
    return odds,results

def get_team_names(data):
    name_cols=data.filter(like='Name').columns
    names=data[name_cols[:2]]
    return names


# split odds and results
def split_data(odds,results,season=2023):
    trainidx=odds[odds.Season<season].index
    testidx=odds[odds.Season>=season].index
    # split odds
    train_odds=odds.loc[trainidx]
    test_odds=odds.loc[testidx]
    #split resilts
    train_results=results.loc[trainidx]
    test_results=results.loc[testidx]
    return train_odds,test_odds,train_results,test_results
'''
these functions are for building the prediction model
'''

## default params for finding a good model
default_params ={   
            'clf__scale_pos_weight':[1,50,99],
              'clf__learning_rate':[.01,.1],
              'clf__gamma':[.1,.3,.5,.8],
              'clf__n_estimators':[1000],
            #   'clf__min_child_weight':[4,8],
             }
def get_features(data,cat_features=['ID','Season','ID']):
    '''
    Every column that is not categorical features will be numerical
    '''
    categorical_features=[]
    for c in cat_features:
        cats=data.filter(like=c).columns
        [categorical_features.append(cat) for cat in cats]
        cat_features=list(set(categorical_features))
    # grab numaricals
    num_features=[col for col in data.columns if col not in cat_features]
    if 'Date' in num_features:
        num_features.remove('Date')
    #turn in to numerical

    data[num_features]=data[num_features].apply(pd.to_numeric,errors='coerce')

    for col in data.filter(like='Rk').columns:
        data[col]=data[col].fillna(400)
    
    return data, num_features, cat_features

def build_feature_processor(num_features,cat_features,verbose=False):
    if verbose:
        print('Build Feature processor...')
    num_processor = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])
    cat_processor = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder',OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_processor = ColumnTransformer(transformers = [
    ('num_pipeline', num_processor,num_features),
    ('cat_pipeline', cat_processor,cat_features),    # ('ord_pipeline', ord_processor,ord_features),
    ],remainder='drop') 
    if verbose:
        print('---------------------------')
        print(f'{len(cat_features)} Categorical Features:')
        print(cat_features)
        print('---------------------------')
        print(f'{len(num_features)}  Numerical Features:')
        print(num_features)
    return feature_processor

def build_pipe(feature_processor,objective='binary:logistic'):
    print('Build Pipeline...')
    forest_clf= xgb.XGBClassifier(objective=objective)
    pipe = Pipeline(steps=[
    ('feature_processor', feature_processor),  
    # ('upsample',SMOTE(k_neighbors=10,sampling_strategy='minority')),
    ('pca',PCA(n_components =40)),
    ('clf',forest_clf)
    ])
    return pipe

def train_model(X,y,pipeline,param_dic,cross_vals=2,n_jobs=-1):
    print('Begin Training...')
    # dict for scores
    scores={}
    # train test split
    xtrain,xtest,ytrain,ytest=train_test_split(X,y)
    grid = GridSearchCV(pipeline,param_dic,cv=cross_vals,scoring='neg_log_loss',n_jobs=n_jobs,verbose=0)
    grid.fit(xtrain,ytrain)
    scores['Training']=grid.best_score_
    model=grid.best_estimator_
    ypred=model.predict(xtest)
    scores['Testing']=accuracy_score(ytest,ypred)

    print(scores)
    print(classification_report(ytest,ypred))

    return model, scores

def build_model_and_train(X,y,model_params=default_params,cross_vals=2,n_jobs=-1, target_col='Home_W',categorical_columns=['ID','Season','ID'],load=False):
    if load:
        model=joblib.load(f'Daily_models/BB_RL_predictor.pkl')
        scores=[0]
        pass

    else:
        X,cat, num=get_features(X,cat_features=categorical_columns)
        feature_processor=build_feature_processor(num,cat)
        pipe=build_pipe(feature_processor)
        model,scores=train_model(X,y,pipeline=pipe,param_dic=model_params,cross_vals=cross_vals,n_jobs=n_jobs)
        print('Done Training! Model Incoming...')
        joblib.dump(model,f'Daily_models/BB_RL_predictor.pkl')

        # model.steps.pop(-3)
        
    
    return model,scores

def make_feature_processor(data,cat_features=['ID','Season','ID']):
    data, num,cat =get_features(data,cat_features=cat_features)
    feature_processor=build_feature_processor(num_features=num,cat_features=cat)
    return feature_processor.fit(data)

def predict_process(observation,model):
    probas=model.predict_proba(observation).flatten()
    obs=model['feature_processor'].transform(observation)
    obs=obs.flatten()
    prob_obs=np.concatenate([probas,obs])
    return prob_obs.reshape(1,-1)

def map_team_names(data):
    data=data.replace(chatgpt_mapping)
    data=data.replace(my_mapping)
    return data

my_mapping={"St Peter's": 'St Peters',
 'SUNY Albany': 'Albany',
 'Loyola-Chicago': 'Loyola Chicago',
 'Southern Miss': 'Southern Mississippi',
 'IL Chicago': 'Illinois Chicago',
 "Mt St Mary's": 'Mount St Marys',
 'UT Arlington': 'Texas Arlington',
 'S Carolina St': 'South Carolina St',
 'Col Charleston': 'College of Charleston',
 'FL Gulf Coast': 'Florida Gulf Coast',
 'CS Fullerton': 'Cal St Fullerton',
 'Loyola MD': 'Loyola Maryland',
 'TX Southern': 'Texas Southern',
 'American Univ': 'American',
 'LIU Brooklyn': 'Long Island Brooklyn',
 'UT San Antonio': 'Texas San Antonio',
 'Miami FL': 'Miami',
 'SF Austin': 'Stephen F Austin',
 'CS Northridge': 'Cal St Northridge',
 'WI Milwaukee': 'Wisconsin Milwaukee',
 'Miami OH': 'Miami Ohio',
 'Kent': 'Kent St',
 'F Dickinson': 'Fairleigh Dickinson',
 'S Dakota St': 'South Dakota St',
 'Ark Pine Bluff': 'Arkansas Pine Bluff',
 'UC Irvine': 'Cal Irvine',
 'Central Conn': 'Central Connecticut St',
 'Monmouth NJ': 'Monmouth',
 'N Kentucky': 'Northern Kentucky',
 'Ark Little Rock': 'Arkansas Little Rock',
 "St Joseph's PA": 'St Josephs',
 'WI Green Bay': 'Wisconsin Green Bay',
 'Louisiana': 'Louisiana Lafayette',
 "St John's": 'St Johns',
 'Boston Univ': 'Boston University',
 'CS Bakersfield': 'Cal St Bakersfield',
 'MS Valley St': 'Mississippi Valley St',
 'Northwestern LA': 'Northwestern St',
 'TAM C. Christi': 'Texas A&M Corpus Christi',
 'E Kentucky': 'Eastern Kentucky',
 'UC Santa Barbara': 'Santa Barbara',
 'SE Louisiana': 'Southeastern Louisiana',
 'N Dakota St': 'North Dakota St',
 'G Washington': 'George Washington',
 'W Michigan': 'Western Michigan',
 'C Michigan': 'Central Michigan',
 'E Washington': 'Eastern Washington',
 'Southern Univ': 'Southern',
 'S Illinois': 'Southern Illinois',
 'UCF': 'Central Florida',
 'NC Central': 'North Carolina Central',
 'Coastal Car': 'Coastal Carolina',
 "St Mary's CA": 'St Marys',
 'Mississippi':'Ole Miss',
 "Penn":"Pennsylvania",
 "N Colorado" :"Northern Colorado",
 }

chatgpt_mapping = {
"A&M-Corpus Christi": "TAM C. Christi",
"Cal St Fullerton": "CS Fullerton",
"Cal St Northridge": "CS Northridge",
"Centenary (LA)": "Centenary",
"Centenary (LA)": "Centenary",
"E. Washington": "E Washington",
"G. Washington": "G Washington",
"IUPU-Indianapolis": "IUPUI",
"Ill.-Chicago": "IL Chicago",
"LA-Lafayette": "Louisiana Lafayette",
"LA-Monroe": "Louisiana",
"LIU-Brooklyn": "LIU Brooklyn",
"Loyola Marymount": "LMU",
"Mississippi Valley St": "MS Valley St",
"Morehead St.": "Morehead St",
"N. Colorado": "N Colorado",
"N. Dakota St.": "N Dakota St",
"N. Kentucky": "N Kentucky",
"N.C. A&T": "NC A&T",
"N.C. Central": "NC Central",
"N.C. State": "NC State",
"S. Carolina St.": "S Carolina St",
"S. Dakota St.": "S Dakota St",
"S. Illinois": "S Illinois",
"SE Louisiana": "Southeastern Louisiana",
"SF Austin": "SF Austin",
"Sam Houston St.": "Sam Houston St",
"St. Bonaventure": "St Bonaventure",
"St. John's": "St John's",
"St. Joseph's (PA)": "St Joseph's PA",
"St. Louis": "St Louis",
"St. Mary's (CA)": "St Mary's CA",
"St. Peter's": "St Peter's",
"Texas A&M-Corpus Christi": "TAM C. Christi",
"Texas Southern": "TX Southern",
"UAB": "UAB",
"UC-Santa Barbara": "UC Santa Barbara",
"UNC-Asheville": "UNC Asheville",
"UNC-Greensboro": "UNC Greensboro",
"UNC-Wilmington": "UNC Wilmington",
"USC Upstate": "USC Upstate",
"Utah Valley": "Utah Valley",
"VCU": "VCU",
"W. Carolina": "W Carolina",
"W. Illinois": "W Illinois",
"W. Kentucky": "Western Kentucky",
"W. Michigan": "W Michigan",
"W. Virginia": "W Virginia",
"WI-Green Bay": "WI Green Bay",
"WI-Milwaukee": "WI Milwaukee",
"Wagner": "Wagner",
"Weber St.": "Weber St",
"Wichita St.": "Wichita St",
"Winthrop": "Winthrop",
"Wright St.": "Wright St",
}