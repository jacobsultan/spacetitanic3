import os
import pandas as pd
from fastbook import *
from fastai.tabular.all import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from dtreeviz.trees import *
import sklearn

def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)

traindf = pd.read_csv('/mnt/c/Users/jsult/Desktop/spacetitanic3/train.csv',low_memory=False)
testdf = pd.read_csv('/mnt/c/Users/jsult/Desktop/spacetitanic3/test.csv',low_memory= False)
traindf.loc[traindf.CryoSleep == True, ['FoodCourt','Spa','RoomService','ShoppingMall','VRDeck']] = traindf.loc[traindf.CryoSleep == True, ['FoodCourt','Spa','RoomService','ShoppingMall','VRDeck']].fillna(0)
traindf.dropna(inplace = True)

def split_names(df):
    # Create new columns for first name and last name
    df['FirstName'] = df['Name'].apply(lambda x: x.split()[0] if pd.notna(x) else pd.NA)
    df['LastName'] = df['Name'].apply(lambda x: x.split()[1] if pd.notna(x) else pd.NA)
    df.drop(columns = ['Name'],inplace = True)
    return df
traindf = split_names(traindf)
testdf = split_names(testdf)

def split_group_number(df):
    df['Group'] = df['PassengerId'].apply(lambda x: x.split("_")[0] if pd.notna(x) else pd.NA)
    df['GroupNumber'] = df['PassengerId'].apply(lambda x: x.split("_")[1] if pd.notna(x) else pd.NA)
    return df
traindf = split_group_number(traindf)
testdf = split_group_number(testdf)


def total_spending(df):
    df['TotalSpending'] = df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck'] + df['RoomService']
    return df
traindf = total_spending(traindf)

spending = ['VRDeck','Spa','RoomService','FoodCourt','ShoppingMall']
traindf['TotalSpendingInCabin'] = traindf.groupby('Cabin')[spending].transform('sum').sum(axis=1)
testdf['TotalSpendingInCabin'] = testdf.groupby('Cabin')[spending].transform('sum').sum(axis=1)

def cabin_size(row):
    samecabin = traindf[traindf.Cabin == row.Cabin]
    return len(samecabin)

traindf['CabinSize'] = traindf.apply(cabin_size, axis = 1)
testdf['CabinSize'] = testdf.apply(cabin_size,axis = 1)


def split_cabin(df):
    df['CabinDeck'] = df['Cabin'].apply(lambda x: x.split("/" )[0] if pd.notna(x) else pd.NA)
    df['CabinSide'] = df['Cabin'].apply(lambda x: x.split("/")[2] if pd.notna(x) else pd.NA)
    df['CabinNum'] = df['Cabin'].apply(lambda x: int(x.split("/")[1]) if pd.notna(x) else pd.NA)

    df.drop(columns = ['Cabin'],inplace = True)
    return df
traindf = split_cabin(traindf)
testdf= split_cabin(testdf)




dep_var = 'Destination'
procs = [Categorify]

cat = ['VIP',
       'CabinDeck', 'CabinSide','CryoSleep','HomePlanet']
cont = ['Age','RoomService', 'ShoppingMall', 'Spa', 'VRDeck','CabinNum','FoodCourt','TotalSpending',
        'TotalSpendingInCabin','CabinSize']
to = TabularPandas(traindf, procs, cat, cont, y_names=dep_var)

xs,y = to.train.xs,to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

m = DecisionTreeRegressor(max_leaf_nodes=10)
m.fit(xs, y)

fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(m,
                   feature_names= xs.columns,
                   filled=True)

#print('decision tree prediction',((m.predict(valid_xs) > 0.5) == valid_y).sum() / len(valid_y))




list(zip(traindf.CabinDeck.unique(),xs.CabinDeck.unique()))



def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

fi = rf_feat_importance(m, xs)
print(fi[:30])


traindf.columns
raindf = traindf.drop(columns = ['PassengerId','CabinNum','Name','TotalSpending'])
raindf = pd.get_dummies(raindf)




df_all_corr = raindf.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'FoodCourt']