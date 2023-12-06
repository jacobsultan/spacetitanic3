raindf = traindf.drop(columns = ['PassengerId','CabinNum','Cabin','Name'])
raindf = pd.get_dummies(raindf)
df_all_corr = raindf.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'Age']