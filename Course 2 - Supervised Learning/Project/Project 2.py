
#%%[markdown]
## About the Data
#%%
get_ipython().system('cat adult.names')

#%%[markdown]
## Importing Dependencies

#### 1. Importing the dependencies for the notebook below

#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

col_names = [
"age", 
"workclass", 
"fnlwgt", 
"education", 
"education-num", 
"marital-status", 
"occupation", 
"relationship", 
"race", 
"sex", 
"capital-gain", 
"capital-loss", 
"hours-per-week", 
"native-country",
"Income Group"]

# %%[markdown]
## Data Collection
#### 2.1 loading the data from adults.data into DataFrame df1

# %%
df1 = pd.read_csv('adult.data')
df1.columns = col_names
df1.sample()

# %%[markdown]
#### 2.2 loading the data from adults.test into DataFrame df2

# %%
df2 = pd.read_csv('adult.test')
df2.reset_index(inplace=True)
df2.columns = col_names
df2.sample()

# %% [markdown]
#### 3. joining the DataFrames into a single DataFrame df

# %%


df = pd.concat([df1, df2], axis=0)
df = df.reset_index(drop=True)
df.sample()


# %%


df["Income Group"].value_counts()


# there are four labels when its actually a binary classification so we need to process and remove the labels with full stop character

# %%


df["Income Group"] = df["Income Group"].map(lambda x: x[:-1] if(x[-1] == ".") else x)
df["Income Group"].value_counts()

#%% [markdown]
#### 4. Handling Missing values:
# Conversion of original data as follows:
# - Discretized agrossincome into two ranges with threshold 50,000.
# - Convert U.S. to US to avoid periods.
# - Convert Unknown to "?"
# so we have to check for "?" in the data and impute those values

# %%


for col in col_names:
    if(df[col].dtype == 'O'):
        print(col + ": " , round(len(df[list(map(lambda x: False if(x == -1) else True, df[col].str.find('?').values))])/len(df)*100, 2), "%")

# %%[markdown]

# so for Categorical variables we have a missing values for 
# - workclass
# - occupation
# - native-country

# Replacing the '?' character will make the imputation process much simple
# i will be making a duplicate dataframe so i can test out which imputation is suitable

# %%
for col in col_names:
    if(df[col].dtype == 'O'):
        df[col] = list(map(lambda x, y: y if(x == -1) else np.nan, df[col].str.find('?').values, df[col].values))


# %%
df.info()


# %%
from copy import deepcopy
df_impute_1 = deepcopy(df)

# %% [markdown]
# i am going to use mode imputation as all the columns are categorical in nature. will be showing the results once this imputation is proved

# %%
for col in df_impute_1.columns:
    df_impute_1[col] = df_impute_1[col].fillna(df_impute_1[col].mode().values[0]).values

# %%[markdown]
# ### Data Exploration

# %%
df_impute_1.info()


# %%
df.describe().T


# %%
df.describe(include="object").T.sort_values(by="unique")


# %%
cat_cols = df.describe(include="object").T.sort_values(by="unique").index


# %%
fig, axs = plt.subplots(3,3, figsize=(15,15))
fig.subplots_adjust(hspace=0.4, wspace=0.9)
for i in range(len(cat_cols)):
    sns.countplot(y = cat_cols[i], data=df, orient="v", ax=axs[int(i/3), i%3])
plt.show()    


# %%
sns.pairplot(df, diag_kind="kde", hue="Income Group")

# %% [markdown]
# from preliminary analysis it seems that the numerical features have not seperation power except age column

# ### Data Preparation
# we already have the nature of each column
# - Numerical Columns
#   - discrete:
#       - education-num
#   - continuous:
#       - age
#       - fnlwgt
#       - capital-gain
#       - capital-loss
#       - hours-per-week
# 
#
# - Categorical Columns 
#   - nominal
#       - marital-status
#       - occupation
#       - relationship
#       - race
#       - sex
#       - native-country
#       - workclass
#   - ordinal
#       - education
