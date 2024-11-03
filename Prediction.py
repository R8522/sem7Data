import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("https://raw.githubusercontent.com/R8522/sem7Data/refs/heads/main/Patient%20Survival%20Prediction.csv")
df.head()

st.write(df)

df.isna().any()
df.isna().sum()

# DATA CLEANING
# REMOVE COLUMN ID
# copy dataframe in new dataframe
dframe = pd.DataFrame(df)
# Remove two columns name is 'C' and 'D'
dframe.drop(['encounter_id', 'patient_id', 'hospital_id', 'icu_id', 'icu_type', 'icu_admit_source', 'icu_stay_type', 'icu_type' ], axis=1,)

dframe.isna().sum()
dframe.duplicated().sum()

# IDENTIFY OUTLIERS
#col = ['pre_icu_los_days', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'hospital_death']

import numpy as np; np.random.seed(42)

data = pd.DataFrame(data = dframe, columns = ['pre_icu_los_days', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'bmi',
                                              'hospital_death'])
#data.dropna(inplace = True)
plt.figure(figsize=(10, 6))  # Adjust the width and height as needed

sns.boxplot(x="variable", y="value", data=pd.melt(data))
plt.show()

st.pyplot(plt.gcf())
