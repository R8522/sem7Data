import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# distributing the dataset into two components X and Y
X = data.drop(columns = ['hospital_death'])
y = data['hospital_death']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

#performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression To the training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 5)
classifier.fit(X_train, y_train)

selected_columns = ['pre_icu_los_days', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'hospital_death']

# Set up subplots for features
fig, axes = plt.subplots(nrows=len(selected_columns), ncols=1, figsize=(10, 6 * len(selected_columns)))

# Plot count distribution for each feature
for i, feature in enumerate(selected_columns):
    sns.countplot(x=feature, data=df, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

st.pyplot(plt.gcf())
