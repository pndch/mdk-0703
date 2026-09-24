import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv('titanic.csv')
print(df.shape); print(df.head()); df.info(); print(df.describe())

kids = df[df['age'] < 12]
print(len(kids), kids['survived'].mean())

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print(df.isna().sum())

print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())
print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))


# fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
# ax[0].hist(df['age'], bins=30); ax[0].set_title('Возраст')
# sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]); ax[1].set_title('Цена по классу')
# sns.barplot(x='sex', y='survived', data=df, ax=ax[2]); ax[2].set_title('Выживаемость')
# fig.tight_layout(); fig.savefig('titanic.png', dpi=120)
# plt.show()

df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv('clean.csv', index=False)

df = pd.read_csv("clean.csv")
print(df.shape); print(df.head()); df.info(); print(df.describe())


survived = df[df["survived"] == 1]["fare"]
died = df[df["survived"] == 0]["fare"]

#sns.barplot(x="survived", y="fare", data=df)

### x - fare; y - survived
plt.hist(x=survived, bins=30, alpha=0.5, label="x=fare, y=survived")
plt.hist(x=died, bins=30, alpha=0.5)
plt.show()