#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
print('슝=3')


# In[4]:


import os
csv_path = os.getenv("HOME") +"/aiffel/pokemon_eda/data/Pokemon.csv"
original_data = pd.read_csv(csv_path)
print('슝=3')


# In[5]:


pokemon = original_data.copy()
print(pokemon.shape)
pokemon.head()


# In[6]:


# 전설의 포켓몬 데이터셋
legendary = pokemon[pokemon["Legendary"] == True].reset_index(drop=True)
print(legendary.shape)
legendary.head()


# In[7]:


# Q. 일반 포켓몬의 데이터셋도 만들어봅시다.
ordinary = pokemon[pokemon["Legendary"] == False].reset_index(drop=True)
print(ordinary.shape)
ordinary.head()


# In[8]:


# 결측치 확인
pokemon.isnull().sum()


# In[9]:


print(len(pokemon.columns))
pokemon.columns


# In[10]:


# #의 종류 수
len(set(pokemon["#"]))


# In[11]:


# 해당 값을 가지는 포켓몬
pokemon[pokemon["#"] == 6]


# In[12]:


# Q. 총 몇 종류의 포켓몬 이름이 있는지 확인해봅시다!
len(set(pokemon["Name"]))


# In[13]:


# 무작위로 두 마리의 포켓몬
pokemon.loc[[6, 10]]


# In[14]:


#각 속성의 종류는 총 몇 가지?
len(list(set(pokemon["Type 1"]))), len(list(set(pokemon["Type 2"])))


# In[22]:


# 차이나는 하나는 무엇?
set(pokemon["Type 2"]) - set(pokemon["Type 1"])


# In[15]:


# type 
types = list(set(pokemon["Type 1"]))
print(len(types))
print(types)


# In[23]:


# 하나의 속성만 가지는 포켓몬 수
pokemon["Type 2"].isna().sum()


# In[16]:


plt.figure(figsize=(10, 7))  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=ordinary, x="Type 1", order=types).set_xlabel('')
plt.title("[Ordinary Pokemons]")

plt.subplot(212)
sns.countplot(data=legendary, x="Type 1", order=types).set_xlabel('')
plt.title("[Legendary Pokemons]")

plt.show()


# In[24]:


# Type1별로 Legendary의 비율을 보여주는 피벗 테이블
pd.pivot_table(pokemon, index="Type 1", values="Legendary").sort_values(by=["Legendary"], ascending=False)


# In[17]:


#Type 2 데이터 분포 plot
plt.figure(figsize=(12, 10))  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=ordinary, x="Type 2", order=types).set_xlabel('')
plt.title("[Ordinary Pokemons]")

plt.subplot(212)
sns.countplot(data=legendary, x="Type 2", order=types).set_xlabel('')
plt.title("[Legendary Pokemons]")

plt.show()


# In[25]:


# Q. Type 2에 대해서도 피벗 테이블을 만들어봅시다.
pd.pivot_table(pokemon, index="Type 2", values="Legendary").sort_values(by=["Legendary"], ascending=False)


# In[18]:


stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
stats


# In[19]:


print("#0 pokemon: ", pokemon.loc[0, "Name"])
print("total: ", int(pokemon.loc[0, "Total"]))
print("stats: ", list(pokemon.loc[0, stats]))
print("sum of all stats: ", sum(list(pokemon.loc[0, stats])))


# In[21]:


# Q. 'pokemon['Total'].values'와 'pokemon[stats].values의 총합'이 같은 포켓몬의 수를 확인해봅시다.
total_values = pokemon['Total'].values
stats_values = []
for i in range(len(pokemon)):
    stats_values.append(pokemon.loc[i, stats].sum())

same_total_stats_sum = 0
for i in range(len(pokemon)):
    if total_values[i] == stats_values[i]:
        same_total_stats_sum += 1
same_total_stats_sum


# In[26]:


fig, ax = plt.subplots()
fig.set_size_inches(12, 6)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

sns.scatterplot(data=pokemon, x="Type 1", y="Total", hue="Legendary")
plt.show()


# In[27]:


fig, ax = plt.subplots()
fig.set_size_inches(12, 6)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

sns.scatterplot(data=pokemon, x="Type 1", y="Total", hue="Legendary")
plt.show()


# In[28]:


fig, ax = plt.subplots()
fig.set_size_inches(12, 6)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

sns.scatterplot(data=pokemon, x="Type 1", y="Total", hue="Legendary")
plt.show()


# In[29]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

sns.scatterplot(data=legendary, y="Type 1", x="Total")
plt.show()


# In[30]:


print(sorted(list(set(legendary["Total"]))))


# In[31]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

sns.countplot(data=legendary, x="Total")
plt.show()


# In[32]:


round(65 / 9, 2)


# In[33]:


# Q. ordinary 포켓몬의 'Total' 값 집합을 확인해봅시다.
print(sorted(list(set(ordinary["Total"]))))


# In[34]:


# Q. 이 집합의 크기(길이)를 확인해봅시다.
len(set(ordinary["Total"]))


# In[35]:


round(735 / 195, 2)


# In[36]:


n1, n2, n3, n4, n5 = legendary[3:6], legendary[14:24], legendary[25:29], legendary[46:50], legendary[52:57]
names = pd.concat([n1, n2, n3, n4, n5]).reset_index(drop=True)
names


# In[37]:


formes = names[13:23]
formes


# In[38]:


legendary["name_count"] = legendary["Name"].apply(lambda i: len(i))    
legendary.head()


# In[39]:


# Q. ordinary 포켓몬의 데이터에도 'name_count' 값을 추가해줍시다.
ordinary["name_count"] = ordinary["Name"].apply(lambda i: len(i))    
ordinary.head()


# In[40]:


plt.figure(figsize=(12, 10))   # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=legendary, x="name_count").set_xlabel('')
plt.title("Legendary")
plt.subplot(212)
sns.countplot(data=ordinary, x="name_count").set_xlabel('')
plt.title("Ordinary")
plt.show()


# In[41]:


print(round(len(legendary[legendary["name_count"] > 9]) / len(legendary) * 100, 2), "%")


# In[42]:


# Q. 일반 포켓몬의 이름이 10글자 이상일 확률을 구해보세요.
print(round(len(ordinary[ordinary["name_count"] > 9]) / len(ordinary) * 100, 2), "%")


# In[43]:


pokemon["name_count"] = pokemon["Name"].apply(lambda i: len(i))
pokemon.head()


# In[44]:


pokemon["long_name"] = pokemon["name_count"] >= 10
pokemon.head()


# In[45]:


pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon.tail()


# In[46]:


pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
pokemon.head()


# In[47]:


print(pokemon[pokemon["name_isalpha"] == False].shape)
pokemon[pokemon["name_isalpha"] == False]


# In[48]:


pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="Porygon Two")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")

pokemon.loc[[34, 37, 90, 131, 252, 270, 487, 525, 794]]


# In[49]:


# Q. 바꿔준 'Name' 컬럼으로 'Name_nospace'를 만들고, 다시 isalpha()로 체크해봅시다.
pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.isalpha())
pokemon[pokemon["name_isalpha"] == False]


# In[50]:


import re


# In[52]:


name = "CharizardMega Charizard X"
name_split = name.split(" ")
name_split
temp = name_split[0]
temp
tokens = re.findall('[A-Z][a-z]*', temp)
tokens
tokens = []
for part_name in name_split:
    a = re.findall('[A-Z][a-z]*', part_name)
    tokens.extend(a)
tokens


# In[53]:


# Q. 다음 코드의 빈칸을 채워주세요.
def tokenize(name):
    tokens = []
    name_split = name.split(" ")
    for part_name in name_split:
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a)
    return np.array(tokens)


# In[54]:


name = "CharizardMega Charizard X"
tokenize(name)


# In[55]:


all_tokens = list(legendary["Name"].apply(tokenize).values)

token_set = []
for token in all_tokens:
    token_set.extend(token)

print(len(set(token_set)))
print(token_set)


# In[56]:


from collections import Counter
a = [1, 1, 0, 0, 0, 1, 1, 2, 3]
Counter(a)
Counter(a).most_common()
most_common = Counter(token_set).most_common(10)
most_common


# In[57]:


for token, _ in most_common:
    # pokemon[token] = ... 형식으로 사용하면 뒤에서 warning이 발생합니다
    pokemon[f"{token}"] = pokemon["Name"].str.contains(token)

pokemon.head(10)


# In[59]:


print(types)


# In[60]:


for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)
    
pokemon[[["Type 1", "Type 2"] + types][0]].head()


# In[61]:


print(original_data.shape)
original_data.head()


# In[62]:


original_data.columns


# In[63]:


features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']


# In[64]:


target = 'Legendary'


# In[65]:


# Q. 'original_data'에서 'features' 컬럼에 해당하는 데이터를 변수 'X'에 저장합니다.
X = original_data[features]
print(X.shape)
X.head()


# In[66]:


# Q. 'target' 컬럼의 데이터를 변수 'y'에 저장합니다.
y = original_data[target]
print(y.shape)
y.head()


# In[67]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[68]:


from sklearn.tree import DecisionTreeClassifier
print('슝=3')


# In[69]:


model = DecisionTreeClassifier(random_state=25)
model


# In[70]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('슝=3')


# In[71]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[72]:


len(legendary)


# In[73]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[74]:


print(len(pokemon.columns))
print(pokemon.columns)


# In[75]:


features = ['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 
            'name_count','long_name', 'Forme', 'Mega', 'Mewtwo','Deoxys', 'Kyurem', 'Latias', 'Latios',
            'Kyogre', 'Groudon', 'Rayquaza','Poison', 'Ground', 'Flying', 'Normal', 'Water', 'Fire',
            'Electric','Rock', 'Dark', 'Fairy', 'Steel', 'Ghost', 'Psychic', 'Ice', 'Bug', 'Grass', 'Dragon', 'Fighting']

len(features)


# In[76]:


target = "Legendary"
target


# In[78]:


# Q. 사용할 feature에 해당하는 데이터를 'X' 변수에 저장합니다.
X = pokemon[features]
print(X.shape)
X.head()


# In[80]:


# Q. 정답 데이터 'y'도 'target' 변수를 이용해 만들어줍시다.
y = pokemon[target]
print(y.shape)
y.head()


# In[81]:


model = DecisionTreeClassifier(random_state=25)
model


# In[82]:


# Q. train 데이터로 decision tree 모델을 학습시키고
# test 데이터로 모델의 예측 값을 얻어봅시다!
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=25)

pred = model.predict(X_test)
pred


# In[83]:


# Q. confusion matrix를 확인해보세요.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[84]:


# Q. classification report도 확인해봅시다!
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




