import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('KNN_Project_Data')

df.head()

sns.pairplot(df)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(df.drop('TARGET CLASS' , axis=1))

dff = sc.transform(df.drop('TARGET CLASS' , axis=1))

dff = pd.DataFrame(dff, columns=df.columns.drop('TARGET CLASS'))
dff.head()

y = df['TARGET CLASS']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( dfff , y, test_size=0.33, random_state=42)

from sklearn.neighbors import  KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X= X_train , y=y_train )

pr = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,pr))
print(classification_report(y_test,pr))

error = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    prr = knn.predict(X_test)
    error.append(np.mean(prr!=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10 )
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
prr = knn.predict(X_test)
print(classification_report(y_test,prr))
print(confusion_matrix(y_test,prr))