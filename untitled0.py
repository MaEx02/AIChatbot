import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = pd.read_csv('C:/Users/shaco/Desktop/heart.csv')
df.head()
num = ['Age','RestingBP','Cholesterol','Oldpeak']
cat = ['Sex','ChestPainType','FastingBS','RestingECG','ExerciseAngina','ST_Slope']
df_num = df[num]
df_num.head()
df_cat = df[cat]
df_cat.head()
df_cat = df_cat.astype('object')
df_cat.head()
df_cat.dtypes
df_cat = pd.get_dummies(df_cat,drop_first=True)
df_cat.head()
df_cat.dtypes
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
df_num = pd.DataFrame(pt.fit_transform(df_num),columns=df_num.columns)
df_num.head()
df_num.skew()
df_feature = pd.concat([df_cat,df_num],axis=1)
df_feature.head()
df_target = df['HeartDisease']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_feature,df_target,test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred))
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
model2 = dt.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
cm2 = confusion_matrix(y_test,y_pred2)
sns.heatmap(cm2,annot=True)
print(classification_report(y_test,y_pred2))