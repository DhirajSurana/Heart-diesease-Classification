<div class="cell markdown">

# Predicting Heart Disease Using Machine Learning

## 1\. Problem Statement

> Given clinical data of a patient, predict whether the patient has
> heart disease.

## 2\. Data

> Kaggle Dataset: <https://www.kaggle.com/c/heart-disease-uci>  
> UCI Dataset: <https://archive.ics.uci.edu/ml/datasets/heart+disease>

## 3\. Evaluation

> If we can reach 95% accuracy, we are good to go.

## 4\. Features

> The data set contains the following features:
> 
> There are ***`13`*** attributes and ***`1`*** target attribute.:
> 
> 1.  `age`: age in years
> 
> 2.  `sex`: sex (1 = male; 0 = female)
> 
> 3.  `cp`: chest pain type
>     
>       - Value 0: typical angina
>       - Value 1: atypical angina
>       - Value 2: non-anginal pain
>       - Value 3: asymptomatic
> 
> 4.  `trestbps`: resting blood pressure (in mm Hg on \> admission to
>     the hospital)
> 
> 5.  `chol`: serum cholestoral in mg/dl
> 
> 6.  `fbs`: (fasting blood sugar \> 120 mg/dl) (1 = true; 0 \> = false)
> 
> 7.  `restecg`: resting electrocardiographic results
>     
>       - Value 0: normal
>       - Value 1: having ST-T wave abnormality (T wave \> inversions
>         and/or ST elevation or depression of \> 0.\> 05 mV)
>       - Value 2: showing probable or definite left \> ventricular
>         hypertrophy by Estes' criteria
> 
> 8.  `thalach`: maximum heart rate achieved
> 
> 9.  `exang`: exercise induced angina (1 = yes; 0 = no)
> 
> 10. `oldpeak` = ST depression induced by exercise \> relative to rest
> 
> 11. `slope`: the slope of the peak exercise ST segment
>     
>       - Value 0: upsloping
>       - Value 1: flat
>       - Value 2: downsloping
> 
> 12. `ca`: number of major vessels (0-3) colored by \> flourosopy
> 
> 13. `thal`:
>     
>       - 0 = normal
>       - 1 = fixed defect
>       - 2 = reversable defect and the label
> 
> 14. `condition`:
>     
>       - 0 = no disease
>       - 1 = disease

</div>

<div class="cell markdown">

### Preparing Tools

We will use the following libraries:

1.  Pandas
2.  Numpy
3.  Sklearn (SciKit-Learn)
4.  Seaborn

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.748054Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.113449Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.11354Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.749023Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.113186Z&quot;}" data-trusted="true">

``` python
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We want our plots to appear in the notebook
%matplotlib inline

# Importing the Machine Learing models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluation 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve
```

</div>

<div class="cell markdown">

### Load Data

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.765452Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.750911Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.750939Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.767049Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.750695Z&quot;}" data-trusted="true">

``` python
df=pd.read_csv('../input/heart-disease/heart.csv')
df.shape
```

</div>

<div class="cell markdown">

### Data Exploration

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.786533Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.768708Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.768743Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.787331Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.768424Z&quot;}" data-trusted="true">

``` python
df.head()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.806632Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.790292Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.790345Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.807545Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.789321Z&quot;}" data-trusted="true">

``` python
df.tail()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.817413Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.809419Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.809455Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.818283Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.809096Z&quot;}" data-trusted="true">

``` python
df.isna().sum()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.876967Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.82044Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.820481Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.877889Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.819723Z&quot;}" data-trusted="true">

``` python
df.describe()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.892099Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.879468Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.879497Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.892802Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.87925Z&quot;}" data-trusted="true">

``` python
df.info()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:41:59.90726Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.894353Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.894386Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:41:59.907911Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.893776Z&quot;}" data-trusted="true">

``` python
df.target.value_counts()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:00.124125Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:41:59.909653Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:41:59.909698Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:00.125091Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:41:59.909278Z&quot;}" data-trusted="true">

``` python
df.target.value_counts().plot.bar(color=['red','blue'],figsize=(10,5));
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:00.135426Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:00.128353Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:00.128385Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:00.136315Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:00.128088Z&quot;}" data-trusted="true">

``` python
df.sex.value_counts()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:00.164638Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:00.137744Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:00.137772Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:00.165634Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:00.13754Z&quot;}" data-trusted="true">

``` python
pd.crosstab(df.sex,df.target)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:00.41837Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:00.167091Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:00.167117Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:00.419066Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:00.166873Z&quot;}" data-trusted="true">

``` python
pd.crosstab(df.target,df.sex).plot.bar(color=['red','blue'],figsize=(10,5))
plt.title("Heart Disease frequency for Sex")
plt.xlabel("0 = No disease 1 = Heart disease")
plt.ylabel("Amount")
plt.legend(['Female', 'Male']);
```

</div>

<div class="cell markdown">

### Age vs Maximum heart rate for Heart Disease

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:00.760389Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:00.421421Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:00.421469Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:00.761022Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:00.42045Z&quot;}" data-trusted="true">

``` python
fig,ax = plt.subplots(figsize=(10,6))
ax.scatter(df.age[df.target==1],df.thalach[df.target==1],color='red')
ax.scatter(df.age[df.target==0],df.thalach[df.target==0],color="blue")
ax.set_title("Age vs Maximum heart rate")
ax.set_xlabel("Age")
ax.set_ylabel("Maximum heart rate")
ax.axhline(df.thalach.mean(),linestyle='--',color='black')
ax.legend(['Heart disease', 'No disease','Average heart rate']);
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:00.989048Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:00.762631Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:00.76267Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:00.990014Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:00.762067Z&quot;}" data-trusted="true">

``` python
df.age.plot.hist(figsize=(10,5),color='blue');
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:01.015292Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:00.99154Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:00.99157Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:01.016501Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:00.991318Z&quot;}" data-trusted="true">

``` python
pd.crosstab(df.cp,df.target)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:01.285338Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:01.018052Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:01.018082Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:01.286295Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:01.01783Z&quot;}" data-trusted="true">

``` python
pd.crosstab(df.cp,df.target).plot.bar(color=['blue','red'],figsize=(10,5));
plt.title("Heart Disease frequency for each type of chest pain")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(['No Disease ', 'Heart disease']);
```

</div>

<div class="cell markdown">

### Finding some correlations

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:02.756228Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:01.288419Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:01.288465Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:02.756944Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:01.287789Z&quot;}" data-trusted="true">

``` python
fig, ax = plt.subplots(figsize=(15,15))
ax=sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.2f', cmap='YlGnBu',ax=ax)
ax.set_title("Correlation between variables",fontsize=20);
```

</div>

<div class="cell markdown">

## 5\. Modelling

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:02.767216Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:02.758639Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:02.758676Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:02.767935Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:02.758041Z&quot;}" data-trusted="true">

``` python
# Split the data into X and y
X = df.drop(['target'], axis=1)
y = df.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:02.795406Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:02.769585Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:02.769625Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:02.796302Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:02.768982Z&quot;}" data-trusted="true">

``` python
X_train
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:02.809699Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:02.798199Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:02.798233Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:02.810404Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:02.797878Z&quot;}" data-trusted="true">

``` python
y_train.value_counts()
```

</div>

<div class="cell markdown">

### To get the right model, we follow the sklearn's model selection map.

You can find the map here:
<https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html>

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:03.210724Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:02.812085Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:02.812118Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:03.211567Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:02.811521Z&quot;}" data-trusted="true">

``` python
models={
    'Logistic Regression':LogisticRegression(max_iter=1000),
    'KNN':KNeighborsClassifier(),
    'Random Forest':RandomForestClassifier()
}

# creating a function to evaluate the models
def fit_and_score(models,X_train,X_test,y_train,y_test):
    """
    fits and evaluates given machine learing models
    models : a dict of different Sklearn machine larning models
    X_train : training data(No labels)
    X_test : test data(No labels)
    y_train : training labels
    y_test : test labels
    """
    np.random.seed(42)
    model_scores = {}
    for name,model in models.items():
        model.fit(X_train,y_train)
        model_scores[name]=model.score(X_test,y_test)
    return model_scores
model_scores=fit_and_score(models,X_train,X_test,y_train,y_test)
model_scores
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:03.422209Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:03.21292Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:03.212949Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:03.423296Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:03.212716Z&quot;}" data-trusted="true">

``` python
model_scores_df=pd.DataFrame(model_scores,index=['Accuracy'])
model_scores_df.T.plot.bar(figsize=(10,5));
```

</div>

<div class="cell markdown">

## 6\. Hyperparameter Tuning

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:03.812507Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:03.424818Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:03.424853Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:03.81321Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:03.424581Z&quot;}" data-trusted="true">

``` python
train_scores=[]
test_scores=[]
neighbours=range(1,20)
knn=KNeighborsClassifier()
for n in neighbours:
    knn.set_params(n_neighbors=n)
    knn.fit(X_train,y_train)
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:03.82207Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:03.815484Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:03.815536Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:03.82326Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:03.81451Z&quot;}" data-trusted="true">

``` python
train_scores
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:03.836261Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:03.825374Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:03.825406Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:03.837047Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:03.82511Z&quot;}" data-trusted="true">

``` python
test_scores
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:04.162981Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:03.839524Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:03.839572Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:04.163775Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:03.838536Z&quot;}" data-trusted="true">

``` python
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(neighbours,train_scores,label='Training Score',color='blue')
ax.plot(neighbours,test_scores,label='Test Score',color='red')
ax.set_title("KNN Score vs Number of Neighbours")
ax.set_xlabel("Number of Neighbours")
ax.set_ylabel("Score")
ax.set_xticks(neighbours)
ax.legend(["Training Score","Test Score"]);
print(f"Maximum KNN Testing Score: {max(test_scores)*100:.2f}%")
```

</div>

<div class="cell markdown">

We saw highest `KNN` score is less than `Logistic Regression` and
`Random Forest` score. So, we will leave KNN and will try to tune
Logistic Regression and Random Forest.

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:04.171095Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:04.165569Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:04.165603Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:04.172102Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:04.165135Z&quot;}" data-trusted="true">

``` python
# Create a hpyerparameter grid for logistic regression
log_reg_grid = {'penalty':['l1','l2'],
                'C':np.logspace(-4,4,20),
                'solver':['liblinear']}

# Create a hpyerparameter grid for Random Forest Classifier
rf_grid = {'n_estimators': np.arange(10,1000,50),
           'max_depth':["None",3,5,10],
           'min_samples_split': np.arange(2,20,20),
           'min_samples_leaf': np.arange(1,20,2)}
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:04.185942Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:04.178547Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:04.178583Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:04.186856Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:04.178258Z&quot;}" data-trusted="true">

``` python
# Hypertune LogisticRegression
np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                n_iter=20,
                                cv=5,
                                verbose=2)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:05.004662Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:04.189774Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:04.189826Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:05.005545Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:04.188822Z&quot;}" data-trusted="true">

``` python
rs_log_reg.fit(X_train,y_train)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:05.014112Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:05.007859Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:05.007913Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:05.015054Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:05.007148Z&quot;}" data-trusted="true">

``` python
rs_log_reg.best_params_
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:05.030421Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:05.016693Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:05.016726Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:05.031509Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:05.016463Z&quot;}" data-trusted="true">

``` python
rs_log_reg.score(X_test,y_test)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:42:05.042265Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:05.033771Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:05.033821Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:42:05.043026Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:05.032911Z&quot;}" data-trusted="true">

``` python
# Hypertune RandomForestClassifier
np.random.seed(42)
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                            param_distributions=rf_grid,
                            n_iter=20,
                            cv=5,
                            verbose=True)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:16.903593Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:42:05.04479Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:42:05.044831Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:16.904484Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:42:05.044504Z&quot;}" data-trusted="true">

``` python
rs_rf.fit(X_train,y_train)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:16.912285Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:16.906198Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:16.906226Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:16.913352Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:16.905962Z&quot;}" data-trusted="true">

``` python
rs_rf.best_params_
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:17.009737Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:16.915091Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:16.915126Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:17.010903Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:16.914881Z&quot;}" data-trusted="true">

``` python
rs_rf.score(X_test,y_test)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:18.603572Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:17.012364Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:17.012395Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:18.605524Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:17.012106Z&quot;}" data-trusted="true">

``` python
# GridSearchCV for LogisticRegression
np.random.seed(42)
log_reg_grid = {'penalty':['l1','l2'],
                'C':np.logspace(-4,4,20),
                'solver':['liblinear']}
gs_log_reg = GridSearchCV(LogisticRegression(),
                            param_grid=log_reg_grid,                                                    
                            cv=5,
                            verbose=True)
gs_log_reg.fit(X_train,y_train)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:18.613543Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:18.607501Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:18.607535Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:18.614607Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:18.607239Z&quot;}" data-trusted="true">

``` python
gs_log_reg.best_params_
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:18.628998Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:18.616913Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:18.61696Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:18.629796Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:18.616394Z&quot;}" data-trusted="true">

``` python
gs_log_reg.score(X_test,y_test)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:18.638637Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:18.631749Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:18.631788Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:18.639376Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:18.63096Z&quot;}" data-trusted="true">

``` python
# #### GridSearchCV for RandomForestClassifier
# np.random.seed(42)
# rf_grid = {'n_estimators': np.arange(10,1000,20),
#             'min_samples_split': np.arange(2,20,15),
#             'min_samples_leaf': np.arange(1,20,2)}
# gs_rf = GridSearchCV(RandomForestClassifier(),
#                     param_grid=rf_grid, 
#                     cv=3,
#                     verbose=True)
# gs_rf.fit(X_train,y_train)
```

</div>

<div class="cell markdown">

## 7\. Evaluating the model

  - roc curve
  - confusion matrix
  - classification report
  - precision
  - recall
  - f1 score

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:18.890972Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:18.641836Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:18.641882Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:18.892005Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:18.640721Z&quot;}" data-trusted="true">

``` python
# plot Roc curve and calculate AUC
plot_roc_curve(gs_log_reg,X_test,y_test);
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:18.904938Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:18.895528Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:18.895582Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:18.906288Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:18.894316Z&quot;}" data-trusted="true">

``` python
confusion_matrix(y_test,gs_log_reg.predict(X_test))
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:19.106479Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:18.908432Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:18.908481Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:19.107402Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:18.908056Z&quot;}" data-trusted="true">

``` python
sns.set(font_scale=1.5)
fig,ax=plt.subplots(figsize=(5,5))
ax=sns.heatmap(confusion_matrix(y_test,gs_log_reg.predict(X_test)),
                                annot=True,
                                cbar=False)
ax.set_title("Confusion Matrix",fontsize=20)
ax.set_ylabel("Predicted Label")
ax.set_xlabel("True Label");
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:19.12324Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:19.10889Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:19.108921Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:19.124273Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:19.10867Z&quot;}" data-trusted="true">

``` python
# classification report
print(classification_report(y_test,gs_log_reg.predict(X_test)))
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:48.412595Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:19.127143Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:19.12722Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:48.413837Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:19.126665Z&quot;}" data-trusted="true">

``` python
eval_metrics=[
    'accuracy','precision','recall','f1'    
]
eval_metrics_results={}
for metric in eval_metrics:
    eval_metrics_results[metric]=cross_val_score(gs_log_reg,X_train,y_train,cv=5,scoring=metric).mean()
eval_metrics_df=pd.DataFrame(eval_metrics_results,index=['Mean'])
eval_metrics_df
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-01T18:43:48.699336Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-01T18:43:48.416816Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-01T18:43:48.416863Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-01T18:43:48.70012Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-01T18:43:48.41642Z&quot;}" data-trusted="true">

``` python
eval_metrics_df.T.plot.bar(figsize=(10,5),color='blue');
plt.title('Evaluation Metrics')
```

</div>
