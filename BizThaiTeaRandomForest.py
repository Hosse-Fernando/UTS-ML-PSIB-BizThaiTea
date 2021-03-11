import pandas as pd
from numpy import asarray
from sklearn import metrics
from sklearn.metrics import classification_report
import cProfile

def efficiencyTest():
    datainput = pd.read_csv("DataBizThaiTea.csv", delimiter=",")
    giveWarning = False
    X = datainput[['Age','Gender','Weather', 'isThirsty']].values


    # Data Preprocessing
    from sklearn import preprocessing

    age_value = ['young', 'middle', 'old']
    label_age = preprocessing.LabelEncoder()
    label_age.fit(age_value)

    for check in X[:, 0] :
        if check not in age_value :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan umur")
        return False
    else:
        X[:, 0] = label_age.transform(X[:, 0])


    genderValue = ['male', 'female']
    label_gender = preprocessing.LabelEncoder()
    label_gender.fit(genderValue)
    for check in X[:, 1] :
        if check not in genderValue :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan gender")
        return False
    else:
        X[:, 1] = label_gender.transform(X[:, 1])

    weatherValue = ['rainy', 'sunny']
    label_weather = preprocessing.LabelEncoder()
    label_weather.fit(weatherValue)
    for check in X[:, 2]:
        if check not in weatherValue:
            giveWarning = True

    if giveWarning:
        print("Ada kesalahan di penamaan weather")
        return False
    else:

        X[:, 2] = label_weather.transform(X[:, 2])

    isThirstyValue = ['Yes', 'No']
    label_thirsty = preprocessing.LabelEncoder()
    label_thirsty.fit(isThirstyValue)
    for check in X[:, 3]:
        if check not in isThirstyValue:
            giveWarning = True

    if giveWarning:
        print("Ada kesalahan di penamaan isThirsty")
        return False
    else:

        X[:, 3] = label_thirsty.transform(X[:, 3])

    y = datainput["Class"].values

    suggestedValue = ['Yes', 'No']
    decision_label = preprocessing.LabelEncoder()
    decision_label.fit(suggestedValue)
    for check in y:
        if check not in suggestedValue:
            giveWarning = True

    if giveWarning:
        print("Ada kesalahan di penamaan class")
        return False
    else:
        y = decision_label.transform(y)

    # train_test_split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier

    thaiTeaRandomForest = RandomForestClassifier(n_estimators=60, random_state=0)

    thaiTeaRandomForest.fit(X_train, y_train)

    prediction = thaiTeaRandomForest.predict(X_test)

    print(prediction)

    print(classification_report(y_test, prediction))

    print("\nRandomForest's Accuracy: ", metrics.accuracy_score(y_test, prediction))
    # precision tp / (tp + fp)
    print("\n RandomForest's Precision: ", metrics.precision_score(y_test, prediction, average="macro"))
    # recall: tp / (tp + fn)
    recall = metrics.recall_score(y_test, prediction, average="macro")
    print("\n RandomForest's Recall: ", recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = metrics.f1_score(y_test, prediction, average="macro")
    print("\n RandomForest's F1: ", f1)

    from sklearn.preprocessing import MinMaxScaler
    scalar = MinMaxScaler()

    normArray = scalar.fit_transform(asarray(X))
    normDataInput = pd.DataFrame(normArray, columns= datainput[['Age','Gender','Weather', 'isThirsty']].columns)
    print("Model Scalability: \n",normDataInput.head())



cProfile.run('efficiencyTest()')

