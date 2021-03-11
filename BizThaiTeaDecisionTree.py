import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from numpy import asarray
from sklearn import tree
import pydotplus
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
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


    weatherValue = ['sunny', 'rainy']
    weather_label = preprocessing.LabelEncoder()
    weather_label.fit(weatherValue)
    for check in X[:, 2] :
        if check not in weatherValue :
            giveWarning = True

    if giveWarning :
        print("Ada kesalahan di penamaan genre")
        return False
    else:

        X[:, 2] = weather_label.transform(X[:, 2])

    isThirstyValue = ['Yes', 'No']
    label_isThirsty = preprocessing.LabelEncoder()
    label_isThirsty.fit(isThirstyValue)
    for check in X[:, 3]:
        if check not in isThirstyValue:
            giveWarning = True

    if giveWarning:
        print("Ada kesalahan di penamaan isThirsty")
        return False
    else:

        X[:, 3] = label_isThirsty.transform(X[:, 3])

    y = datainput["Class"]

    # train_test_split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

    drugTree.fit(X_train, y_train)
    prediction = drugTree.predict(X_test)

    print(prediction)

    print(metrics.classification_report(y_test, prediction))

    print("\nDecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, prediction))
    # precision tp / (tp + fp)

    print("\n DecisionTrees's Precision: ", metrics.precision_score(y_test, prediction, average="macro"))
    # recall: tp / (tp + fn)

    recall = metrics.recall_score(y_test, prediction, average="macro")
    print("\n DecisionTree's Recall: ", recall)
    # f1: 2 tp / (2 tp + fp + fn)

    f1 = metrics.f1_score(y_test, prediction, average="macro")
    print("\n DecisionTree's F1: ", f1)

    from sklearn.preprocessing import MinMaxScaler
    scalar = MinMaxScaler()

    normArray = scalar.fit_transform(asarray(X))
    normDataInput = pd.DataFrame(normArray, columns= datainput[['Age','Gender','Weather', 'isThirsty']].columns)
    print("Model Scalability: \n",normDataInput.head())

    # Show Image
    data = tree.export_graphviz(drugTree, out_file=None, feature_names=['Age','Gender','Weather', 'isThirsty'])
    graph = pydotplus.graph_from_dot_data(data)
    graph.write_png('mydecisiontree.png')

    img = pltimg.imread('mydecisiontree.png')
    plt.imshow(img)
    plt.show()

cProfile.run('efficiencyTest()')



