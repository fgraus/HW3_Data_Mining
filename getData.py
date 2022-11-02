import pandas as pd



def getIrisData():
    irisData = pd.read_csv("Resources/Iris/iris.txt", sep=" ", header=None)
    irisData.columns = ["sepal_L","sepal_W","petal_L","petal_W"]
    # normalize elements
    for feature_name in irisData.columns:
        max_value = irisData[feature_name].max()
        min_value = irisData[feature_name].min()
        irisData[feature_name] = (irisData[feature_name]- min_value) / (max_value - min_value)
    irisData['center'] = -1
    return irisData

def getNumbersData():
    imageNumber = pd.read_csv("Resources/ImageNumbers/Numbers.txt", sep=",", header=None)
    # normalize elements
    nunique = imageNumber.nunique()
    cols_to_drop = nunique[nunique == 1].index
    imageNumber = imageNumber.drop(cols_to_drop, axis=1)
    imageNumber = imageNumber/255
    imageNumber['center'] = -1
    return imageNumber

getNumbersData()

