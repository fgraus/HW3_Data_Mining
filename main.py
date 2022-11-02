import numpy as np
from getData import getIrisData
from getData import getNumbersData


def main(iris = False, kNumber = 10):

    if(iris):
        data = getIrisData()
        data_clustered = k_means(data,kNumber)
        validation_number = validate_with_silhouette(data_clustered)
        print(validation_number)
        #writeData(data)
    else:
        data = getNumbersData()
        data_clustered = k_means(data,kNumber, iris=False)
        validation_number = validate_with_silhouette_image(data_clustered)
        writeData(data)
        print(validation_number)

    return 0

def k_means(data, kNumber, iris = True):
    centers_points = getInitialCenters(data, kNumber, randomMethod=True)
    stop = False
    iteration = 0
    while(not stop):
        data = assignPoints(data, centers_points, iris)
        old_points = centers_points.copy()
        centers_points = computeNewCenters(data,centers_points)
        stop = iteration >= 10 or checkIfContinue(old_points, centers_points)
        iteration = iteration + 1
        print('Iteracion: ' + str(iteration))
    
    return data

def getInitialCenters(data,kNumber, randomMethod = False, k_means_plus = False, bisecting_k_means = False):
    if(randomMethod):
        # return n random samples from the data
        return data.sample(n = kNumber)
    if(k_means_plus):
        centers = data.sample(n = 1)
        for i in range(kNumber-1):
            for center in range(centers.shape[0]):
                for point in range(data.shape[0]):
                    print('nais')

            print('nada')

        return 0
    if(bisecting_k_means):
        return 0
    return 0

def assignPoints(data, centers_points, iris = True):
    for element in range(data.shape[0]):
        nearest_point = 0
        nearest_distance = 99999999999999
        for center in range(centers_points.shape[0]):
            distance = computerDistance(data.iloc[element], centers_points.iloc[center], iris)
            if(distance < nearest_distance):
                nearest_point = center
                nearest_distance = distance
        data.at[element,'center'] = nearest_point
    return data

def computeNewCenters(data, centers_points):
    for center in range(centers_points.shape[0]):
        clouster = data[data.center == center]
        for feature in range(centers_points.shape[1]-1):
            # get the media of the feature value
            mean = clouster.iloc[:,[feature]].mean().iloc[0]
            centers_points.at[centers_points.iloc[center].name, centers_points.columns[feature]] = mean
    return centers_points

def checkIfContinue(old_points, new_points):
    return old_points.equals(new_points)

def computerDistance(data, possibleCenter, iris):
    distance = 0
    # if iris we use euclidean distance
    if (iris):
        for i in range(data.shape[0]-1):
            distance += (possibleCenter.iloc[i] - data.iloc[i])**2
    # if images we use cos similarity
    else:
        dat = data.iloc[:len(data)-1].values
        cen = possibleCenter.iloc[:len(data)-1].values
        distance = 1 - np.dot(dat,cen) / (np.sqrt(np.dot(dat,dat))*np.sqrt(np.dot(cen,cen)))
    return distance

def validate_with_silhouette(data):
    validate = 0
    for element in range(data.shape[0]):
        a = 0
        b = 0
        for feature in range(data.shape[1]-1):
            value = data.iloc[element].iloc[feature]
            clouster = data[data.center == data.iloc[element].center]
            notClouster = data[data.center != data.iloc[element].center]
            a += abs((clouster.iloc[:,feature] - value).mean())
            b += abs((notClouster.iloc[:,feature] - value).mean())
        
        a = a/(data.shape[1]-1)
        b = b/(data.shape[1]-1)

        validate += (b-a)/max(a,b)

    return validate / data.shape[0]

def validate_with_silhouette_image(data):
    validate = 0
    for element in range(data.shape[0]):
        a = 0
        b = 0
        clouster = data[data.center == data.iloc[element].center]
        notClouster = data[data.center != data.iloc[element].center]
        for i in range(clouster.shape[0]):
            a += computerDistance(data.iloc[element], clouster.iloc[i], iris=False)
        a = a/clouster.shape[0]
        for i in range(notClouster.shape[0]):
            b += computerDistance(data.iloc[element], notClouster.iloc[i], iris=False)
        b = b/notClouster.shape[0]

        validate += (b-a)/max(a,b)

    return validate / data.shape[0]

def writeData(data):
    file = open('Resources/iris.dat','w')
    for i in range(data.shape[0]):
        file.write(str(int(data.iloc[i].iloc[-1] + 1)) + '\n')
    file.close()

main()
