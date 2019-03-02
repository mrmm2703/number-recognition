print "Importing modules..."
import csv
import numpy as np
from sklearn import svm
import os.path
from sklearn.externals import joblib
from PIL import Image
import matplotlib.pyplot as plt
import random
import time
print "Modules successfully imported."
print ""

x = []
y = []
counter = 0

if os.path.isfile("model.joblib") is False:
    print "Model not already created. Creating model..."
    print "Opening database..."
    with open('train.csv', 'rb') as trainFile:
        csvfile = csv.reader(trainFile)
        print "Loading database into memory..."
        for row in csvfile:
            counter = counter + 1
            if counter == 1:
                continue
            shortArray = []
            y.append(int(row[0]))
            counter2 = 0
            for column in row:
                counter2 = counter2 + 1
                if counter2 == 1:
                    continue
                try:
                    shortArray.append(float(1.0 / 255.0) * float(column))
                except:
                    shortArray.append(0)
            x.append(shortArray)
        print "Database loaded."
        print ""

    print "Creating model..."
    reg = svm.SVC(gamma='scale')
    reg.fit(x, y)
    print "Model created."
    joblib.dump(reg, "model.joblib")
else:
    reg = joblib.load("model.joblib")

testArray = np.array([])
testArray256 = np.array([])
with open('test.csv', 'rb') as file:
    counter2 = 0
    csvfile = csv.reader(file)
    for i, row in enumerate(csvfile):
        if i == 24163:
            for column in row[1:]:
                testArray = np.append(testArray, (float(1.0 / 255.0) * float(column)))
                testArray256 = np.append(testArray256, int(column))

print ""
print "Testing image..."
print str(str(reg.predict([testArray]))[1]) + " is the predicted number."
testArray256 = testArray256.reshape(28, 28)
plt.matshow(testArray256)
plt.show()
