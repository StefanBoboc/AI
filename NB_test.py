import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB

from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

trainImage = []
trainLable = []

testImage = []

# subprogram ce citeste datele de test
def testTxt():
    file = open('./data/test.txt')

    firstLine = 0

    for line in file:
        if firstLine != 0:
            aux = line.strip()      # punem datele in lista
            testImage.append(aux)   # adaugam la lista 'testImage'
        else:
            firstLine = 1

    file.close()

# subprogram ce citeste si combina datele de train cu cele de validare
def trainTxt():
    # Train_start
    file = open('./data/train.txt');

    firstLine = 0

    for line in file:
        if firstLine != 0:
            aux = line.strip().split(',')   # scapam de '\n' de la final de rand
            trainImage.append(aux[0])   # adaugam numele imaginii la o lista
            trainLable.append(aux[1])   # adaugam lablel-ul imaginii la o lista
        else:
            firstLine = 1

    file.close()
    # Train_end

    # Validation_start
    file = open('./data/validation.txt');

    firstLine = 0

    for line in file:
        if firstLine != 0:
            aux = line.strip().split(',')   # scapam de '\n' de la final de rand
            trainImage.append(aux[0])   # adaugam numele imaginii la o lista
            trainLable.append(aux[1])   # adaugam lablel-ul imaginii la o lista
        else:
            firstLine = 1

    file.close()
    # Validation_start


# apelarea subprogramelor de citire date
trainTxt()
testTxt()

vectorTrain = []    # lista de dimensiune 4 pentru datele de train
vectorTest = []     # lista de dimensiune 4 pentru datele de test

# obtinerea imaginilor in lista de train folosind path-ul imaginilor
for name in trainImage:
    im = cv2.imread('./data/train+validation/'+name) # citirea imaginilor si transformarea lor in 4D

    # aducem lista din 4D in 2D,
    # pentru a putea fi folosita in modelele sklearn.naive_bayes
    im = im.flatten()
    vectorTrain.append(im)  # adaugam imaginea

# obtinerea imaginilor in lista de validare folosind path-ul imaginilor
for name in testImage:
    im = cv2.imread('./data/test/'+name) # citirea imaginilor si transformarea lor in 4D
    # aducem lista din 4D in 2D,
    # pentru a putea fi folosita in modelele sklearn.naive_bayes
    im = im.flatten()
    vectorTest.append(im) # adaugam imaginea

# transformarea listelor de date in np.array pentru a fi acceptate de model
vectorTrain = np.array(vectorTrain)
vectorTest = np.array(vectorTest)

# transformarea listei de label-uri in np.array pentru a fi acceptate de model
trainLable = np.array(trainLable)

model = MultinomialNB(alpha=1.0)
model.fit(vectorTrain, trainLable)
prediction = model.predict(vectorTest)

# acuratete = np.mean(prediction == validationLable)
# print("Acuratete: ",acuratete)

# subprogram pentru generarea fisierului csv cu predictiile
def generateSubmission():
    submission = open("submission.csv", "w")
    submission.write('id,label\n')
    for i in range(len(prediction)):
        submission.write(testImage[i]+','+prediction[i]+'\n')

    submission.close()

generateSubmission()


