import os
import matplotlib.pyplot as plt

def getFeatList(fileName):

    fid = open(fileName, 'r')
    Lines = fid.readlines()

    featList = []
    for line in Lines:
        split = line.strip().split()
        if len(split) > 1:
            featList.append(split[0].lower())

    return featList

def AlexComputer():
    if os.getcwd() == r'C:\Users\Alex\PycharmProjects\DeepEarnings':
        return True
    else:
        return False

def MakePlot(history):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.legend(['train', 'val'], loc='upper right')
    plt.show()