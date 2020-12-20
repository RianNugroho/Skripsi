import numpy as np
import math
import cv2
import pandas as pd


def ReLU(layer):
    results=np.zeros(layer.shape)
    for i in range(len(layer)):
        for j in range(len(layer[i])):
            results[i,j]=max(0,layer[i,j])

    return results



def SoftMax(layer):
    sumExp=0
    for i in range(len(layer)):
        for j in range(len(layer[i])):

            sumExp+=math.exp(layer[i,j])
    results=np.exp(layer)/sumExp
    return results

def Dense(input_layer,weights,biases):
    results=np.dot(weights,input_layer)+biases
    return results

def normalize(img):
    return img/255

def convolve(img, filter,num_units,padding="valid", strides=1,padding_size=0):
    if (padding=="same"):
        results=np.zeros((img.shape[0],img.shape[1],num_units))
    else:
        dim_kiri=int(math.floor(((img.shape[0]+2*padding_size-filter.shape[1])/strides)+1))
        dim_kanan=int(math.floor(((img.shape[1]+2*padding_size-filter.shape[2])/strides)+1))
        results=np.zeros((dim_kiri,dim_kanan,num_units))

    for units in range(num_units):
        i=0
        while(i<results.shape[0]):
            j=0
            while(j<results.shape[1]):
                for k in range(len(filter[units])):
                    for l in range(len(filter[units,k])):
                        for m in range(len(filter[units,k,l])):
                            shiftIndex=0
                            if(padding=="valid"):
                                shiftIndex+=int(filter.shape[0]/2)-padding_size
                            try:
                                results[int(i/strides),int(j/strides), units] += img[i + k - 1 + shiftIndex , j + l - 1 + shiftIndex , int(m*num_units)] * filter[units,k, l, m]
                            except Exception as e:
                                results[int(i/strides),int(j/strides), units] += 0
                if(results[int(i/strides),int(j/strides), units]>1.0):
                    results[int(i / strides), int(j / strides), units]=1.0
                if(results[int(i/strides),int(j/strides), units]<0):
                    results[int(i / strides), int(j / strides), units]=0
                j+=strides
            i+=strides

    return results



def maxpool(img, size):
    result=np.zeros((int(math.ceil(img.shape[0]/size)),int(math.ceil(img.shape[1]/size)), img.shape[2]))
    for unit in range(img.shape[2]):
        for i in range(len(result)):
            for j in range(len(result[i])):
                try:
                    result[i, j, unit] = np.amax(img[i:i + size, j:j + size, unit])
                except:
                    result[i, j, unit] = np.amax(img[i:, j:, unit])
    return result


def Inception(input_layer, filters):
    result=np.concatenate((convolve(input_layer,filters[0],filters[0].shape[0],padding="same"),
                           convolve(input_layer,filters[1],filters[1].shape[0],padding="same"),
                           convolve(input_layer,filters[2],filters[2].shape[0],padding="same")),axis=2)
    return result

def backward(layers, target):
    results=[]
    ##update layer 12
    crossentropyLoss=np.sum(np.multiply(target,np.log(layers[11]))+np.multiply((1-target),np.log(1-layers[11])))
    expSumSoftmax=np.sum(np.exp(layers[11]))
    dSoftmax=np.multiply(layers[11],expSumSoftmax-layers[11])/math.pow(expSumSoftmax,2)

    return results


def main():

    datasets = pd.read_csv("fer2013.csv")
    x = np.array([[int(pix) for pix in image.split()]
                  for image in datasets.pixels]).reshape(-1, 48, 48, 1)
    filter1 = np.random.uniform(low=-3, high=3, size=(3, 3, 3, 1))
    filter2 = np.random.uniform(low=-3, high=3, size=(10, 3, 3, 5))
    filter3 = np.random.uniform(low=-3, high=3, size=(3, 3, 3, 10))
    filter4 = np.random.uniform(low=-3, high=3, size=(1, 3, 3, 3))

    sobel_filter_vertical = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]).reshape((1,3,3,1))
    sobel_filter_horizontal = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).reshape((1,3,3,1))
    filterInception = [filter1, sobel_filter_vertical, sobel_filter_horizontal]
    print(x.shape)
    for i in range(50):
        for j in range(len(x)):
            layer1 = Inception(x[j], filterInception)
            layer2 = convolve(layer1, filter2, 10, padding="valid")
            layer3 = maxpool(layer2, 2)
            layer4 = convolve(layer3, filter3, 3, padding="valid")
            layer5 = maxpool(layer4, 2)
            layer6 = convolve(layer5, filter4, 1, padding="valid")
            layer7 = maxpool(layer6, 2)
            layer8 = layer7.flatten()
            print(layer8.shape)

np.random.seed(0)
datasets = pd.read_csv("fer2013.csv")

x = np.array([[int(pix) for pix in image.split()]
              for image in datasets.pixels]).reshape(-1, 48, 48, 1)

filter1 = np.random.uniform(low=-3, high=3, size=(3, 3, 3, 1))
sobel_filter_vertical = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]).reshape((1,3,3,1))
sobel_filter_horizontal = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).reshape((1,3,3,1))
filterInception = [filter1, sobel_filter_vertical, sobel_filter_horizontal]



filter2 = np.random.uniform(low=-1, high=1, size=(3, 1, 1, 5))
filter3 = np.random.uniform(low=-1, high=1, size=(4, 3, 3, 5))
filter4 = np.random.uniform(low=-1, high=1, size=(3, 5, 5, 5))
filter5 = np.random.uniform(low=-1, high=1, size=(10, 3, 3, 10))
filter6 = np.random.uniform(low=-1, high=1, size=(3, 3, 3, 10))
filter7 = np.random.uniform(low=-1, high=1, size=(4, 3, 3, 3))


weight1=np.random.uniform(low=-1,high=1,size=(50,100))
weight2=np.random.uniform(low=-1,high=1,size=(10,50))

bias1=np.random.uniform(low=-1,high=1,size=(50,1))
bias2=np.random.uniform(low=-1,high=1,size=(10,1))


img=normalize(x[1])
layer1=Inception(img, filterInception)
print("layer1",layer1.shape)
layer2=Inception(layer1, [filter2,filter3,filter4])
print("layer2",layer2.shape)
layer3=convolve(layer2,filter5,10,padding="valid")
layer4=maxpool(layer3,2)
print("layer4",layer4.shape)
layer5=convolve(layer4,filter6,3,padding="valid")
layer6=maxpool(layer5,2)
print("layer6",layer6.shape)
layer7=convolve(layer6,filter7,4,padding="valid")
layer8=maxpool(layer7,2)
print(layer8.shape)
layer9=layer8.reshape((100,1))
layer10=Dense(layer9,weight1,bias1)
layer11=ReLU(layer10)
print(layer11.shape)
layer12=Dense(layer11,weight2,bias2)
layer13=SoftMax(layer12)
print(layer13.shape)
print(layer13)