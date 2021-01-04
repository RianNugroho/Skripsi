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

def maxPoolDerivative(dFront,prevLayer,next_layer,max_pool_size):
    dCurrent=np.zeros(prevLayer.shape)
    print("prevLayer",prevLayer.shape,"nextLayer",next_layer.shape)
    for unit in range(dFront.shape[2]):
        for i in range(dFront.shape[0]):
            for j in range(dFront.shape[1]):
                startX=max_pool_size*i
                startY=max_pool_size*j
                for k in range(max_pool_size):
                    for l in range(max_pool_size):
                        if(prevLayer[startX-1+k,startY-1+l,unit]==next_layer[i,j,unit]):
                            dCurrent[startX-1+k,startY-1+l,unit]=1
    return dCurrent

def ConvolveFilterDerivative(dFront, dPrev):
    dim_kiri = dPrev.shape[0]-dFront.shape[0]+1
    dim_kanan = dPrev.shape[1]-dFront.shape[1]+1
    result=np.zeros((dFront.shape[2],dim_kiri,dim_kanan,dPrev.shape[2]))
    for unit in range(dFront.shape[2]):
        for i in range(dim_kiri):
            for j in range(dim_kanan):
                for channel in range(dPrev.shape[2]):
                    for k in range(dFront.shape[0]):
                        for l in range(dFront.shape[1]):
                            result[unit,i,j,channel]+=dPrev[i+k,j+l,channel]*dFront[k,l,unit]
    return result

def ConvolveInputDerivative(dFront,Filter:



def backward(layers, target, weights,max_pool_size):
    resultWeight=[]
    resultBias=[]
    resultFilter=[]


    ##update Softmax layer
    ##crossentropyLoss=np.sum(np.multiply(target,np.log(layers[12]))+np.multiply((1-target),np.log(1-layers[12])))*-1/len(layers[12])
    print(layers[12].shape)
    print((1-layers[12]).shape)
    dPred=-1*np.multiply(target,1/layers[12])+np.multiply((1-target),1/(1-layers[12]))
    expSumSoftmax=np.sum(np.exp(layers[11]))
    dL13=np.multiply(layers[11],expSumSoftmax-layers[11])/math.pow(expSumSoftmax,2)
    dL12=layers[10]
    print(dPred.shape,dL13.shape,dL12.T.shape)
    dW2=np.dot(dPred*dL13,dL12.T)
    dB2=dPred*dL13
    dFront=dPred*dL13
    print("dw2",dW2.shape,"dB2",dB2.shape)
    resultWeight.append(dW2)
    resultBias.append(dB2)

    ##update Dense layer 1
    dFront=np.dot(weights[1].T,dFront)
    dL11=np.where(layers[9]>0,layers[9],1)
    dL10=layers[8]
    print("dFront",dFront.shape,"dL11",dL11.shape,"dL10",dL10.shape)
    dW1=np.dot(dFront*dL11,dL10.T)
    dB1=dFront*dL11
    resultWeight.append(dW1)
    resultBias.append(dB1)
    print(dFront.shape)

    ##update Convolution Layer ke 3
    dFront=np.dot(weights[0].T,dFront)
    dFront=dFront.reshape(layers[7].shape)
    print("dFront",dFront.shape)
    dL8=maxPoolDerivative(dFront,layers[6],layers[7],max_pool_size[2])
    dL7=layers[5]
    print("dFront", dFront.shape, "dL8", dL8.shape, "dL7", dL7.shape)
    dFilter5=ConvolveFilterDerivative(dL8,dL7)
    resultFilter.append(dFilter5)
    dFront=np.copy(dL8)

    ##update Convolution Layer ke 2

    dL6 = maxPoolDerivative(dFront, layers[4], layers[5], max_pool_size[1])
    dL5 = layers[3]
    dFilter4 = dFront * dL6 * dL5
    resultFilter.append(dFilter4)
    dFront *= dL6

    ##update Convolution Layer ke 1
    dL4 = maxPoolDerivative(dFront, layers[2], layers[3], max_pool_size[0])
    dL3 = layers[1]
    dFilter3 = dFront * dL4 * dL3
    resultFilter.append(dFilter3)
    dFront *= dL4

    ##update Inception Layer ke 2
    print(dFront.shape)

    return resultWeight,resultBias,resultFilter




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
layers=[]
layer1=Inception(img, filterInception)
layers.append(layer1)
print("layer1",layer1.shape)
layer2=Inception(layer1, [filter2,filter3,filter4])
layers.append(layer2)
print("layer2",layer2.shape)
layer3=convolve(layer2,filter5,10,padding="valid")
layers.append(layer3)
layer4=maxpool(layer3,2)
layers.append(layer4)
print("layer4",layer4.shape)
layer5=convolve(layer4,filter6,3,padding="valid")
layers.append(layer5)
layer6=maxpool(layer5,2)
layers.append(layer6)
print("layer6",layer6.shape)
layer7=convolve(layer6,filter7,4,padding="valid")
layers.append(layer7)
layer8=maxpool(layer7,2)
layers.append(layer8)
print(layer8.shape)
layer9=layer8.reshape((100,1))
layers.append(layer9)
layer10=Dense(layer9,weight1,bias1)
layers.append(layer10)
layer11=ReLU(layer10)
layers.append(layer11)
print(layer11.shape)
layer12=Dense(layer11,weight2,bias2)
layers.append(layer12)
layer13=SoftMax(layer12)
layers.append(layer13)
print(layer13.shape)
print(layer13)
target=np.array([1,0,0,0,0,0,0,0,0,0]).reshape((10,1))
a,b,c=backward(layers,target,[weight1,weight2],[2,2,2])