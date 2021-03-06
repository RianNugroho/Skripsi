import numpy as np
import math
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

def Normalize(img):
    return img/255

def Convolve(img, filter,num_units,padding="valid", strides=1,padding_size=0):
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



def Maxpool(img, size):
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
    result=np.concatenate((Convolve(input_layer,filters[0],filters[0].shape[0],padding="same"),
                           Convolve(input_layer,filters[1],filters[1].shape[0],padding="same"),
                           Convolve(input_layer,filters[2],filters[2].shape[0],padding="same")),axis=2)
    return result

def MaxpoolDerivative(dFront,prevLayer,next_layer,max_pool_size):
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

def Add_padding_dFront(layer,desired_result_size):
    tambahKiri=int((desired_result_size[0]-1)/2)
    tambahKanan=int((desired_result_size[1]-1)/2)
    dim_kiri=layer.shape[0]+tambahKiri*2
    dim_kanan=layer.shape[1]+tambahKanan*2
    result=np.zeros((dim_kiri,dim_kanan,layer.shape[2]))
    result[tambahKiri:result.shape[0]-tambahKiri,tambahKanan:result.shape[1]-tambahKanan,:]=np.copy(layer)
    return result

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

def FullConvolutionDerivative(dFront,Filter, padding="valid"):
    if(padding=="valid"):
        dim_kiri = dFront.shape[0] + Filter.shape[1] - 1
        dim_kanan = dFront.shape[1] + Filter.shape[2] - 1
        result = np.zeros((dim_kiri, dim_kanan, Filter.shape[3]))
        filter = np.rot90(Filter, 2)
        for unit in range(filter.shape[0]):
            for i in range(dim_kiri):
                for j in range(dim_kanan):
                    for k in range(Filter.shape[1]):
                        for l in range(Filter.shape[2]):
                            for m in range(Filter.shape[3]):
                                try:
                                    result[i, j, m] += dFront[i - Filter.shape[1] + 1, j - Filter.shape[2] + 1, unit] * \
                                                       Filter[unit, k, l, m]
                                except:
                                    result[i, j, m] += 0
    elif(padding=="same"):
        result = np.zeros((dFront.shape[0],dFront.shape[1], Filter.shape[3]))
        filter = np.rot90(Filter, 2)
        padding = int((Filter.shape[0]-1)/2)
        for unit in range(filter.shape[0]):
            for i in range(result.shape[0]-padding):
                for j in range(result.shape[1]-padding):
                    for k in range(Filter.shape[1]):
                        for l in range(Filter.shape[2]):
                            for m in range(Filter.shape[3]):
                                try:
                                    result[i, j, m] += dFront[i - Filter.shape[1] + 1+padding, j - Filter.shape[2] + 1 +padding, unit] * \
                                                       Filter[unit, k, l, m]
                                except:
                                    result[i, j, m] += 0

    return result

def AdamOptim(t,v,s,beta1,beta2,epsilon,derivative):
    newV=beta1*v+(1-beta1)*derivative
    newS=beta2*s+(1-beta2)*np.dot(derivative,derivative)
    Vcorr=newV/(1-int(math.pow(beta1,t)))
    Scorr=newS/(1-int(math.pow(beta2,t)))
    result=Vcorr/(np.sqrt(Scorr)+epsilon)

    return result,newV,newS


def Backward(layers, target, weights,max_pool_size, filters, input):
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
    print("dw1", dW1.shape, "dB1", dB1.shape)
    resultWeight.append(dW1)
    resultBias.append(dB1)
    print(dFront.shape)

    ##update Convolution Layer ke 3
    dFront=np.dot(weights[0].T,dFront)
    dFront=dFront.reshape(layers[7].shape)
    print("dFront",dFront.shape)
    dL8=MaxpoolDerivative(dFront,layers[6],layers[7],max_pool_size[2])
    dL7=layers[5]
    print("dFront", dFront.shape, "dL8", dL8.shape, "dL7", dL7.shape)
    dFilter7=ConvolveFilterDerivative(dL8,dL7)
    print("dFilter7",dFilter7.shape)
    resultFilter.append(dFilter7)
    dFront=FullConvolutionDerivative(dL8,filters[6])
    print(dFront.shape)

    ##update Convolution Layer ke 2

    dL6 = MaxpoolDerivative(dFront, layers[4], layers[5], max_pool_size[1])
    dL5 = layers[3]
    print("dFront", dFront.shape, "dL6", dL6.shape, "dL5", dL5.shape)
    dFilter6 = ConvolveFilterDerivative(dL6,dL5)
    resultFilter.append(dFilter6)
    print("dFilter6", dFilter6.shape)
    dFront =FullConvolutionDerivative(dL6,filters[5])
    print("dFront", dFront.shape)

    ##update Convolution Layer ke 1
    dL4 = MaxpoolDerivative(dFront, layers[2], layers[3], max_pool_size[0])
    dL3 = layers[1]
    print("dFront", dFront.shape, "dL4", dL4.shape, "dL3", dL3.shape)
    dFilter5 = ConvolveFilterDerivative(dL4,dL3)
    resultFilter.append(dFilter5)
    print("dFilter5", dFilter5.shape)
    dFront = FullConvolutionDerivative(dL4,filters[4])

    ##update Inception Layer ke 2
    dL2=layers[0]
    print("dFront", dFront.shape, "dL2", dL2.shape)
    dFilter2=ConvolveFilterDerivative(dFront[:,:,:3],dL2)
    dFilter3=ConvolveFilterDerivative(dFront[:,:,3:7],Add_padding_dFront(dL2,(3,3)))
    dFilter4=ConvolveFilterDerivative(dFront[:,:,7:],Add_padding_dFront(dL2,(5,5)))
    a=FullConvolutionDerivative(dFront[:,:,:3],filters[1],padding="same")
    b=FullConvolutionDerivative(dFront[:,:,3:7],filters[2],padding="same")
    c=FullConvolutionDerivative(dFront[:,:,7:],filters[3],padding="same")
    dFront=a+b+c
    print("dFront",dFront.shape)
    print("dFilter2", dFilter2.shape, "dFilter3", dFilter3.shape, "dFilter4", dFilter4.shape)
    resultFilter.append(dFilter4)
    resultFilter.append(dFilter3)
    resultFilter.append(dFilter2)

    ##update layer Inception 1
    dL1=input
    print("dFront", dFront.shape, "dL1", dL1.shape)
    dFilter1=ConvolveFilterDerivative(dFront[:,:,:3],Add_padding_dFront(dL1,(3,3)))
    resultFilter.append(dFilter1)

    return resultWeight,resultBias,resultFilter




np.random.seed(0)
datasets = pd.read_csv("fer2013.csv")
print("len",len(datasets))
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


img=Normalize(x[1])
print(img.shape)
layers=[]
layer1=Inception(img, filterInception)
layers.append(layer1)
print("layer1",layer1.shape)
layer2=Inception(layer1, [filter2,filter3,filter4])
layers.append(layer2)
print("layer2",layer2.shape)
layer3=Convolve(layer2,filter5,10,padding="valid")
layers.append(layer3)
layer4=Maxpool(layer3,2)
layers.append(layer4)
print("layer4",layer4.shape)
layer5=Convolve(layer4,filter6,3,padding="valid")
layers.append(layer5)
layer6=Maxpool(layer5,2)
layers.append(layer6)
print("layer6",layer6.shape)
layer7=Convolve(layer6,filter7,4,padding="valid")
layers.append(layer7)
layer8=Maxpool(layer7,2)
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
a,b,c=Backward(layers,target,[weight1,weight2],[2,2,2],[filter1,filter2,filter3,filter4,filter5,filter6,filter7], img)