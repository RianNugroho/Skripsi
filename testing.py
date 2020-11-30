import math

def matmul(a,b):
    print(len(a[0]), len(b))
    if(len(a[0])!=len(b)):
        print("(",len(a),len(a[0]),")","tidak sama dengan","(",len(b),len(b[0]),")")
        return
    else:
        res = []
        for i in range(len(a)):
            temp = []
            for j in range(len(b[0])):
                temp += [0]
            res += [temp]

        for i in range(len(b[0])):
            for j in range(len(a)):
                for k in range(len(b)):
                    res[j][i]+=a[j][k]*b[k][i]

    return res

def addPadding(imgChannels, addition):
    res=[]
    addition=int(addition)
    for i in range(len(imgChannels)+addition*2):
        temp=[]
        for j in range(len(imgChannels[0])+addition*2):
            temp1=[]
            for k in range(len(imgChannels[0][0])):
                temp1+=[0]
            temp+=[temp1]
        res+=[temp]
    for i in range(len(imgChannels)):
        for j in range(len(imgChannels[i])):
            for k in range(len(imgChannels[i][j])):
                res[i + (addition)][j + (addition)] [k]= imgChannels[i][j][k]
    return res

def convolve(imgChannels, kernels, same_padding=True, strides=0):
    if (len(imgChannels[0][0]) != len(kernels[0][0])):
        print("Jumlah channel tidak sama")
        return
    else:
        addition = 0
        res = []
        if (same_padding):
            for i in range(len(imgChannels) + addition * 2):
                temp = []
                for j in range(len(imgChannels[0]) + addition * 2):
                    temp1 = []
                    for k in range(len(kernels)):
                        temp1 += [0]
                    temp += [temp1]
                res += [temp]
            addition = (len(kernels[0]) - 1) / 2
            imgChannels = addPadding(imgChannels, addition)
        else:
            for i in range(len(imgChannels)-len(kernels[0])+1):
                temp = []
                for j in range(len(imgChannels[0])-len(kernels[0][0])+1 ):
                    temp1 = []
                    for k in range(len(kernels)):
                        temp1 += [0]
                    temp += [temp1]
                res += [temp]
        print(len(res), len(res[0]), len(res[0][0]))
        for num_kernels in range(len(kernels)):
            for i in range(len(res)):
                for j in range(len(res[i])):
                    for k in range(len(kernels[num_kernels])):
                        for l in range(len(kernels[num_kernels][k])):
                            for m in range(len(res[i][j])):
                                res[i][j][num_kernels] += imgChannels[i + k][j + l][m] * kernels[num_kernels][k][l][m]
                    if (res[i][j][num_kernels] > 255):
                        res[i][j][num_kernels] = 255
                    if (res[i][j][num_kernels] < 0):
                        res[i][j][num_kernels] = 0

        return res

def maxPool(img, size, same_padding=False):
    res=[]
    for i in range(int(len(img)/size)):
        temp = []
        for j in range(int(len(img[0])/size )):
            temp1 = []
            for k in range(len(img[0][0])):
                temp1 += [0]
            temp += [temp1]
        res += [temp]
    for i in range(len(res)):
        for j in range(len(res[i])):
            for m in range(len(res[i][j])):
                maximum = -1
                for k in range(size):
                    for l in range(size):
                        maximum = max(maximum,img[i*size + k][j*size + l][m])
                res[i][j][m]=maximum
    return res


def softmax(layers):
    sumE=0
    temp=[]
    for i in range(len(layers)):
        sumE=math.exp(layers[i])
    temp+=[math.exp(layers[i])/sumE]
    return temp


def ReLU(layers):
    temp=[]
    for i in range(len(layers)):
        temp+=[max(0,layers[i])]
    return temp

def InceptionLayer(img, kernels, num_unit, reduce_dim=False, dim_red_kernels=[]):
    res=[]
    temp=[]
    if(reduce_dim):
        if(dim_red_kernels==[]):
            print("kernel untuk reduksi dimensi kosong")
        else:
            for num_kernels in range(len(kernels)):
                temp += [convolve(img, dim_red_kernels)]
        for num_kernels in range(len(kernels)):
            for i in range(len(num_unit[num_kernels])):
                res += [convolve(temp[num_kernels], kernels[num_kernels][i])]
    else:
        for num_kernels in range(len(kernels)):
            for i in range(len(num_unit[num_kernels])):
                res += [convolve(img, kernels[num_kernels][i])]

    return res





img=[[[65,12,23],[45,23,98],[76,199,68]],[[69,100,200],[56,98,123],[98,194,39]],[[94,192,200],[45,98,129],[156,17,143]]]
kernels=[[[[1,1,1],[0,0,0],[-1,-1,-1]],[[1,1,1],[0,0,0],[-1,-1,-1]],[[1,1,1],[0,0,0],[-1,-1,-1]]]]
imgConvol1=convolve(img, kernels, same_padding=False)
print(imgConvol1, len(imgConvol1), len(imgConvol1[0]), len(imgConvol1[0][0]))
kernel0=[]
SobelFilterVertical=[[[1,2,1],[0,0,0],[-1,-2,-1]],[[1,2,1],[0,0,0],[-1,-2,-1]],[[1,2,1],[0,0,0],[-1,-2,-1]]]

print(maxPool(img,2))