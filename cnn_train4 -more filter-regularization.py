import random
import os
import sys
import math
import numpy as np
import copy



trainPath = "D://python2.7.6//MachineLearning//CNN//trainingDigits"
dataName="D://python2.7.6//MachineLearning//CNN//x16x16.txt"
testPath = "D://python2.7.6//MachineLearning//CNN//testDigits"
 
     

global classDic;classDic={}
 
global epoch;epoch=3
global alpha;alpha=0.2
global pooldim,pooldd,outdim,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
pooldim=2;pooldd=4
outdim=4;outdd=16
convdim=8;convdd=64
numflt=40
fltdim=9;fltdd=81
xdim=16;xdd=256
nhh=outdd*numflt
numc=10
lbd=0#regularization para penalty for para-model complexity structure risk from overfitting

######################

def loadData():
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    global dataMat,yMat,labelList
    dataList=[]
    labelList=[]
    ###################labellist
    for filename in os.listdir(trainPath):
        pos=filename.find('_')
        clas=int(filename[:pos])
        if clas not in classDic:classDic[clas]=1.0
        else:classDic[clas]+=1.0
        labelList.append(clas)
    ##########
    obs=[]
    content=open(dataName,'r')
    line=content.readline().strip('\n').strip(' ')
    line=line.split(' ')
    #print line,len(line)
    while len(line)>1:
        obs=[float(n) for n in line if len(n)>1]
        #print 'o',obs,len(obs)
        
        line=content.readline().strip('\n').strip(' ');line=line.split(' ')
         
        dataList.append(obs);#print 'datalist',len(dataList)
    ##########
    print '%d obs loaded'%len(dataList),len(labelList),'labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
     
    #####
    dataMat=np.mat(dataList)
     
    #####
    num,dim=np.shape(dataMat)
    yMat=np.zeros((num,numc))
    for n in range(num):
        truey=labelList[n]
        yMat[n,truey]=1.0


def initialH():
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    global dataMat,yMat
    global hMat,hhMat,outputMat
    num,dim=np.shape(dataMat)
    hMat=np.mat(np.zeros((numflt,convdd)))#nfilter,nh
    outputMat=np.mat(np.zeros((num,numc))) #nclass
    hhMat=np.mat(np.zeros((numflt,outdd)))#nhh
    
    
     

def initialPara():
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    global Cmat,Wmat,bmat,bbmat #initial from random eps
    num,dim=np.shape(dataMat)
     
    Cmat=np.mat(np.zeros((numc,nhh)))
    Wmat=np.mat(np.zeros((numflt,fltdd)))
    bmat=np.mat(np.zeros((1,numflt)))
    bbmat=np.mat(np.zeros((1,numc)))
    for i in range(numc):
        for j in range(nhh):
            Cmat[i,j]=random.uniform(0,0.1)
    for i in range(numflt):
        for j in range(fltdd):
            Wmat[i,j]=random.uniform(0,0.1)
    
    #######
    for j in range(numflt):
        bmat[0,j]=random.uniform(0,0.1)
     
    for j in range(numc):
        bbmat[0,j]=random.uniform(0,0.1)
     
    

            
def initialErr():#transfer err sensitive
    global errW,errC,up1,up2
    global dataMat
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    n,d=np.shape(dataMat)
    errW=np.mat(np.zeros((1,convdd)))
    errC=np.mat(np.zeros((1,numc)))
    up1=np.mat(np.zeros((outdim,outdim)))
    up2=np.mat(np.zeros((convdim,convdim)))
     
    
def initialGrad():
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    global gradc,gradw,gradb,gradbb
    gradc=np.mat(np.zeros((numc,nhh)))
    gradw=np.mat(np.zeros((numflt,fltdd)))
    gradb=np.mat(np.zeros((1,numflt)))
    gradbb=np.mat(np.zeros((1,numc)))

def forward(x): #xi index not xvector
    global hMat,hhMat,outputMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    xvec=dataMat[x,:]
    ######1x256->16x16->64x81   9x9filter
    x16=vec2mat(xvec,xdim,xdim)
    #print x16
    #### ->64x81    (16-9+1)x(16-9+1) 8x8
    x64=np.mat(np.zeros((convdd,fltdd)))##64 pieces of dim81 patch
    i=0
    for hang in range(convdim):
        for lie in range(convdim):
            patch=x16[hang:hang+fltdim,lie:lie+fltdim]  #[0:9]==0,1,2,,,8 no 9
            #print patch
            pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
            x64[i,:]=pVec
            i+=1

    #####conv
    for patch in range(convdd):
        for kernel in range(numflt):
            con=Wmat[kernel,:]*x64[patch,:].T#1x81  x  81x1
            con=con[0,0]+bmat[0,kernel]
            con=1.0/(1.0+math.exp((-1.0)*con))
            hMat[kernel,patch]=con
    #####pool
    for k in range(numflt): #each kernel
        ####1x64->8x8 featmap  , 4x4 poolmap
        feaMap=vec2mat(hMat[k,:],convdim,convdim) 
        ####pool with 2x2 window mean pooling
        poolMap=np.mat(np.zeros((outdim,outdim)))
        for hang in range(outdim):
            for lie in range(outdim):
                patch=feaMap[hang*pooldim:hang*pooldim+pooldim,lie*pooldim:lie*pooldim+pooldim]
                v=patch.flatten().mean()
                poolMap[hang,lie]=v
        #####4x4->1x16 poolmap
        hhMat[k,:]=poolMap.flatten()
    #######full connect
    hhvec=hhMat.flatten()#5x16 -> 1x80
    fvec=hhvec*Cmat.T+bbmat#1x80  x  80x10==1x10
    outputMat[x,:]=softmax(fvec)
    ######
     
    return x16
                
def calcGrad(x,x16):#x index not vec
    global pooldim,pooldd,outd,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    global hMat,hhMat,outputMat,yMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    global gradc,gradw,gradb,gradbb
    global errW,errC,up1,up2
    ####err c floor
    fy=outputMat[x,:]-yMat[x,:] #matric 1x10
    sgm=outputMat[x,:].A*(1.0-outputMat[x,:].A)
    errC=np.mat(fy.A*sgm)#matric 1x10
    ######grad c
    hhflat=hhMat.flatten()#1x80 matric
    gradc=errC.T*hhflat+lbd*Cmat  #10x1  x  1x80==10x80  #lbd*Cmat->regularization
    gradbb=copy.copy(errC)#1x10 ##cannot change  at the same time
    #####5 kernel
    for k in range(numflt):
        ####calc up1
        vec=errC*Cmat[:,k*outdd:k*outdd+outdd] #1x10  x  10x16==1x16
        up1=vec2mat(vec,outdim,outdim) #1x16->4x4
        ######calc up2 :upsample: expand and divide 2x2 pooling windon
        for hang in range(outdim):
            for lie in range(outdim):
                m=up1[hang,lie]/float(pooldd)#not pool dimdim, but pool window dimdim
                mat2x2=np.mat(np.zeros((pooldim,pooldim)))+m#2x2 window filed with mean/4
                up2[pooldim*hang:pooldim*hang+pooldim,pooldim*lie:pooldim*lie+pooldim]=mat2x2
        #####8x8->1x64
        vecUp2=up2.flatten()#matric 1x64
        ####err for w floor
        sgm=hMat[k,:].A*(1.0-hMat[k,:].A)#1x64 array
        errW=np.mat(sgm*vecUp2.A)#1x64 matric
        ######calc w grad :conv2(x,errw,valid) 16x16 conv with 8x8==9x9
        ####x 16x16->(8x8)x81  conv with patch/filter 8x8
        x81=np.mat(np.zeros((fltdd,convdd)))##81 pieces of dim64 patch8x8
        i=0
        for hang in range(fltdim):
            for lie in range(fltdim):
                patch=x16[hang:hang+convdim,lie:lie+convdim]  #[0:9]==0,1,2,,,8 no 9
                #print patch
                pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
                x81[i,:]=pVec; 
                i+=1
         
        ###conv x with filter 8x8
        gradw[k,:]=errW*x81.T#1x64  x  64x81==1x81
        ########
        gradb[0,k]=errW.sum(1)[0,0]
    ########add regularization
    gradw=gradw+lbd*Wmat #5x81
    ##############gradient normalize
    for k in range(numc):
        gradc[k,:]=normalize(gradc[k,:],'vector')
    for k in range(numflt):
        gradw[k,:]=normalize(gradw[k,:],'vector')
    
def updatePara():
    global Cmat,Wmat,bmat,bbmat
    global gradc,gradw,gradb,gradbb
    Cmat=Cmat+alpha*(-1.0)*gradc
    Wmat=Wmat+alpha*(-1.0)*gradw
    bmat=bmat+alpha*(-1.0)*gradb
    bbmat=bbmat+(-1.0)*alpha*gradbb
    
def calcLoss():
    global Cmat,Wmat,bmat,bbmat
    global outputMat,yMat,dataMat # fk is calculated with old para
    num,dim=np.shape(dataMat)
    loss=0.0
    for n in range(num)[:100]:
        diff=outputMat[n,:]-yMat[n,:]#1x10 mat
        ss=diff*diff.T;ss=ss[0,0]
        loss+=ss
    #print 'least square loss',loss
    ####add regularization
    modew=Wmat.A*Wmat.A;modew=modew.sum(1).sum(0)*lbd   #sum(0).sum(1) not work   ---- .sum(1).sum(0) work
    modec=Cmat.A*Cmat.A;modec=modec.sum(1).sum(0)*lbd
    loss=loss+modew+modec
    return loss
        
        
    
    
    
    
        
        
        
        
    
    
        
    
    

##########################################
def vec2mat(vec,nhang,nlie): #input vec 1x16 ouput matric 4x4
    if nhang!=nlie:print 'num hang must = num lie'
    n=nhang#for example :1x16->4x4
    Mat=np.mat(np.zeros((n,n)))
    for hang in range(n):
        for lie in range(n):
            pos=n*hang+lie
            Mat[hang,lie]=vec[0,pos]
    return Mat
    
    
def shuffleObs():
    global dataMat
    num,dim=np.shape(dataMat) #1394 piece of obs
    order=range(num)[:]  #0-100  for loss calc,101...for train obs by obs ///not work. must use whole set to train
    random.shuffle(order)
    return order
    
    

def softmax(outputMat): #1x10 vec
    vec=np.exp(outputMat)  #1x10  #wh+b
    ss=vec.sum(1);ss=ss[0,0]
    outputMat=vec/(ss+0.000001)
    return outputMat
    
def normalize(vec,opt):
    if opt=='prob': #in order to sum prob=1
        ss=vec.sum(1)[0,0]
        vec=vec/(ss+0.000001)
    if opt=='vector': #in order to mode or length ||vec||=1
        mode=vec*vec.T
        mode=math.sqrt(mode[0,0])
        vec=vec/(mode+0.000001)
    if opt not in ['vector','prob']:
        print 'only vector or prob'
    return vec 

        

                

        
        
            




###################main
loadData()
initialH()
initialPara()
initialErr()
initialGrad()

#####
for ep in range(epoch):
    obsList=shuffleObs()
    alpha/=2.0
    for obs in obsList[:]: #obs=x index not vec
        #obs=random.sample(range(10),1)[0]
        x16=forward(obs)
        loss=calcLoss()#loss calc with old para
        calcGrad(obs,x16)
        updatePara()
    print  'epoch %d loss %f'%(ep,loss)


###output

#####output para w m n c ,b
global Cmat,Wmat,bmat,bbmat
outPath="D://python2.7.6//MachineLearning//CNN//para"
 
outfile1 = "C.txt"
outfile2 = "W.txt"

outfile3 = "B.txt"
outfile4 = "BB.txt"
 

outPutfile=open(outPath+'/'+outfile1,'w')
n,m=np.shape(Cmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Cmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
##
outPutfile=open(outPath+'/'+outfile2,'w')
n,m=np.shape(Wmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Wmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
###
outPutfile=open(outPath+'/'+outfile3,'w')
n,m=np.shape(bmat)

for j in range(m):
    outPutfile.write(str(bmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
## 
outPutfile=open(outPath+'/'+outfile4,'w')
n,m=np.shape(bbmat)

for j in range(m):
    outPutfile.write(str(bbmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()


 
 

 
    
         
    
    
    







    
    
