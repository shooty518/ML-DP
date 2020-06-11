


import numpy as np
import matplotlib.pyplot as plt
import random as rm

#資料引入
while True:
    s = input("Please input you want to see dataset(number 1~4): ")
    if not s.isdigit():
        continue
    s = int(s)
    if s>=1 and s <=4:
        if s==1:
            data = np.loadtxt("dataset1.txt",delimiter=",")
        elif s==2:
            data = np.loadtxt("dataset2.txt",delimiter=",")
        elif s==3:
            data = np.loadtxt("dataset3.txt",delimiter=",")
        else:
            data = np.loadtxt("dataset4.txt",delimiter=",")
        break



p1=data[:,0]
p2=data[:,1]
t =data[:,2]

#給初始的線
w1=rm.random()
w2=rm.random()
b=rm.random()
η=5
#rm.seed(1230)
epoch=0

#主要計算
print("Perceptron Learning Rule.")
while True:
    epoch=epoch+1
    error_check=0
    for i in range(len(p1)):
        
        a=p1[i]*w1+p2[i]*w2+b
        #Hardlim
        if a>=0:
            a=1
        else:
            a=-1
        #如果不符合進行更新
        if a!=t[i]:
            w1=w1+η*(t[i]-a)*p1[i] 
            w2=w2+η*(t[i]-a)*p2[i]
            b=b+η*(t[i]-a)
            error_check=error_check+1
        else:
            continue;
    #如果這個世代都沒有錯誤跳出
    if error_check==0:
        print("convergence on %d epoch."%epoch)
        break
    #限定做幾個世代
    elif epoch>1000:
        print("It is not convergence.")
        #計算有幾個點錯誤
        error_check=0       
        for i in range(len(p1)):
            a=p1[i]*w1+p2[i]*w2+b
            if a>=0:
                a=1
            else:
                a=-1
            if a!=t[i]:
                error_check=error_check+1
            else:
                continue
        print("%d point is error."%error_check)
        break
    else:
        continue
    


#畫圖
positive_x=[]
positive_y=[]
negitive_x=[]
negitive_y=[]        
   

#分群
for i in range(len(p1)):
    if t[i]==1:
        positive_x.append(p1[i])
        positive_y.append(p2[i])
    elif t[i]==-1:
        negitive_x.append(p1[i])
        negitive_y.append(p2[i])
#畫線
x1=np.linspace(max(p1)+5,min(p1)-5)
x2=(-x1*w1-b)/w2
        
fig = plt.figure() 
ax1 = fig.add_subplot(111) 
ax1.set_title('Perceptron Learning Rule') 

plt.xlabel('X') 
plt.ylabel('Y') 
plt.plot(x1,x2)

#畫點
positive=ax1.scatter(positive_x,positive_y,s=100,c = 'blue',marker = 'o') 
negitive=ax1.scatter(negitive_x,negitive_y,s=100,c = 'black',marker = 'x')

plt.legend((positive,negitive),("target=1","target=-1")) 
plt.grid(True)


print("[w1,w2,b]=[%f,%f,%f]"%(w1,w2,b))


plt.show() 