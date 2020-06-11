
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

w1=rm.random()
w2=rm.random()
b=rm.random()
η=0.0001
#rm.seed(1230)
epoch=0    

#主要計算   
print("This is Widrow-Hoff Learning.")
while True:
    epoch=epoch+1
    error_sum=0
    error=0
    for i in range(len(p1)):
        
        a=p1[i]*w1+p2[i]*w2+b
        
        error=t[i]-a
        #每一代都做距離更新
        w1=w1+η*error*p1[i]
        w2=w2+η*error*p2[i]
        b=b+η*error
        #mean-square error
        error_sum=error_sum+error*error
        mse=error_sum/len(p1)
        
    
    if mse<0.5:
        print('convergence on %d epoch.'%epoch)
        print("MSE:",mse)
        break
    elif epoch>1000:
        print("It is not convergence.")
        break
    else:
        continue;
    



    
    
    
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
ax1.set_title('Widrow-Hoff Learning') 

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