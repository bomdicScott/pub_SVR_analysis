from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
import unicodecsv as csv
from plot_picture import *
from SVR_support_function import *

def Lin_clean_data(data,target,least):
    i=0
    while(len(data) > i):
        target[i] = float(target[i])
        for j in range(len(data[0])):
            data[i][j] = float(data[i][j])
            if(data[i][j] == -1 or data[i][j] == 0 or target[i] < least):
                del data[i]
                del target[i]
                break
            if(j == len(data[i])-1):
                i += 1
    return data,target

with open("C:/Users/sean/Desktop/SVR_DATA/edwademd.csv","rb") as data_file:
    data,target = [],[]
    for row in csv.reader(data_file):
        data += [[row[0],row[4],row[6],row[10]]]
        target += [row[9]]

data,target = Lin_clean_data(data[1:],target[1:],2)

point = 6000
X_train = data[:point-1]
X_test = data[point:point+int(point*0.2)]
y_train = target[:point-1]
y_test = target[point:point+int(point*0.2)]


svr = LinearSVR(C=0.1)
svr_model = svr.fit(X_train,y_train)
lin = svr.predict(X_train)
lin_test = svr.predict(X_test)

lin,lin_test = data_normalize(y_train,y_test,lin,lin_test)

print("Train score : ",score(y_train,lin))
print("Train average error : ",sum(abs(y_train-lin)) / float(len(y_train)))

print("Fit score : ",score(y_test,lin_test))
print("Fit average error : ",sum(abs(y_test-lin_test)) / float(len(y_test)))

figure1 = plt.figure(1,figsize=[20,10])
draw_pic(range(len(X_train)),range(len(X_test)),lin,lin_test,y_train,y_test,label='lin',figure=figure1)
figure1.savefig("C:/Users/sean/Desktop/SVR_DATA/linSVR.png",dpi=300,format="png")
plt.close(1)

#look at the results
#plt.plot(range(len(y_train)), y_train, c='k', label='data')
#plt.hold('on')
#plt.plot(range(len(lin)),lin, c='g', label='RBF model')
#plt.plot(range(len(lin_test)), lin_test, c='r', label='Linear model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.xlim(0,len(lin))
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()


