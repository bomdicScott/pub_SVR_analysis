from sklearn.svm import SVR
import matplotlib.pyplot as plt
import unicodecsv as csv
from plot_picture import *
from SVR_support_function import *

def build_svr(x,y,x_test,kernel='rbf',c=0.1,gamma=0.1,degree=2):
    svr = SVR(kernel=kernel, C=c, gamma=gamma,degree=degree)
    svr_model = svr.fit(x,y)
    result = svr_model.predict(x)
    result_test = svr_model.predict(x_test)

    coef = svr.coef_

    print(score(y,result))
    print(score(y_test,result_test))

    return result,result_test,coef

def clean_data(count,data,target):
    if(float(target[count]) < 2 ):
        del target[count]
        del data[count]
        clean_data(count,data,target)
    else:
        return

def compute_predict(X_train,X_test,coef):
    com_lin,com_lin_test = [],[]
    for count in range(len(X_train)):
        com_lin += [float(X_train[count][0])*coef[0][0]+float(X_train[count][1])*coef[0][1]+float(X_train[count][2])*coef[0][2]]
    for count in range(len(X_test)):
        com_lin_test += [float(X_test[count][0])*coef[0][0]+float(X_test[count][1])*coef[0][1]+float(X_test[count][2])*coef[0][2]]

    return com_lin,com_lin_test


###############################################################################
# Generate sample data
with open("C:/Users/sean/Desktop/SVR_DATA/DATA1.csv","rb") as data_file:
    data,target = [],[]
    for row in csv.reader(data_file):
        data += [[row[4],row[6],row[10]]]
        target += [row[5]]

data = data[3:]
target = target[3:]
for count in range(len(target)):
    if(count < len(target)):
        clean_data(count,data,target)

point = 100
X_train = data[:point-1]
X_test = data[point:point+int(point*0.1)]
y_train = target[:point-1]
y_test = target[point:point+int(point*0.1)]

#data = [[1.0,0.5],[0.5,1.0],[1.5,0.0],[0.0,1.5],[1.0,0.5],[1.0,0.5],[1.5,0.5]]
#target = [1.0,1.2,0.8,1.2,1.0,1.0,1.5]
#X_train = data[:4]
#X_test = data[5:]
#y_train = target[:4]
##y_test = target[5:]

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

# Fit regression model
#rbf,rbf_test = build_svr(X_train,y_train,X_test,kernel='rbf',c=0.1)
lin,lin_test,coef = build_svr(X_train,y_train,X_test,kernel='linear',c=0.1)

com_lin,com_lin_test = compute_predict(X_train,X_test,coef)
#print(X_test[0],lin_test[0],"test")
#print(X_test[1],lin_test[1],"test")
#for count in range(len(X_train)):
#    print(X_train[count],lin[count],y_train[count])

###############################################################################
# look at the results
figure1 = plt.figure(1,figsize=[20,10])
figure2 = plt.figure(2,figsize=[20,10])
#figure3 = plt.figure(3,figsize=[20,10])

x_axis = range(len(X_train))
x_axis_test = range(len(X_test))

draw_pic(x_axis,x_axis_test,com_lin,com_lin_test,y_train,y_test,label='com_lin',figure=figure1)
draw_pic(x_axis,x_axis_test,lin,lin_test,y_train,y_test,label='linear',figure=figure2)
#draw_pic(x_axis,x_axis_test,lin2,lin_test2,y_train,y_test,label='linear',figure=figure3)

figure1.savefig("C:/Users/sean/Desktop/SVR_DATA/SVR_com_lin.png",dpi=300,format="png")
figure2.savefig("C:/Users/sean/Desktop/SVR_DATA/SVR_lin.png",dpi=300,format="png")
#figure3.savefig("C:/Users/sean/Desktop/SVR_DATA/SVR_lin2.png",dpi=300,format="png")

#figure.show()
#plt.close(1)
plt.close(2)

