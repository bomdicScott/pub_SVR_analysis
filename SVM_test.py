import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import unicodecsv as csv

def draw_pic(x_axis,x_axis_test,y_train,y_test,data_train,data_test,label,figure):
    sub_pic1 = figure.add_subplot(211)
    sub_pic2 = figure.add_subplot(212)

    sub_pic1.scatter(x_axis, data_train, c='k', label='data')
    sub_pic2.scatter(x_axis_test, data_test, c='k', label='data')

    sub_pic1.plot(x_axis,y_train,c='b',label=label+" train model")
    sub_pic2.plot(x_axis_test,y_test,c='r',label=label+" test model")

    sub_pic1.set_xlim(-10, len(x_axis))
    sub_pic2.set_xlim(-10, len(x_axis_test))

    sub_pic1.set_ylabel("target")
    sub_pic2.set_ylabel("target")
    sub_pic2.set_xlabel("data")

    sub_pic1.legend()
    sub_pic2.legend()

def build_svr(x,y,x_test,kernel='rbf',c=0.1,gamma=0.1,degree=2):
    svr = SVR(kernel=kernel, C=c, gamma=gamma)
    svr_model = svr.fit(x,y)
    result = svr_model.predict(x)
    result_test = svr_model.predict(x_test)

    return result,result_test
###############################################################################
# Generate sample data
with open("C:/Users/sean/Desktop/SVR_DATA/DATA1.csv","rb") as data_file:
    data,target = [],[]
    for row in csv.reader(data_file):
        data += [[row[4],row[6]]]
        target += [row[5]]

def clear_data(count,data,target):
    if(float(target[count]) < 2 ):
        del target[count]
        del data[count]
        clear_data(count,data,target)
    else:
        return
#cv = ShuffleSplit(100, n_iter=10, test_size=0.1, random_state=0)
print(len(data))
data = data[3:]
target = target[3:]
for count in range(len(target)):
    if(count < len(target)):
        clear_data(count,data,target)
print(len(data))

point = 9000
X_train = data[:point]
X_test = data[point+1:]
y_train = target[:point]
y_test = target[point+1:]

# Fit regression model
rbf,rbf_test = build_svr(X_train,y_train,X_test,kernel='rbf',c=0.1,gamma=0.1)
lin,lin_test = build_svr(X_train,y_train,X_test,kernel='linear',c=0.1,gamma=0.1)
#lin2,lin_test2 = build_svr(X_train,y_train,X_test,kernel='linear',c=0.1,gamma=0.1)
#svr_sig = SVR(kernel='sigmoid',C=1e5,gamma=0.1)
#svr_poly = SVR(kernel='poly', C=10,degree=2)

###############################################################################
# look at the results
figure1 = plt.figure(1,figsize=[20,10])
figure2 = plt.figure(2,figsize=[20,10])
#figure3 = plt.figure(3,figsize=[20,10])

x_axis = []
for i in range(len(X_train)):
    x_axis += [str(i)]

x_axis_test = []
for i in range(len(X_test)):
    x_axis_test += [str(i)]

draw_pic(x_axis,x_axis_test,rbf,rbf_test,y_train,y_test,label='rbf',figure=figure1)
draw_pic(x_axis,x_axis_test,lin,lin_test,y_train,y_test,label='linear',figure=figure2)
#draw_pic(x_axis,x_axis_test,lin2,lin_test2,y_train,y_test,label='linear',figure=figure3)

figure1.savefig("C:/Users/sean/Desktop/SVR_DATA/SVR_rbf.png",dpi=300,format="png")
figure2.savefig("C:/Users/sean/Desktop/SVR_DATA/SVR_lin.png",dpi=300,format="png")
#figure3.savefig("C:/Users/sean/Desktop/SVR_DATA/SVR_lin2.png",dpi=300,format="png")

#figure.show()
plt.close(1)
plt.close(2)

