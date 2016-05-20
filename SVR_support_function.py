def score(y_true,y_pred,weight=1):
    y_true = [float(x) for x in y_true]
    numerator = sum(weight * (true-predict)**2 for true,predict in zip(y_true , y_pred))

    average = sum(y_true)/float(len(y_true))
    denominator = sum(weight * (x - average)**2 for x in y_true)

    return 1-numerator/denominator

def data_normalize(y_train,y_test,y_train_pred,y_test_pred):
    train_difference = sum(y_train_pred - y_train) / float(len(y_train))
    test_difference = sum(y_test_pred - y_test) / float(len(y_test))
    return y_train_pred-train_difference,y_test_pred-test_difference