def draw_pic(x_axis,x_axis_test,y_train,y_test,data_train,data_test,label,figure):
    sub_pic1 = figure.add_subplot(211)
    sub_pic2 = figure.add_subplot(212)

    sub_pic1.scatter(x_axis, data_train, c='k', label='data')
    sub_pic2.scatter(x_axis_test, data_test, c='k', label='data')

    sub_pic1.plot(x_axis,y_train,c='b',label=label+" train model")
    sub_pic2.plot(x_axis_test,y_test,c='r',label=label+" test model")

    sub_pic1.set_xlim(-10, len(x_axis)+5)
    sub_pic2.set_xlim(-10, len(x_axis_test)+5)
    sub_pic1.set_ylim(-2,15)
    sub_pic2.set_ylim(-2,15)

    sub_pic1.set_ylabel("target")
    sub_pic2.set_ylabel("target")
    sub_pic2.set_xlabel("data")

    sub_pic1.legend()
    sub_pic2.legend()