import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import pickle

# taking data from files
with open('../data/test.pkl', 'rb') as file1:
    data = pickle.load(file1)
with open('../data/train.pkl', 'rb') as file2:
    data2 = pickle.load(file2)

# shuffling data
np.random.shuffle(data2)
np.random.shuffle(data)

# splitting data into coordinates
x = data2[:,:-1]
y = data2[:,1]
test_x = data[:,:-1]
test_y = data[:,1]

# initialising tables
v_table = np.zeros((10,10))
b_table = np.zeros((10,10))

# initialising arrays
bias = np.zeros((20))
bias2_avg = np.zeros((20))
var_avg = np.zeros((20))
err_avg = np.zeros((20))
tot_avg = np.zeros((20))
irr_avg = np.zeros((20))
xarr = np.zeros((20))


# splitting training set into 10 parts
train_x = np.array((np.array_split(x, 10)))
train_y = np.array((np.array_split(y, 10)))


# calc for each degree
for degree in range (1,21):

    res1 = np.zeros((10,80))
    res2 = np.zeros((10,80))
    #for training set
    for i in range (0,10):
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        #Transform the polynomial features as required ans training model
        X = poly.fit_transform(train_x[i])
        reg = LinearRegression()

        reg.fit(X, train_y[i])
        X_TEST = poly.fit_transform(test_x)
        y_predict = reg.predict(X_TEST)

        res1[i] = y_predict
        res2[i] = (test_y - y_predict)**2

    #calculate bias
    point_avg=np.mean(res1,axis=0)
    bias2_avg[degree-1]=np.mean((point_avg-test_y)**2)
    bias[degree-1]=np.mean(np.abs((point_avg-test_y)))
    new_avg=np.mean(res2, axis=0)
    #calculate variance
    point_var = np.var(res1,axis=0)
    var_avg[degree-1]=np.mean(point_var)
    #irreducible error calc
    err_avg[degree-1]=np.mean(new_avg) - (bias2_avg[degree-1] + var_avg[degree-1])
    # total error calc
    tot_avg[degree-1]=np.mean(new_avg)

fin_avg = np.mean(err_avg)

use = np.zeros((20))
for degree in range (1,21):
    use[degree-1] = fin_avg
    xarr[degree-1] = degree
# table 1
bias_table=pandas.DataFrame({'Degree':np.array(range(1,21)),'Bias':bias,'Variance': var_avg})
print(bias_table.to_string(index=False))
print('')
# table 2
error_table=pandas.DataFrame({'Degree':np.array(range(1,21)), 'Irreducible Error':err_avg[:]})
print(error_table.to_string(index=False))
plt.plot(xarr[:], bias2_avg[:], label='Bias^2', color = 'blue')
plt.plot(xarr[:], var_avg[:],label='Variance', color = 'red')
plt.plot(xarr[:], tot_avg[:],label='Total Error', color = 'black')
plt.plot(xarr[:], err_avg[:],label='Irreducible Error', color = 'lightgreen')
plt.ylabel('Error', fontsize='medium')
plt.xlabel('Model Complexity', fontsize='medium')
plt.title("Bias vs Variance")
plt.legend()
plt.show()
plt.xlabel('Model Complexity', fontsize='medium')
plt.ylabel('Irreducible Error', fontsize='medium')
plt.title("Irreducible Error")
plt.plot(err_avg[:],label='Irreducible Error', color = 'black')
plt.plot(use[:],label='mean', color='lightgreen')
plt.legend()
plt.show()
