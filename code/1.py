import numpy
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas

# taking data from files
with open('../data/test.pkl', 'rb') as file1:
    data = pickle.load(file1)
with open('../data/train.pkl', 'rb') as file2:
    data2 = pickle.load(file2)

# shuffling data
size = data2.shape[0]
numpy.random.shuffle(data2)
size = data.shape[0]
numpy.random.shuffle(data)

# splitting data into coordinates
x = data2[:,:-1]
y = data2[:,1]
test_x = data[:,:-1]
test_y = data[:,1]

# splitting training set into 10 parts
train_x = numpy.array((numpy.array_split(x, 10)))
train_y = numpy.array((numpy.array_split(y, 10)))

# initialising tables
v_table = numpy.zeros((10,10))
b_table = numpy.zeros((10,10))

# initialising arrays
bias = numpy.zeros((20))
bias2_avg = numpy.zeros((20))
var_avg = numpy.zeros((20))
err_avg = numpy.zeros((20))
tot_avg = numpy.zeros((20))
irr_avg = numpy.zeros((20))
xarr = numpy.zeros((20))

# calc for each degree
for degree in range (1,21):
    res1 = numpy.zeros((10,80))
    res2 = numpy.zeros((10,80))
    #for training set
    for i in range (10):
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        #Transform the polynomial features1 as required
        X = poly.fit_transform(train_x[i])
        X_TEST = poly.fit_transform(test_x)
        reg = LinearRegression()

        #Train the model for the chosen training set
        reg.fit(X, train_y[i])
        y_predict = reg.predict(X_TEST)

        res1[i] = y_predict
        res2[i] = (test_y - y_predict)**2

    #calculate bias
    point_avg=numpy.mean(res1,axis=0)
    bias2_avg[degree-1]=numpy.mean((point_avg-test_y)**2)
    bias[degree-1]=numpy.abs(numpy.mean((point_avg-test_y)))
    new_avg=numpy.mean(res2, axis=0)
    #calculate variance
    point_var = numpy.var(res1,axis=0)
    var_avg[degree-1]=numpy.mean(point_var)
    #irreducible error calc
    err_avg[degree-1]=numpy.mean(new_avg) - (bias2_avg[degree-1] + var_avg[degree-1])
    # total error calc
    tot_avg[degree-1]=numpy.mean(new_avg)

fin_avg = numpy.mean(err_avg)

use = numpy.zeros((20))
for degree in range (1,21):
    use[degree-1] = fin_avg
    xarr[degree-1] = degree
# table 1
bias_table=pandas.DataFrame({'Degree':numpy.array(range(1,21)),'Bias':bias,'Variance': var_avg})
print(bias_table.to_string(index=False))
print('')
# table 2
error_table=pandas.DataFrame({'Degree':numpy.array(range(1,21)), 'Irreducible Error':err_avg[:]})
print(error_table.to_string(index=False))
plt.plot(xarr[:], bias2_avg[:], label='Bias^2', color = 'blue')
plt.plot(xarr[:], var_avg[:],label='Variance', color = 'red')
plt.plot(xarr[:], tot_avg[:],label='Total Error', color = 'black')
plt.plot(xarr[:], err_avg[:],label='Irreducible Error', color = 'lightgreen')
plt.xlabel('Model Complexity', fontsize='medium')
plt.ylabel('Error', fontsize='medium')
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
