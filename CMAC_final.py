# Libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# class like structure for association_matrix.
class association_matrix:
    def __init__(self):
        self.index = 0
        self.weight = []

# functions for generation association matrix
def assoc_values(ind, beta):
    weights = []
    b = (beta//2)
    for i in range(ind - b, ind + b + 1):
        middle=[i,1]
        weights.append(middle)
    return weights

def assoc_values_cont(ind, beta):
    weights = []
    index = []
    weightage = []
    b = beta/2
    #top locs weights
    index = math.floor(ind - b)
    weightage = math.ceil(ind - b) - (ind - b)
    
    top_cell = []
    top_cell.append(index)
    top_cell.append(weightage)
    
    if weightage != 0:
        weights.append(top_cell)
    
    #middle loc weights

    for index in range(math.ceil(ind - b), math.floor(ind + b + 1)):
        mid_cell = []
        mid_cell.append(index)
        mid_cell.append(1)
        weights.append(mid_cell)

    #bottom loc weights

    index = math.floor(ind+beta/2)
    weightage = (ind + beta/2) - math.floor(ind+beta/2)

    bottom_cell = []
    bottom_cell.append(index)
    bottom_cell.append(weightage)

    if weightage!=0:
        weights.append(bottom_cell)
    return weights

def assoc_index(i,beta,assoc_num,sample):
    i = int(i)
    a_ind = beta//2 + ((assoc_num - 2*(beta//2))*i)/sample
    return math.floor(a_ind)

def meanSqEr(weights, synapse_weight,X,Y):
    meansq =0
    for i in range(0,len(synapse_weight)):
        sum_syn = 0
        for j in synapse_weight[i]:
            sum_syn = sum_syn + weights[j[0]]*j[1]
        meansq += (sum_syn - Y[i])**2
    return meansq

def test(weights,synapse_weight):
    output = []
    for i in range(0,len(synapse_weight)):
        sum_syn = 0
        for j in synapse_weight[i]:
            sum_syn += weights[j[0]]*j[1]
        output.append(sum_syn)
    return output

def error_calc(output,y_test):
    error=0
    for i in range(len(output)):
        error=error+abs(output[i]-y_test[i])/abs(output[i]+y_test[i])
    # print(error)
    if float(error)>100:
        error=100
    return error

# Initialization of parameters
Fs = 100 # Range of x
f = 1 # frequency of sin
sample = 100 # number of samples
beta = 5 # overlap
assoc_num = 35 # number of weights.

x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)

### Plot the sin function.
plt.plot(x, y, 'r')
plt.title('1-D Sin function')
plt.show()

# getting 100 data points between 0 to 100
func = np.stack((x.T, y.T), axis=0)
data = func.T
np.random.shuffle(data)

# getting training data and testing data.
train_data = data[:70]
test_data = data[70:]

                ########## Training Section ##########

X = train_data[:, 0]
Y = train_data[:, 1]
X_test = test_data[:,0]
Y_test = test_data[:,1]

# plotting Training data
plt.figure(1)
plt.plot(X,Y,'r+',label = 'Training Data')
plt.plot(X_test,Y_test,'b+',label = 'Testing Data')
plt.title('Separated Training and Testing Data')
plt.legend()
plt.show()

################################ Continous ####################################
test_error_mat=[]
time_matrix=[]
for beta in range(1,34):
    print("Continous: Overlap=",beta)
    start_time = time.time()
    #initializing weightrainings to 1
    weights = np.ones((35,1))
    rate = 0.05 # learning rate

    synapse = association_matrix()

    for ind in X:
        synapse.index = assoc_index(ind, beta , assoc_num, sample)
        synapse.weight.append(assoc_values_cont(synapse.index, beta))

    # Initialization of error measuring parameters
    error_list = []
    error_plot = []
    prevError = 0
    currentError = 10
    iterations = 0

    # Running for 1000 iterations.
    
    while iterations < 500 and abs(prevError - currentError) > 0.00001:
        prevError = currentError
        #print(abs(prevError- currentError))
        for i in range(0,len(synapse.weight)):
            sum_syn = 0
            for j in synapse.weight[i]:
                sum_syn += weights[j[0]]*j[1]
            error = sum_syn - Y[i]
            correction  = error/beta
            for j in synapse.weight[i]:
                weights[j[0]] -= rate*correction*j[1]
        currentError = float(meanSqEr(weights,synapse.weight,X,Y))
        error_list.append(currentError)
        iterations += 1
        error_plot.append(iterations)
    if beta==5:
    #figure for error convergence
        plt.figure(2)
        plt.plot(np.asarray(error_plot), np.asarray(error_list), 'b--',label = 'error convergence')
        plt.legend()
        plt.title('Error Convergence g=5 : Continous')
        plt.show()
    time_matrix.append(time.time()-start_time)
    # print(iterations,'iterations')
    # print(abs(prevError - currentError),'error') 

                    ########## Testing the model - Continous##########


    synapse_test = association_matrix()

    for ix in X_test:
        synapse_test.index = (assoc_index(ix, beta , assoc_num, sample))
        synapse_test.weight.append(assoc_values_cont(synapse_test.index, beta))

    output = test(weights, synapse_test.weight)
    test_error=error_calc(output,Y_test)
    test_error_mat.append(abs(100-float(test_error)))

    if beta==5:
    # Figure with actual VS expected
        plt.figure(3)
        plt.plot(X,Y,'g+',label = 'training data')
        plt.plot(X_test,Y_test,'b+',label = 'test data')
        plt.plot(X_test,np.asarray(output),'ro', label = 'predicted outputs')
        plt.legend()
        plt.title('Test data vs Predicted Output : Continous')
        plt.show()
################################ Discrete ####################################
test_error_mat_disc=[]
time_matrix_disc=[]
for beta in range(1,34):
    print("Discrete: Overlap=",beta)
    start_time = time.time()
    #initializing weightrainings to 1
    weights = np.ones((35,1))
    rate = 0.05 # learning rate

    synapse = association_matrix()

    for ind in X:
        synapse.index = assoc_index(ind, beta , assoc_num, sample)
        synapse.weight.append(assoc_values(synapse.index, beta))

    # Initialization of error measuring parameters
    error_list = []
    error_plot = []
    prevError = 0
    currentError = 10
    iterations = 0

    # Running for 1000 iterations.
    
    while iterations < 500 and abs(prevError - currentError) > 0.00001:
        prevError = currentError
        #print(abs(prevError- currentError))
        for i in range(0,len(synapse.weight)):
            sum_syn = 0
            for j in synapse.weight[i]:
                sum_syn += weights[j[0]]*j[1]
            error = sum_syn - Y[i]
            correction  = error/beta
            for j in synapse.weight[i]:
                weights[j[0]] -= rate*correction*j[1]
        currentError = float(meanSqEr(weights,synapse.weight,X,Y))
        error_list.append(currentError)
        iterations += 1
        error_plot.append(iterations)
    if beta==5:
    #figure for error convergence
        plt.figure(2)
        plt.plot(np.asarray(error_plot), np.asarray(error_list), 'b--',label = 'error convergence')
        plt.legend()
        plt.title('Error Convergence g=5 : Discrete')
        plt.show()
    time_matrix_disc.append(time.time()-start_time)
    # print(iterations,'iterations')
    # print(abs(prevError - currentError),'error') 

                    ########## Testing the model -Discrete ##########


    synapse_test = association_matrix()

    for ix in X_test:
        synapse_test.index = (assoc_index(ix, beta , assoc_num, sample))
        synapse_test.weight.append(assoc_values(synapse_test.index, beta))

    output = test(weights, synapse_test.weight)
    test_error=error_calc(output,Y_test)
    test_error_mat_disc.append(abs(100-float(test_error)))
     
    if beta==5:
    # Figure with actual VS expected
        plt.figure(3)
        plt.plot(X,Y,'g+',label = 'training data')
        plt.plot(X_test,Y_test,'b+',label = 'test data')
        plt.plot(X_test,np.asarray(output),'ro', label = 'predicted outputs')
        plt.title('Test data vs Predicted Output : Discrete')
        plt.legend()
        plt.show()
    

print('Accuracy:Discrete',test_error_mat_disc)
print('Time of convergence:Discrete',time_matrix_disc)

print('Accuracy:Continous',test_error_mat)
print('Time of convergence:Continous',time_matrix)

# print(test_error_mat)
plt.figure(3)
plt.plot(test_error_mat,'g*-',label = 'Continous')
plt.plot(test_error_mat_disc,'r*-',label = 'Discrete')
plt.title('Accuracy Comparision of Continous vs Discrete')
plt.legend(loc='best')
plt.show()

plt.figure(4)
plt.plot(time_matrix,'b*-',label = 'Continous')
plt.plot(time_matrix_disc,'g*-',label = 'Discrete')
plt.title('Time of Convergence Comparision of Continous vs Discrete')
plt.legend(loc='best')
plt.show()

plt.figure(5)
# plt.plot(test_error_mat,'g*-',label = 'Continous')
plt.plot(test_error_mat_disc,'r*-',label = 'Discrete')
plt.title('Accuracy of Discrete CMAC vs Overlap')
plt.legend(loc='best')
plt.show()

plt.figure(6)
plt.plot(time_matrix,'b*-',label = 'Continous')
# plt.plot(time_matrix_disc,'g*-',label = 'Discrete')
plt.title('Accuracy of Continuous CMAC vs Overlap')
plt.legend(loc='best')
plt.show()