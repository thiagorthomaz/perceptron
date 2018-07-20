from random import choice 
from numpy import array, dot, random


#Step function
unit_step = lambda x: 0 if x < 0 else 1

#Map possible input to the expected output.
training_data = [ 
    (array([0,0,1]), 0), 
    (array([0,1,1]), 1), 
    (array([1,0,1]), 1), 
    (array([1,1,1]), 1),
]

#Generate three random numbers between 0 and 1 as the initial weights
w = random.rand(3);

#erros store the erro values
#eta controls the learning rate
#n number of learning interations
errors = [] 
eta = 0.2 
n = 100


for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x

for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))


