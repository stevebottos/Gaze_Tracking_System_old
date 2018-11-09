# First the imports:
import numpy as np
from pomegranate import * # for HMM stuff
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)
pd.set_option('display.width', 1000)

# Using the pomegranate package we can build and train an HMM as follows:

# First, the emission probabilities for each state:
d1 = DiscreteDistribution({'b1' : 0.7, 'b2' : 0.3})
d2 = DiscreteDistribution({'b1' : 0, 'b2' : 1})

# Attaching the emission probabilities to their proper states
s1 = State(d1, name="s1")
s2 = State(d2, name="s2")

model = HiddenMarkovModel('Gaze Tracker')
model.add_states([s1, s2])

# Pi probabilities are added using "model.start" and the state that we're describing
model.add_transition(model.start, s1, 1)

# State transitions are defined this way
model.add_transition(s1, s1, 0.3)
model.add_transition(s1, s2, 0.7)
model.add_transition(s2, s1, 0.7)
model.add_transition(s2, s2, 0.3)
model.bake()

dat = np.genfromtxt('Coords35.csv', delimiter=",")
plt.plot(dat[:,0], dat[:,1], 'b.')
plt.title("Gaze Points")
plt.xlabel("x-coordinates")
plt.ylabel("y-coordinates")


obs_seq = np.array(np.zeros((10)), dtype = str)
x_store = np.array(np.zeros((10)))
running_total = 0
count = 1
y_mean = 0
pred = 0
line = 1
xmax = 0

true_line_nums = [] # the simulated data was created such that every 60 points a new line began
true_ln = 1 # the first true line begins at 1
pred_line_nums = []
for i in range(len(dat)):

    # At each time step, a new gaze point is read and fed to the algorithm.
    x = dat[i,0]
    y = dat[i,1]

    # Each new y-value is added to the running total and the count is increased, as long as the reader remains on the current line of text
    running_total += y
    y_mean = running_total/count
    count += 1

    # Putting together x_store. Before 10 gaze points of data have been received as input, store each value after the other in x_store. Once more than 10 values have been seen, store each new value in x_store and pop the oldest value out of the array
    if i <= 9:
        x_store[i] = x
    else:
        hold = np.append(x_store, x)
        x_store = hold[1:]

    # Computing xmax
    xmax = np.max(x_store)

    # This is the "reset" portion of the algorithm. If State 1 has been guessed more than 5 times in the past 10 time steps, ie: the state sequence consists of 60% State 1 predictions, then reset the running total, set xmax to be the x-value at this point in time, and reset the observation sequence to observe b1 entirely. In effect, start fresh at this new line.
    if np.sum(pred) > 5:
        xmax = x
        count = 1
        running_total = 0
        # Call this new line "line n+1"
        line += 1
        obs_seq = ['b1','b1','b1','b1','b1','b1','b1','b1','b1','b1']

    # Observations
    if i ==0:
        observation = 'b1'
    if i >= 1:
        if (x == xmax or x >= xmax-25) and abs(abs(y) - abs(y_mean))<40:
            observation = 'b1'
        elif x < xmax and abs(abs(y) < abs(y_mean)):
            observation = 'b2'
        # An old observation that I've left in in case it must be added in the future
        # elif dat[i-1,0] < x and abs(abs(y) < abs(y_mean)):
        #     observation = 'O3'

    # Observation sequence. As with x_store, fill the array with new observations first. Once it is full, pop off the oldest observation and add the newest oservation.
    if i <= 9:
        obs_seq[i] = observation
        x_store[i] = x
    else:
        hold = np.append(obs_seq, observation)
        obs_seq = hold[1:]

        # this part is just to populate an array of the true line numbers and the predicted line numbers
        if i > 9:

            if i % 60 == 0:
                true_ln += 1
                print(true_ln)

            if len(true_line_nums) == 0:
                true_line_nums = np.array([[true_ln]])
                pred_line_nums = np.array([[line]])
            else:
                true_line_nums = np.concatenate((true_line_nums, [[true_ln]]))
                pred_line_nums = np.concatenate((pred_line_nums, [[line]]))

        # Begin predictions once the observation sequence is full. The "predict" function comes built in with Pomegranate and outputs the result of the Viterbi algorithm on the given observation sequence, outputting a new array full of predicted states rather than observations.
        pred = model.predict(obs_seq)
        # print(i, observation, "Line:", line)

print("Accuracy:", sum(pred_line_nums == true_line_nums)/len(true_line_nums)*100)
plt.show()




