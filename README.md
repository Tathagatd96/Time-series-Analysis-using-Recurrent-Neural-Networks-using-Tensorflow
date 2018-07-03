# Time-series-Analysis-using-Recurrent-Neural-Networks-using-Tensorflow

## Getting Started
We will be solving a simple problem where we are generating a sine wave and providing a batch of the sine wave to the RNN and asking it to predict the next value on the batch i.e. the value one time-step ahead. This is a really simple problem(this is a beginner’s tutorial) as we are only looking one time-step ahead, but the same implementation can be applied to predict data several time-steps ahead.

Recurrent neural networks can remember the state of an input from previous time-steps which helps it to take a decision for the future time-step. Watch the animation below carefully, and make sure you understand it.(Shoutout to iamtrask blog.)
![recurrence_gif](https://user-images.githubusercontent.com/28685502/42218832-702c9104-7ee7-11e8-9ffe-a57e93e3ebe8.gif)

We generate a batch of the values on a sine wave to test our model.
![sine2_copy](https://user-images.githubusercontent.com/28685502/42218834-7061fa56-7ee7-11e8-901d-d598b00a2e38.png)


## Requirements
 * Python 3.5 or above + pip
 * Tensorflow 1.6 or above
 * Pandas
 * Numpy
 * Scikit-learn
 * Matplotlib
 
 ## Running the model
 Install the required libraries and run this file.It will take about 20 seconds depending on your processor. 
 ```python
 python tensorflow_RNN_time-series.py
 ```

 ## Results
![output](https://user-images.githubusercontent.com/28685502/42218830-6fbf4cac-7ee7-11e8-8571-2c44415d21e8.png)

This is the final output visualization of our model. The training instance indicates the batch from the current time-step. The target represents the batch from the next time-step. And, the predictions are the points that were predicted by our model for the next time-step. So, essentially the closer your prediction points are to the target, the better your model will be.



 




