# Sequence prediction analysis

The Sequence prediction analysis tests are done to analyze several different elements of different algorithms.

### Networks to Evaluate

 - Fully Connected Networks
 - LSTM Networks
 - Other RNNs (bidirectionals, GRUs, ClockWork RNNs...)
 - TCN (Temporal Convolutional Networks)
 - MANNs (different ones, to determine)
 - Networks Augmented with Attention Mechanisms
 
### Evaluation Elements:

The following elements must appear in the report of each experiment:

* Algorithm ID (name), reference to the architecture details
* Date
* Time
* Number of Parameters in the network
* Training Time (including the hardware where it was trained)
* Performance (accuracy/something else .. yet to define) at different future points
* Loss (log loss, RMS, etc) at different future points
* Performance decay on the future time, starting from the last training point
* Optimizer used (Adam, Gradient Descent, Momentum, ... etc)
* Input queue size
* Output Queue size
* Feedback Error metric (SME, absolute, log, log-loss, ...) 
* Using/Not using space-time coding (details later, this is something I want to elaborate more and research)
* Comments
* Some others that I have yet to think about

## Assumptions

* The input to the network is standardized for all the models to be tested

#### Architecture

More details in the Architecture Section

* The general architecture is stable, this is, an input queue, an output queue and an error signal that is fed back to the predictor network
* The input is temporally saved in a limited dimension input queue, this queue can or not be used by the network
* The output of the prediction is also kept in an output queue, as before this queue can or not be used by the network
* The error is *always* computed, and is fed back into the network (based on ideas from control systems) in the next time step

#### Resources

* The network must be small enough to be trained in my computer with an Intel i7 7700K and a NVidia GTX 1080

#### Training Data

* All networks will use the same data, pre-processed in the same manner, no specific pre-processing or adaptation should be done
* The input dimension to study is 1D vectors, this means vectors of size (1xN) such as N is a Natural Number
* Training data can be an audio sequence, a list of prices of an exchange, text, MIDI signal, a bitwise signal (example CommaAI-env)

## Architecture

The Architecture is separated in 2 main parts:

1. General Architecture
2. Specific Method Architecture

### General Architecture

The General Architecture can be seen in the following image:

#TODO !!!!!!!!!!!!!!!!!


The architecture consists of an input queue and an output queue. 

The input one provides the data to the predictor network (PN).

The output one stores the predicted data from the predictor network (PN).

An error signal is computed and fed back into the Predictor Network (PN).

### Specific Method Architecture

The point #2 depends on the particularities of the network (LSTM, Fully Connected, Convolutional, etc) so this will be specified in each section separately.


## Input and output queue specificities

The input and output queues can be composed of several queues, these can be the following ones:

1. The actual input data (Mandatory)
2. Time Coding of the input (Optional to be evaluated)
3. Space Coding of the input (Out of scope from the current study)


### The actual input data

This data is the data from the input signal that I'm trying to predict. 

This data can be:

 * A scalar input such as: sound signal, prices from the stock market, temperatures, linear position, ...
 * A Vector input such as text symbols, midi signals, ...
 
### Time Coding of the input
 
This is a code that signifies the TIME at which the input arrives.
 
T his idea comes from the notion that our brain contain not only the symbols arriving in due time, but also there are sensors in our body that allows us to know how much we have moved, how much time has passed, ....

The argument is that for many operations we need this data to actually 

Just to be short without to much reference for the moment (I might write more details on this later):

* in the brain there are neurons that can be stable, unstable, bi-stables and oscilating, 
* that there are some others that are sensitive to location and orientation ( V1 ),
* or spacially related (latest work by deepmind in grid neurons),
* the work on Attention Is All You Need uses positional encoding for words in a sinusoidal maner for this.


### Space Coding for the input

The idea is the same as the Time coding, but for spatial dimensions. This study does not target any evaluation in this domain but is worth to mention it for future work

