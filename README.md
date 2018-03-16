# Minibrain

As something Demis Hassabis said and profoundly inspired me: "Solve Intelligence, then use it to solve everything else"

@ Demis Hassabis, Geoffrey Hinton, Alex Graves: Would you help me?
I need:

- Mentoring (any help is welcome here)
- Time (I currently have a family and life to maintain, so time is limitted)
- Money (If I just could work on this full time, currently I fund this myself)


- Why Do I publish this in github?

Because I want to, because science needs to be public and open for everybody and because I need help.

- What help do I need?

In short, Knowledge (mentoring), time (I have a non-research job now).

Coding, creating experiments, documenting, processing power, whatever help you can think of.

## Acknowledgements:

To all the people that create knowledge and share it and to all the people that shares open source software without them the current state of science would not be possible.


## Goal

This project aims at understanding and developing a minibrain structure in which I have been studying and working on designing since beginnings 2012 when I bought the book "Neural Engineering" by Prof. Eliasmith. 

I have done several trials before and have managed to understand different parts of the systems, much of the inspiration comes from Nengo project and Professor Eliasmith, Demis Hassabis and Geoffrey Hinton. I've had many discussions with my good friend and PhD in Neuroscience Paula  Sanz Leon whom I admire profoundly. I am lucky to have many friends in the research community who have and keep inspiring me, even if they don't know.



_The project aims to:_

 - Being able to develop and train different parts independently
 - Understand different types of sensors (images, sound, text)
 - Understanding different types of predictor and reconstruction schemas
 - Understanding if/how it is possible to use attention mechanisms for different purposes
 - Trying to minimize training cost for each part (this is due to my limitted resources in time and money)
 - Try to make a single architecture that can tackle different tasks without the need to retrain (non destructive re-training)
 
 
 
_This project Stages are:_

 1. Developing different kinds of Image Autoencoders with Foveal Attention Mechanisms
 2. Use the encoding part of the autoencoders for different operations, including trying use it in a gym like scenario.
 3. ???
 ?. Create predictors for the gym like scenarios to be able to predict future scenes
 ?. use DNC or other memory mechanism to try t
 ?. Actuator mechanisms, interaction
 ???? SUCCESS ?? 
 
 
## A bit of background

My (slow and self funded) research path is inspired in dozens if not hundreds of read papers and books from Neuroscience, to psicology, Hardware, Computer Science and Behavioural Economics. I have enjoyed hundreds of hours reading about different topics and will keep doing so.

There are currently several approaches to understanding intelligence, mine is a mixture of several ones. 

The ones trying to understand the brain and find answers to how it is built (I believe that the Neural Engineering Framework from Waterloo University, Prof. Eliasmith's lab is the most interesting one up to date), there are others that are content with finding better optimal solutions to a problem, like the image recognition challenges where deep CNNs are the winning ones now.

I do believe that all of them are not only good paths, but also necessary to understand more what intelligence means and how to build it, although in my time at INRIA (Institute National de Recherche en Informatique et en Automatique - France) I also saw that research is also affected by "fashion" and "hype" which means that many things go unfunded because it is not in the "things of the moment", also I saw (and see in my friends that do work on research) how much of researchers time was spent on looking for financing instead of doing what they are good and love doing.
That is one of the reasons I finally left for the industry, and now that I can fund my own research (although slowly) I do it independently (the other is because I couldn't get funded in the subject I wanted to work on).

So now I dedicate most of my (free) time to trying to understand structures, architectures and reasons I believe that today we have most of the needed tools to start building more general intelligence that we have up to today, nevertheless this will need dedicated hardware architectures that not only allow for fast tensor operations, but can also tackle branching, which today is slow due to context changes between different elements (like a GPGPU and CPU). I do read lots of papers on Deep Learning, but I'm not an expert in statistics or the internal maths of each thing, I work on satisfying my curiosity just for the pleasure of it.

## Approach

**My approach is the following:**

- I do NOT look for making a more accurate (MNIST or other targeteted specific task here), although they should be working good enough and if it works better, GREAT!
- I DO want to create something that is more general and can learn several tasks together
- I DO want to be able to independently and incrementally train different parts of the system
- I DO want to build hierarchical structures
- I DO want to create an something that can learn to learn
- I DO want to create something that can behave according to the **context**


## I try here to list the basic principles:

- The brain is built by different structures, these structures are interconnected in different ways, hierarchies, parallel, and functions.
- Each part has a dedicated task -> Here is where I separate in _sensors_, _predictors_, _controllers_ and _actuators_
- The human (and by the way any mamalian brain) **has prior knowledge** this means we do not part from zero -> here is
- We can work on a hierarchy of concepts, each concept is embedded in a set of **contexts** that indicate the current situation giving place to abstraction levels
- Each abstraction level does not need to work at the same _speed_, higher levels need to do more internal processing, while lower levels need to **act** fast. This is inspired on the principle of reflex creation by repetition (think of one of my favorite arts, the martial arts)
- We learn to learn the target functions, think of it as an **Inverse Reinforcement Learning** problem
- We can take advantage of Computer Science and many already developed functions (like *DSP*) instead of making the networks learn the functions, make them learn to select the functions to use
- Convolutions are great, Circular convolutions are even better because they maintain the size of the vector space
- Distributed Sparse Representations are great, they allow for 
- Why not using Multinomial Probability Distribution instead of Gausian for Autoencoders? this might lead to actually start creating contexts