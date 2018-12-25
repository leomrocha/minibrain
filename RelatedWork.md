# Related work

This file tries to start writing a a bit of the ideas that are in each paper and the elements that might be nice to try in the current Few-Shot Music Generation project


### Neural Map: Structured Memory for Deep Reinforcement Learning
Emilio Parisotto, Ruslan Salakhutdinov
(Submitted on 27 Feb 2017)

Seems interesting, could use the idea of the neural-map in a generalized neural-context-map, where the map represents an N dimensional state of the current context this way we might be able to do search on the context space.
Cotext as the music style for example?

### A Simple Neural Attentive Meta-Learner (SNAIL)
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, Pieter Abbeel
15 Feb 2018 (modified: 25 Feb 2018)ICLR 2018

Quite interesting idea, it seems simpler than other mechanisms, the idea of using temporal convolutional networks with attention mechanism is on the axe of TCNs.

### Independently Controllable Features
Emmanuel Bengio, Valentin Thomas, Joelle Pineau, Doina Precup, Yoshua Bengio
(Submitted on 22 Mar 2017)

This is a great paper!!! allows for learning and separating features that control only one aspect of the output


### One-shot Learning with Memory-Augmented Neural Networks
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap
(Submitted on 19 May 2016)

Interesting simplification of NTMs to do one-shot learning. Issues with changing tasks if the memory is not cleaned up

I'd like to try this one


### Gradient Episodic Memory for Continual Learning
David Lopez-Paz and Marc’Aurelio Ranzato
2017

Interesting that the "Task Descriptors" are similar to what a  "context" would be. There is no zero-shot learning tackled in the paper, although is named as a posibility.

GREAT NEW THING: It allows for positive backwards transfer during learning, improving already learned tasks when learning new tasks. -> this implies having to have memory for previous predictions AND having to recompute the gradients for each of those before making a gradient update (which is slower). I don't see how to easilly mix it with approaches like "one-shot learning with Memory-Augmented Networks" Santoro et al. 2016


### Grid Long Short-Term Memory
Nal Kalchbrenner, Ivo Danihelka, Alex Graves
(Submitted on 6 Jul 2015 (v1), last revised 7 Jan 2016 (this version, v3))

Seems an interesting and powerful concept, might be useful as a controller for a DNC or an NTM or something else for example, although the number of parameters scares me and there is no reference to smaller sets and training difficulty.


### An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
Shaojie Bai, J. Zico Kolter, Vladlen Koltun
(Submitted on 4 Mar 2018 (v1), last revised 19 Apr 2018 (this version, v2))
Great paper, shows how to do sequence modeling with convolutional attentive networks, should use it for sequence modeling and prediction
This papers introduces the Temporal Convolutional Networks (TCNs) and there is a reference to the source code (already cloned in my github)

TCN code has many examples and seems like an interesting thing to try

### Model-Free Episodic Control
Charles Blundell, Benigno Uria, Alexander Pritzel, Yazhe Li, Avraham Ruderman, Joel Z Leibo, Jack Rae, Daan Wierstra, Demis Hassabis
(Submitted on 14 Jun 2016)

Interesting, good as model, has a nice parallel with One-Shot Learning with Memory-Augmented NNs (basically simplified NTMs), interesting overview of dimensionality reduction in Representations.
Encouraging results although it lacks a complexity and training difficulty analysis.
Episodic Control Buffer = 1M entries (much bigger than the Simplified NTMs).
Much more data efficient than other state of the art (in 2016).
I think that is overpassed by Gradient Episodic Memory for Continual Learning


### Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
(Submitted on 12 Jun 2017 (v1), last revised 6 Dec 2017 (this version, v5))
Interesting concept taking out convolutions and recurrence.
The idea of adding positional encoding with sinusoidals seems quite similar to my idea of time dependence encoding and relates to grid-neurons (deepmind 2018) and space and directional coding tuning neurons (V1)



### Performance RNN: Generating Music with Expressive Timing and Dynamics.
Ian Simon and Sageev Oore.
Magenta Blog, 2017.
https://magenta.tensorflow.org/performance-rnn

* Read website description, interesting project, although I do have the same issue as presented with the long-term structure. In my oppinion the issue with long-term structure generation is the lack of a hierarchical structure in the concept generation (and prediction) level joined with the extra difficulty given by mixing the time shift as another type of event instead of generating an encoding that could help the network separate timing from actual notes.


### Learning to Remember Rare Events
Łukasz Kaiser, Ofir Nachum, Aurko Roy, Samy Bengio
(Submitted on 9 Mar 2017)

Is great that it doesn't need to reset during training. Can be added to any part of a supervised neural network. Life long learning, memory is key-value pairs. Is for SUPERVISED classification tasks
https://github.com/RUSH-LAB/LSH_Memory


### Matching Networks for One Shot Learning
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra

Interesting about the concepts, although I need to dig much deeper to understand all the behind the scenes (sequence 2 sequence embeddings and treating the inputs as sets) from previous papers to get in in more depth


### Neural Turing Machines
Alex Graves, Greg Wayne, Ivo Danihelka
(Submitted on 20 Oct 2014 (v1), last revised 10 Dec 2014 (this version, v2))

Great paper introducing external memory, was superseeded by Differentiable Neural Computer


### Hybrid computing using a neural network with dynamic external memory
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu & Demis Hassabis

Impressive paper on extending capabilities and taking out limitations of the NTMs showing results in different tasks that are beyond any previous work.
Introduces DNCs

### Implementation and Optimization of Differentiable Neural Computers
Carol Hsin
https://web.stanford.edu/class/cs224n/reports/2753780.pdf

More details and studies on DNCs (previous paper)


### One-Shot Generalization in Deep Generative Models
Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, Daan Wierstra
(Submitted on 16 Mar 2016 (v1), last revised 25 May 2016 (this version, v2))

Presents a class of sequential generative models that are built on the principles of feedback and attention. Has great attention and iteration

Cons: Lacks self control in # of steps. Needs a lot of data


### Neural Episodic Control
Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adrià Puigdomènech, Oriol Vinyals, Demis Hassabis, Daan Wierstra, Charles Blundell
(Submitted on 6 Mar 2017)

Allows for fast assimilation of new experiences (few-shot?), the most interesting concept is the introduction of the  **Differentiable Neural Dictionary**



## Might be useful:


### Towards deep learning with segregated dendrites
Jordan Guerguiev,1,2 Timothy P Lillicrap,3 and Blake A Richards1,2,4
https://www.ncbi.nlm.nih.gov/pubmed/29205151
https://elifesciences.org/articles/22901

Amazing paper, they provide a new neuron model that is not only more biologically inspired but also provides SoTA results without the need to do backpropagation in the current way but using a local update learning rule. A comparison is done with backpropagation too.
The neuron model provides a similar implementation of what a neuron would look like in 3 modules:
1. the apical dendrites (feedback)
2. the somatical body
3. the basal dendrites
Each neuron not only generates a signal but tries to predict the next layers output. This has a parallel with Hinton's Capsules and Dinamyc Routing between Capsules.



### Compositional Attention Networks for Machine Reasoning
Drew A. Hudson, Christopher D. Manning
(Submitted on 8 Mar 2018 (v1), last revised 24 Apr 2018 (this version, v2))

https://www.facebook.com/iclr.cc/videos/2127071060655282/
(I love the way Manning explains and presents, except for the ammount of text in slides)
Limitation in what it will answer, it's not "independent" there is a big step missing

I find the architecture really good, but I don't see how to use it as a "generic module", this is due to the nature of the control that is given.


### Implementation and Optimization of Differentiable Neural Computers
Carol Hsin
https://web.stanford.edu/class/cs224n/reports/2753780.pdf

More details and studies on DNCs (previous paper)


### NEURAL RANDOM-ACCESS MACHINES
Karol Kurach  & Marcin Andrychowicz & Ilya Sutskever
2016

I don't see a direct impact in the current work but might give a few ideas for the future

It seems there are issues with the optimization algorithms: difficult to train. Nice idea on how the external memory is handled, constant access time for other memories. Interesting way of defining fuzzy pointers that can point memory. Is interesting that as CPUs it handles program, data and pointers as all data. Is interesting that the number of times-steps is NOT given by the programmer but decided by the network. Check further but might be possible to merge this approach (the modules) with the Neural Programmer, although I don't know how will it be possible to "save" and "load" programs with an approach like this one as it seems that the controller is the one learning the algorithm, there is no multi-task shown in the paper.


### A neural model of hierarchical reinforcement learning (PLoS ONE, 2017)
Daniel Rasmussen, Aaron R. Voelker, Chris Eliasmith

Impressive paper that shows how to implementreinforcement learning with a biologically plausible architecture showing SoTA and biologically comparable results. Seems quite interesting architecture for an intelligent agent as it shows a modular architecture that could also be implemented with current SoTA techniques in Deep Learning.


### Adaptive Computation Time for Recurrent Neural Networks
Graves 2017

Allows the network to decide how much time to dedicate to the computation with an upper limit.
