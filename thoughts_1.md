Thoughts About  Representation Formation & Consciousness



## Dedication:
To my family, my amazing and wife who is carrying our baby and who pushed me to take some time for my research and learning instead of taking on consultancy jobs.

I Thank all the people that have done their research public, including the companies that have contributed and contribute to the development of open science. Without all of you this work (although small) could not have been possible. Please keep doing your great job the entire world needs you.

As this is just a discussion with not too much code behind, I invite all people to help me not only discuss this, but implement the different tests and parts that need to be created in order to test the ideas here.


## Abstract:

This paper aims to present an architectural idea of how compositionality, causality and common representation can be created challenging some parts of the current techniques used in deep learning.
The idea behind is to create representations that can be used to operate with and give the network the priors to be able to learn how to use the composition and decomposition operations instead of having to learn the operations by itself, as well as giving the network the power to predict what steps caused the current state
this is also extended with a more comprehensive and modular architecture that I would love to try (and slowly building and testing different parts).
I release all these ideas in the public domain as I believe that science needs to be open, and of course, none of this is coming out of thin air, but comes from the small part of accumulated human knowledge that I’ve been able to get my hands on.


## Notes:

This paper tries to put in written things I’ve been reflecting about for a long time now, reading from different domains, from neuroscience, computer science, cognitive psychology, software architecture and I have seen a huge push forward in the last 2 years. Now I think is the moment where I can start writing even if it is only in my own time and doing experiments.

This paper aims at helping in the discussion of Consciousness and follows the paper “The Consciousness Prior” by Yoshua Bengio Some examples will be written here and source code are available. Moreover I’m an architectural and schema thinker, which means I’ll be taking more of this angle.

Note: I would love to be able to do this full time, but I have a family and I do have to work for a living. For the moment the funding for this work comes from taking partial time at work but I won’t be able to do it for much longer.


## Raw thoughts:

I challenge also the idea that all the parts of the hardware development must be generic for all AI approaches, my thought is that we can take advantage of the different parts we know already, for example the huge amount of work in Digital Signal Processing (DSP), and instead of giving the network the need to learn these operations, get the network to use the available ones, many of those operations are differentiable so they should not be impossible to adapt to gradient descent methods. In this paper I propose one of such ideas.

As our intelligence has evolved we are born with different priors, these priors allows us to create different representations during different parts of our development. During this growth our brain structures develop and change continually faster at the beginning and slower later.
Predators are usually more immature than preys at the moment of birth (you can observe for example a baby giraffe which can walk almost at the moment of birth, while humans take months to even be able to move by themselves, this might indicate that the priors encoded might be differentiated for the needed survival tasks of each species.

At the end, to be able to create real Artificial General Intelligence I think we’ll create entities that have some of the same issues, caveats and limitations than the current intelligent animals here … explain this more

Compressed representations are an intuition that can be taken from psychology, behavioural economics ( here  Dan Ariely references), many AI techniques (Autoencoders, Variational Autoencoders …. )

The idea of disentangled representations can be observed as a particularity of distributed sparse representations (eliasmith, HTMs from Numenta, Bengio ) …

The idea of attention can be seen also in the brain with the gating of lower level neurons to select the higher level ones (eliasmith)

The ideas of compositionality, causality, ….. , can be seen implemented in Cognitive Architectures

I believe I think the ideas of consciousness presented by Y. Bengio in his paper can be used in conjunction with the ones presented here

Control Theory approach can also be viewed as a clue on how many of the tasks can be done, even if it is for the sake of taking the general ideas, but the following ones can be viewed as quite important:

In Control Theory we use Frequency domain to facilitate operations i.e as product in frequency domain is the convolution in time domain. Laplace and Fourier transform can provide a useful for this goal. Also the feedback loops could be of use. In this work we explore the idea of using the error directly.
In Control Theory we study the Zeros to see the system stability, we can have stable, bi-stable, metastable, oscillatory and unstable (that we don’t want) systems. The brain also contains neurons that can have these kind of behaviours (we don’t want unstable and this might be the cause of some diseases)


## Introduction


Lets base ourselves in the priors needed to operate on the thought level instead of the
Semantic Vector Architectures (TODO references) present the opportunity of compositionality and be able to recover

The brain might act this way, the Semantic Pointer Architecture (SPA) presented by Eliasmith (TODO reference) and the experiments shown in the Nengo platform show that the SPA used with Sparse Distributed Representations can operate at different cognitive levels.

The idea of compression (non-linear) and decompression (linear) is shown also by eliasmith team in Waterloo university to correctly work in their framework

The idea of a “cleanup memory” is also presented in eliasmith works and several other papers have studied content addressable memories in the Deep Learning world (NTMs, DCNs, … TODO more references) , these kind of memories are also available since a long time ago in hardware, which I argue, could be directly used this way, and instead of worrying about the run-time in software, a lookup operation could take a few cycles for an entire memory block.

(I don’t like the current state of the art for data encoding)
Current state of the art for data encoding can be quite limiting, for example in the NLP case these representations are learned with approaches such as Word2Vec (reference) which, besides being an amazing work, limit the creation and learning of new symbols during run-time. The current work proposal also address this issue.

The idea that the brain contains different regions, and each region can either self-regulate and/or help regulate other regions (hormone secretion for example)

Taking into account the following facts:

- We could learn with infinite time and infinite data and a deep enough fully connected NN any (differentiable) function, but we do not have those conditions
- We need priors to do more smart things, this approach has given several break-throughs like convolutional neural networks, attentional mechanisms, external memories.
We have the knowledge of some of these biases like the acceleration sensors in the ear, the operations that we know from signal processing and control theory, the psychological and neuroscience studies, and so on

Let’s use those facts ...

The following priors and ideas are used in this proposal:

- Circular Convolutions (based on Holographic Representations)
- Error Feedback
- Self Contained Modules that can possibly have as input higher level elements
- Attention Mechanisms
- Memory (and maybe ... just maybe an external neural DB)
- System Stability -> in this case, as time (or spatial coding as an extension) as an extra input
- An encoding proposal that can handle unseen inputs and values (up to certain order of magnitude threshold)
- Time is an essential element of cognition and intelligent behaviour (look for some sources)
- Meta-Learning approaches


### The general idea is to generate modules that can do the following:

* Modules are “Stackable” each level of the stack can handle a higher level of abstraction (to study yet the idea

* Construct from a raw input, this construction will be done by a Memory and Attentional Augmented Neural Network (MAANN)


* The focus on the current paper is the encoding, combination and decoding of input signals

* The idea is the following the input is encoded as a vector representing an abstract element.

For the purpose of this paper, the input will be a sequential and encoded with a time-coding manner (will be presented later)

### The Layer Component contains the following modules:

#### The encoder Module:
This module contains a domain dependent encoder, sound, image, text and other information might take advantage of different encoding types and methods, as the brain does or as is used in DSP techniques.

#### The input Prediction Module:
- An input queue
- A sequence predictor MAANN
- A predicted queue (with some steps in the future) that contains also the level of confidence in each prediction
- An prediction error signal, this is constantly generated from the input and predicted queues and given back to the sequence predictor giving it the possibility to adjust to the changing situation.
- The Composition Module:
- An output queue with the composed symbols (see next point)
- A current symbol register: containing the current on-the-build representation
- A Compositional network: this network just decides if the input needs to be convolved with the current internal symbol (replacing the current symbol register), has to start creating a new one (push the new input) or discard the input. This allows the network to create different time-scales as the output queue will only be pushed a new symbol when this symbol is ready
-
#### The Causal Module:
- A Causal Network: this network operates on the composed symbol output trying to recreate the input sequence that created it
- An error signal: as the input sequence is also available
-
#### The Compression Module:
- The idea is to maintain or reduce the dimensionality of the vector while maintaining enough sparsity on the vector representation to allow compositionality without augmenting the noise, even for higher abstraction layers, this compression module does the following:
- Takes the created symbol of the output queue
- Compresses it (for example with Locality Sensitive Hashing techniques or a network whose purpose is that)
- Sends the compressed output to the next abstraction layer.
- The idea of the compressed vector can be:
    * Diminish the size of the vector
    * Make it sparser (with PCA for example), this is as more data is convolved inside, the vector becomes less sparse adding to the possibility of errors in higher abstraction layers
    * Both

#### The Decompression Module:
* Can take a higher level symbol and decompress it to the current level, this of course will make some error, as such the brain employs (eliashmith et al. ) a cleanup memory. Some other techniques can be explored
* The Cleanup Memory module:
* Maintains learned symbols such as the decompressor can remap to more original representations

#### The Controller:
* This module is there trained to do the needed task, this module might also be a MAANN

The idea of having hierarchical representations is not new, the problem is that this hierarchy can grow big, from the point of view of a single bit input (CommaAI-env) we can have bits-bytes-letters-words-phrases-paragraphs-chapters-.... And so on, so the number of levels can be too much, the great idea would be to make the previously discussed module recurrent, such as it can track not only the symbol, but the symbol level, this in the same space (even if with different referenced memories) might be convenient for being able to do some low-high level. If the

The modular separation of each layer allows for being able to learn new abstractions once a lower level was learned, we could think of it as the brain develops different abstractions at different ages
Plus the idea that each module has a clear input and output and most of them have an error signal, this allows for modular training (and continuous training from the architecture’s perspective only, there is then the target to give the particular networks the possibility to do continual learning)

Also the modular architecture can halt the learning for different parts to study their effect or put in production environments, while the MAANNs keep the property of zero-one-few shot learning.
