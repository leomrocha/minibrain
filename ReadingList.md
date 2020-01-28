# Paper Notes:

This file contains a list of papers that I've read, I'm reading or I want to read in a close future, but it keeps growing.
I just take a few notes and write down some questions and ideas on those quite fast. I don't usually check for mistakes, but I do correct them if I find them when looking for something here.

### REVISITING SMALL BATCH TRAINING FOR DEEP NEURAL NETWORKS
https://arxiv.org/pdf/1804.07612.pdf
batch sizes entre 2 y 32, no mas ni menos. (recommended by Yann LeCun)
The paper is analyzed in images, the issue with text is that most studies are done with big batches (up to size 4M), 
 No study shows the difference between small and big batches, but due to n# of parameters in current NLP tasks and 
 the variability in input and output tasks I would think that big batches might be needed. 

### Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf
https://arxiv.org/abs/1701.06538
https://openreview.net/forum?id=B1ckMDqlg

Interesting paper, although from what I'm looking for the lesson learned is that conditional execution makes a good choice and specialist/expert parts for each different task and choosing them with an expert chooser that is also trained there.


### Books and Lectures online for NLP

* [Stanford cs224](http://web.stanford.edu/class/cs224n/)
* [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html)
* [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)
* [BLEU Score and Metrics](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)


### On the Use of ArXiv as a Dataset
* [arXiv](https://arxiv.org/abs/1905.00075)
* [Associated Code in Github](https://github.com/mattbierbaum/arxiv-public-datasets) 

### MUSE: Multilingual Unsupervised and Supervised Embeddings
* https://github.com/facebookresearch/MUSE

### WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia
* https://arxiv.org/abs/1907.05791
* https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix

### Bilingual Lexicon Induction with Semi-supervision in Non-Isometric Embedding Spaces
* https://arxiv.org/abs/1908.06625

Barun Patra, Joel Ruben Antony Moniz, Sarthak Garg, Matthew R. Gormley, Graham Neubig


### The Measure of Intelligence
Francois Chollet Google, Inc. November 6, 2019
https://arxiv.org/pdf/1911.01547.pdf

### Sparse Networks from Scratch: Faster Training without Losing Performance
https://arxiv.org/abs/1907.04840
Trying to get better things from the resources we have, faster training

### DEEP DOUBLE DESCENT: WHERE BIGGER MODELS AND MORE DATA HURT
OpenAI
  * https://mltheory.org/deep.pdf
  * https://openai.com/blog/deep-double-descent/

Interesting and need to read to understand better the complexity of number of samples and model size on the model performance
TODO study it better  

*In general, the peak of test error appears systematically when models are just barely able to fit the train set.*

### LEARNING A NATURAL LANGUAGE INTERFACE WITH NEURAL PROGRAMMER 
https://openreview.net/pdf?id=ry2YOrcge

Neural network augmented with external discrete operations (database operations)
 
### ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding
This is the current SoTA paper on NLP
* [ArXiv](https://arxiv.org/abs/1907.12412)
* [Source Code](https://github.com/PaddlePaddle/ERNIE)
 
### Augmented RNNs
[blog post on the subject (2017)](https://distill.pub/2016/augmented-rnns/)

### Meta Learning

### Meta-Learning a Dynamical Language Model
Thomas Wolf, Julien Chaumond, Clement Delangue
* https://arxiv.org/abs/1803.10631
* https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a

This paper seems interesting for NLP applications (check it out and the blog post too)

    """
    As a consequence, we would like our model to store information in a multi-scale hierarchical way
    where
    1. short time-scale representations can be encoded in neural activations (hidden state),
    2. medium time-scale representations can be encoded in the dynamic of the activations by
    using dynamic weights, and
    3. long time-scale memory can be encoded in a static set of weights of the network
    """

### Differentiable plasticity: training plastic neural networks with backpropagation
Thomas Miconi, Jeff Clune, Kenneth O. Stanley
(Submitted on 6 Apr 2018 (v1), last revised 19 Jun 2018 (this version, v2))
* https://arxiv.org/pdf/1804.02464.pdf
* https://github.com/uber-research/differentiable-plasticity

Incredible paper with a great plasticity method that is easy to implement.
Number of trainable parameters duplicates, plus there is a hebbian trace to keep track of.
Memory usage is 3 times the one of a normal network but the power of the networs seems much higher.


### BACKPROPAMINE: TRAINING SELF-MODIFYING NEURAL NETWORKS WITH DIFFERENTIABLE NEUROMODULATED PLASTICITY
* https://openreview.net/pdf?id=r1lrAiA5Ym
* https://github.com/leomrocha/backpropamine

THIS Paper -> I need to implement everything from the Differential plasticity plus this one ( relatively small changes )

### Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions
* https://arxiv.org/abs/1901.01753

This paper too might be interesting for generating the training elements


### One-Shot Neural Architecture Search via Compressive Sensing
* https://arxiv.org/abs/1906.02869

### Meta-Learning of Neural Architectures for Few-Shot Learning
* https://arxiv.org/abs/1911.11090

### Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting
* https://arxiv.org/abs/1904.00310

### Online Meta-Learning
* https://arxiv.org/abs/1902.08438

### Meta-Learning Representations for Continual Learning
* https://arxiv.org/abs/1905.12588

### http://www.nld.ds.mpg.de/~michael/CNS_course_2005/papers/Woergoetter/doya02.pdf
( Old paper  from 2005?)
http://www.nld.ds.mpg.de/~michael/CNS_course_2005/papers/Woergoetter/doya02.pdf


### AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence
* https://arxiv.org/pdf/1905.10985.pdf

### Relational recurrent neural networks
* https://arxiv.org/abs/1806.01822

### RECURRENT INDEPENDENT MECHANISMS
* https://arxiv.org/pdf/1909.10893.pdf

### Metalearning and Neuromodulation 
[Kenji Doya 2002](http://www.nld.ds.mpg.de/~michael/CNS_course_2005/papers/Woergoetter/doya02.pdf)

Interesting points:
* Dopamine -> Regulation of Error in reward prediction
* Serotonine -> Time scale of neural prediction
* Noradrenaline -> Randomness in action selection
* Acetylcholine -> Speed of memory update

Long and tough paper, need to read it through more in deep

### Training Deep Nets with Sublinear Memory Cost
* https://arxiv.org/pdf/1604.06174.pdf
* https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html
* https://pytorch.org/docs/stable/checkpoint.html
Gradient checkpointing (memory at the expense of time) for DNNs from OpenAI, makes things feasible 10x larger networks feasible


## For NLP

__
### Datasets:
* [List of many datasets](http://nlpprogress.com/english/question_answering.html)
* [HotspotQA](https://hotpotqa.github.io/)
* [QAngaroo](https://qangaroo.cs.ucl.ac.uk/) 
* [Universal Dependencies Treebanks](https://universaldependencies.org/)
* [OSCAR](https://traces1.inria.fr/oscar/) -> used in [FastText](https://arxiv.org/pdf/1802.06893.pdf)
* [CommonCrawl](https://commoncrawl.org/)
* [Project Gutenberg](https://www.gutenberg.org/)
* [Project Gutenberg (French) ](https://www.gutenberg.org/wiki/FR_Page_d%27Accueil)
* [Kiwix - offline copies of free information](https://www.kiwix.org/en/downloads/kiwix-content-packages/)
* [ArXiv Complete Dataset](https://arxiv.org/help/bulk_data_s3)

### Scaling Laws for Neural Language Models
* https://arxiv.org/abs/2001.08361

WOOOOW this study is amazing and to take into account for any and avery work on NLP.
I need a more in deep reading and study of this paper to set the parameters ... But from the overview I did it's a great job.

### Multilingual Denoising Pre-training for Neural Machine Translation
* Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer
* Multilingual BART (mBART)
* https://arxiv.org/abs/2001.08210

### How Multilingual is Multilingual BERT?
* Telmo Pires, Eva Schlinger, Dan Garrette
* https://www.aclweb.org/anthology/P19-1493/
Bert is nice but has some issues

### Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT
* Shijie Wu, Mark Dredze
* https://www.aclweb.org/anthology/D19-1077/

### XQA: A Cross-lingual Open-domain Question Answering Dataset
* Jiahua Liu, Yankai Lin, Zhiyuan Liu, Maosong Sun
* https://www.aclweb.org/anthology/P19-1227/

### MLQA: Evaluating Cross-lingual Extractive Question Answering
* Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, Holger Schwenk
* https://arxiv.org/abs/1910.07475

### Learning bilingual word embeddings with (almost) no bilingual data
Mikel Artetxe, Gorka Labaka, Eneko Agirre
* https://www.aclweb.org/anthology/P17-1042.pdf

### MultiFiT: Efficient Multi-lingual Language Model Fine-tuning
* Julian Eisenschlos, Sebastian Ruder, Piotr Czapla, Marcin Kadras, Sylvain Gugger, Jeremy Howard
* https://www.aclweb.org/anthology/D19-1572/

### Deep Learning for Symbolic Mathematics
* https://arxiv.org/abs/1912.01412
* Guillaume Lample, François Charton

Interesting, need a dataset and source code though

### Reformer: The Efficient Transformer - Google
* https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html
* https://github.com/google/trax/tree/master/trax/models/reformer
* https://arxiv.org/abs/2001.04451

### Compressive Transformers for Long-Range Sequence Modelling ￼
* Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap
* 25 Sep 2019 (modified: 24 Dec 2019)
* https://openreview.net/forum?id=SylKikSYDH
THIS PAPER for long term dependencies!!! Has some nice things on the axis of some ideas I want to implement later.

* https://openreview.net/forum?id=SylKikSYDH

    """We present the Compressive Transformer, an attentive sequence model which 
    compresses past memories for long-range sequence learning. We find the 
    Compressive Transformer obtains state-of-the-art language modelling results 
    in the WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97bpc 
    respectively. We also find it can model high-frequency speech effectively 
    and can be used as a memory mechanism for RL, demonstrated on an object 
    matching task. To promote the domain of long-range sequence learning, we 
    propose a new open-vocabulary language modelling benchmark derived from books,
    PG-19. """

  "We show this can be built with simple dense linear-algebra components, such as convolutions, and can reduce both the space and compute cost
of our models ... "

### BERT-AL: BERT for Arbitrarily Long Document Understanding ￼
* Ruixuan Zhang, Zhuoyu Wei, Yu Shi, Yining Chen
* 25 Sep 2019 (modified: 24 Dec 2019)ICLR 2020 Conference Blind Submission
* https://openreview.net/forum?id=SklnVAEFDB

Uses Transformer + LSTM between transformer layers for the long term range dependency
Is something like what I want to do, except that I want to focus the LSTM [NTM|DNC|...] somewhere between the encoder and decoder while the encoder only deals with short term dependencies

### A Hierarchical Multi-task Approach for Learning Embeddings from Semantic Tasks
* https://arxiv.org/abs/1811.06031
Victor Sanh, Thomas Wolf, Sebastian Ruder


### Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes
* Jack W Rae, Jonathan J Hunt, Tim Harley, Ivo Danihelka, Andrew Senior, Greg Wayne, Alex Graves, Timothy P Lillicrap
* (Submitted on 27 Oct 2016)
* https://arxiv.org/abs/1610.09027

Neural networks augmented with external memory - Sparse Access Memory (SAM)

It seems that they have exactly what I was looking for in external memory management for the working memory (not the long term memory access and saving)

    In this paper, we present a MANN named SAM (sparse access memory). 
    By thresholding memory modifications to a sparse subset, and using 
    efficient data structures for content-based read operations, 
    our model is optimal in space and time with respect to memory size,
    while retaining end-to-end gradient based optimization. 

They also apply this to the DNC (Differentiable Neural Computer)

Maybe mixing this with LSH (Locality Sensitive Hashing) 
something good will come -> this is named in the paper as future work and 
goes well with latest Reformer Paper from Google

Also notice that the paper from Facebook on Large Memories with Product Keys does sparse read/write too 
during training (although is not the same kind of mechanism, is not a working memory) 

The tough part will be to get the datasets and tasks to train it.

### Sebastian RUDER
* [10 ML & NLP Research Highlights of 2019](https://ruder.io/research-highlights-2019/)
* [Unsupervised Cross-lingual Representation Learning](https://ruder.io/unsupervised-cross-lingual-learning/#unsupervised-deep-models)
@inproceedings{ruder2019unsupervised,
  title={Unsupervised Cross-Lingual Representation Learning},
  author={Ruder, Sebastian and S{\o}gaard, Anders and Vuli{\'c}, Ivan}, 
  booktitle={Proceedings of ACL 2019, Tutorial Abstracts},
  pages={31--38},
  year={2019}
}

### Depthwise Separable Convolutions for Neural Machine Translation
* Lukasz Kaiser, Aidan N. Gomez, Francois Chollet
* https://arxiv.org/abs/1706.03059


### Cross-lingual Language Model Pretraining
Guillaume Lample, Alexis Conneau
* https://arxiv.org/abs/1901.07291
* [source code](https://github.com/facebookresearch/XLM)


###Discrete Autoencoders for Sequence Models
* Łukasz Kaiser, Samy Bengio
* https://arxiv.org/abs/1801.09797


### Swag: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference 
* https://arxiv.org/pdf/1808.05326.pdf
SWAG Dataset


### Annotation Artifacts in Natural Language Inference Data
https://arxiv.org/abs/1803.02324

### BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
* https://arxiv.org/pdf/1910.13461.pdf
* This paper shows a sequence to sequence autoencoder denoising objective

 
### Are distributional representations ready for the real world? Evaluating word vectors for grounded perceptual meaning
* https://arxiv.org/abs/1705.11168

### Unsupervised Cross-lingual Representation Learning at Scale
https://arxiv.org/abs/1911.02116
Facebook Research, SoTA on multilingual models by far.

Studies how and what to train to improve as much as possible low resource languages while maintaining strong performance in high resource language datasets.

### T5 - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
https://arxiv.org/pdf/1910.10683.pdf
This is the biggest and SoTA paper by Google on NLP, the models are HUGE (up to 11B parameters)
paper is long (53 pages)
Need to read it, study it, overfit on it :p

Interesting, really interesting. Many lessons to get from there, a hugh study on different parameters, techniques and other elements.

If there is one paper to read on NLP this is the one. An overview of the state of the art, the progress, what are the important things to learn, etc.  

### Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference
https://www.aclweb.org/anthology/P19-1334.pdf

### The Bitter Lesson - Rich Sutton - March 13, 2019
http://www.incompleteideas.net/IncIdeas/BitterLesson.html
Interesting read, ignores the case of the curation of the data for training and other aspects but it goes into the subject of Processing power >>> than manual decisions 

### ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators 
https://openreview.net/forum?id=r1xMH1BtvB
https://openreview.net/pdf?id=r1xMH1BtvB

Abstract: While masked language modeling (MLM) pre-training methods such as BERT produce excellent results on downstream NLP tasks, they require large amounts of compute to be effective. These approaches corrupt the input by replacing some tokens with [MASK] and then train a model to reconstruct the original tokens. As an alternative, we propose a more sample-efficient pre-training task called replaced token detection. Instead of masking the input, our approach corrupts it by replacing some input tokens with plausible alternatives sampled from a small generator network...

Proposes another way of training with some ideas from GANs (but not gans) and changing the objective to detecting which tokens are generated and which are false.
Seems interesting and at the same time seems more difficult to create and tune as needs to be trained , tuned and hyper-parameters found.

### Learning Word Vectors for 157 Languages
https://arxiv.org/abs/1802.06893

### Parameter-Efficient Transfer Learning for NLP
((Adapters: A Compact and Extensible Transfer Learning Method for NLP))
https://arxiv.org/pdf/1902.00751.pdf
Interesting and similar to some of my ideas (Adapters network for transfer and adding new tasks)

The Adapter Modules are trained (and grow ?) for each new Task, these are then fine tuned while the rest of the network is frozen.

### BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning
https://arxiv.org/abs/1902.02671
This paper shows a way of using something n the style of Adapter Modules, but instead of adding it inside the transformer layer it adds it in parallel with a smaller attention, it also evaluates a serial element instead of a parallel one.
 

### ERNIE 2.0: A CONTINUAL PRE-TRAINING FRAMEWORK FOR LANGUAGE UNDERSTANDING
https://arxiv.org/pdf/1907.12412.pdf
https://github.com/PaddlePaddle/ERNIE

This one seems interesting:

From Abstract: ".... In order to extract to the fullest extent, the lexical,
syntactic and semantic information from training corpora, we propose a continual pre-training framework named ERNIE 2.0 which builds and learns incrementally pre-training tasks through constant
multi-task learning..."

### On the Variance of the Adaptive Learning Rate and Beyond
https://arxiv.org/abs/1908.03265

### Semi-supervised Multitask Learning for Sequence Labeling
https://arxiv.org/abs/1704.07156

### Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges
This paper
https://arxiv.org/pdf/1907.05019.pdf
https://medium.com/syncedreview/google-introduces-huge-universal-language-translation-model-103-languages-trained-on-over-25-74f0eb71b177
Great Paper to check ideas and see wht works and what does not and why

### Parameter-Efficient Transfer Learning for NLP
https://arxiv.org/pdf/1902.00751.pdf
Technique used/evaluated in T5-Google

### Professor Forcing: A New Algorithm for Training Recurrent Networks
https://arxiv.org/pdf/1610.09038.pdf
Professor forcing, a technique modified from Teacher Forcing

### Answering Complex Open-domain Questions Through Iterative Query Generation
https://arxiv.org/pdf/1910.07000.pdf
http://ai.stanford.edu/blog/answering-complex-questions/
Interesting method and training, still we are far from a real solution. Uses DrQA and other things
Code available with stanfordnlp framework

### Can You Tell Me How to Get Past Sesame Street? Sentence-Level Pretraining Beyond Language Modeling
https://www.aclweb.org/anthology/P19-1439.pdf

### Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting
https://arxiv.org/abs/1904.00310
Interesting but not what I'm looking for. This one use Neural Architecture Search (NAS) too resource intensive for me.

### Revisiting Character-Based Neural Machine Translation with Capacity and Compression
https://www.aclweb.org/anthology/D18-1461.pdf
This one is interesting
Is basically one of the papers I was looking for. More depth solves the issue of character level. Also they outperform other models.
They study different NMT networks and how they work.

### Cognitive Graph for Multi-Hop Reading Comprehension at Scale
https://arxiv.org/pdf/1905.05460.pdf
This paper seems interesting and on the lines of what I've been working on
I have to give it a more thorough read

### Simple and Effective Multi-Paragraph Reading Comprehension
https://arxiv.org/abs/1710.10723

### MULTI-STEP RETRIEVER-READER INTERACTION FOR SCALABLE OPEN-DOMAIN QUESTION ANSWERING
https://openreview.net/pdf?id=HkfPSh05K7

### Multi-hop Reading Comprehension through Question Decomposition and Rescoring
https://www.aclweb.org/anthology/P19-1613.pdf

### The Web as a Knowledge-base for Answering Complex Questions
https://www.aclweb.org/anthology/N18-1059.pdf

### Latent Retrieval for Weakly Supervised Open Domain Question Answering
https://arxiv.org/pdf/1906.00300.pdf

### CodeSearchNet Challenge: Evaluating the State of Semantic Code Search
https://arxiv.org/pdf/1909.09436.pdf

### R3: Reinforced Ranker-Reader for Open-Domain Question Answering
https://arxiv.org/pdf/1709.00023.pdf

### Generalization through Memorization: Nearest Neighbor Language Models
* https://arxiv.org/abs/1911.00172v1

  
### Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks
* Kuan Fang, Alexander Toshev, Li Fei-Fei, Silvio Savarese
* (Submitted on 9 Mar 2019)
* https://arxiv.org/abs/1903.03878
* https://arxiv.org/pdf/1903.03878.pdf

This paper is interesting in the memory management, defines the Scene Memory Transformer (SMT) for robotics, but maybe is something similar to what I need to do in my work 

    Although the scene memory grows linearly with the
    length of the episode, it stores only an embedding vector at
    each steps. Therefore, we can easily store hundreds of observations without any burden in the device memory. This
    overhead is justified as it gives us higher performance compared to established policies with more compact memories.
    Further, as the computational complexity of the original model grows quadratically with the size of the scene
    memory, we introduce a memory factorization procedure as
    part of SMT. This reduces the computational complexity to
    linear. The procedure is applied when the number of the
    stored observations is high. In this way, we can leverage
    a large memory capacity without the taxing computational
    overhead of the original model.

### Deep Equilibrium Models

* https://arxiv.org/abs/1909.01377v2
__
### A Comprehensive Survey on Graph Neural Networks
https://arxiv.org/abs/1901.00596

### Graph Neural Networks: A Review of Methods and Applications
https://arxiv.org/pdf/1812.08434.pdf

## Graph Convolutional Neural Networks for Web-Scale Recommender Systems
https://arxiv.org/pdf/1806.01973.pdf
Pinterest PinSage algorithm NN

### Tutorial on Graph Neural Networks
https://medium.com/@BorisAKnyazev/tutorial-on-graph-neural-networks-for-computer-vision-and-beyond-part-1-3d9fada3b80d

### How Powerful are Graph Neural Networks?
https://arxiv.org/abs/1810.00826

### Dynamic Coattention Networks For Question Answering
https://arxiv.org/abs/1611.01604

### Levenshtein Transformer
THIS ONE !!! as I was looking on how to do this
https://arxiv.org/pdf/1905.11006.pdf

### Pay Less Attention with Lightweight and Dynamic Convolutions
https://arxiv.org/pdf/1901.10430.pdf

### ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS
 https://openreview.net/pdf?id=H1eA7AEtvS
 https://arxiv.org/pdf/1909.11942.pdf

### Neural Text Generation with Unlikelihood Training
https://arxiv.org/abs/1908.04319 

### Adaptive Attention Span in Transformers
https://arxiv.org/abs/1905.07799

### Augmenting Self-attention with Persistent Memory
https://arxiv.org/abs/1907.01470

### Large Memory Layers with Product Keys
https://arxiv.org/abs/1907.05242

### Language Modeling with Gated Convolutional Networks
https://arxiv.org/abs/1612.08083

### Ilustrated Transformer
 http://jalammar.github.io/illustrated-transformer/

### Memory Augmented Neural Network Adaptive Controllers: Performance and Stability
https://arxiv.org/pdf/1905.02832.pdf

###  Generating Long Sequences with Sparse Transformers 

### Convolutional Self-Attention Networks
https://arxiv.org/abs/1904.03107

### Self-Attention with Relative Position Representations
https://arxiv.org/pdf/1803.02155.pdf

### The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
https://arxiv.org/abs/1611.09326


## Other papers

### Hierarchical Attention Networks for Document Classification
* Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy
* https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
* https://www.aclweb.org/anthology/N16-1174/

Interesting concept but still has the problems I want to solve, dynamic input lengths, more flexibility ...

### A distributional code for value in dopamine-based reinforcement learning
Will Dabney 1,5 *, Zeb Kurth-Nelson 1,2,5 , Naoshige Uchida 3 , Clara Kwon Starkweather 3 ,
Demis Hassabis 1 , Rémi Munos 1 & Matthew Botvinick 1
https://doi.org/10.1038/s41586-019-1924-6
DeepMind

THIS PAPER IS AMAZING!!!, if solely can be mixed with the ones from UberAILabs

### Deep Equilibrium Models
https://arxiv.org/pdf/1909.01377.pdf

### Convolution with Logarithmic Filter Groups for Efficient Shallow CNN
https://arxiv.org/pdf/1707.09855.pdf


### Dual Attention Network for Scene Segmentation
https://arxiv.org/pdf/1809.02983.pdf

###  Attention Augmented Convolutional Networks
https://arxiv.org/pdf/1904.09925v1.pdf
This idea is great for NLP too in 1D convolutions

### Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE)


### Universal Language Model Fine-tuning for Text Classification



### Compact Bilinear Pooling
https://arxiv.org/abs/1511.06062

### NEURAL GPUS LEARN ALGORITHMS
Łukasz Kaiser & Ilya Sutskever 2016

Interesting and seems to add into the turing complete neural networks
Has big issues with training -> low success rate (1 in almost 800) and resource intensive


### Improving the Neural GPU Architecture for Algorithm Learning
Karlis Freivalds, Renars Liepins
Much easier to train than the previous paper, architecture also a bt simpler, stil some weak points in training and for actual use in real situation

Dig deper here, this could be the basis of a complex controller for a compositional and memory based program AI


### NEURAL PROGRAMMER : INDUCING LATENT PROGRAMS WITH GRADIENT DESCENT
Arvind Neelakantan
University of Massachusetts Amherst
Quoc V. Le
Ilya Sutskever
2016
It creates programs but needs LSTM memories, hard to train, the programs are not explicit on an external memory but saved in the weights, this does not allow for ease of compositionality or interchangeability


### NEURAL PROGRAMMER-INTERPRETERS
Scott Reed & Nando de Freitas
2016

Needs high level of supervision, problematic. Has a "library of programs", has "compositional" program structure with a memory, sub-programs can receive arguments. Contains a good list of references to the same higher level ideas I have.
Dig Deeper here to understand how the compositional structure is handled and understand better the limitations to see how to improve it


### Attentive Recurrent Comparators
Pranav Shyam, Shubham Gupta, Ambedkar Dukkipati

Interesting, check again for the Vision part, should be able to use this fovea-style instead of my multiresolution one (which is not necesarilly differentiable).
Idea: Evaluate the effects of multi-resolution to help in the current adversarial examples


### Neural Map: Structured Memory for Deep Reinforcement Learning
Emilio Parisotto, Ruslan Salakhutdinov
(Submitted on 27 Feb 2017)

*Overviewed, seems interesting, could use the idea of the neural-map in a generalized neural-context-map, where the map represents an N dimensional state of the current context this way we might be able to do search on the context space


### A Simple Neural Attentive Meta-Learner (SNAIL)
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, Pieter Abbeel
15 Feb 2018 (modified: 25 Feb 2018)ICLR 2018

Quite interesting idea, it seems simpler than other mechanisms, the idea of using temporal convolutional networks with attention mechanism is on the axe of TCNs and the ideas I have for a predictive network

I find that there lacks comparison with other attentional mechanisms MANNs on one-shot learning to understand if it actually works

There are a few points I have to dig deeper to understand the implementation of it

### The Consciousness Prior (Project in progress)
Yoshua Bengio

Interesting starter idea, but there is stil the question on what to do as a toy example and how to measure it?



### Adaptive Computation Time for Recurrent Neural Networks
Graves 2017

*Read -> re-read to understand it better


### CommAI: Evaluating the first steps towards a useful general AI
Marco Baroni, Armand Joulin, Allan Jabri, Germàn Kruszewski, Angeliki Lazaridou, Klemen Simonic, Tomas Mikolov
(Submitted on 31 Jan 2017 (v1), last revised 27 Mar 2017 (this version, v2))

Interesting, introduces a framework for AGI evaluation, will have to check the framework and see how to use it. Might be a nice benchmark for the predictive hierarchical symbol generation that I want to build in a future


### One Model To Learn Them All
Lukasz Kaiser, Aidan N. Gomez, Noam Shazeer, Ashish Vaswani, Niki Parmar, Llion Jones, Jakob Uszkoreit
(Submitted on 16 Jun 2017)

TODO Read again and check what I can do with it to compose different ideas now that I've read so much since then


### Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience
Ross W. Gayler
(Submitted on 13 Dec 2004)

Interesting and shows examples on how to use and apply examples to solve the issues there. Symbolic Architectures seem quite simple in the concept (TODO extend the current 1D implementations that are in this repo to 2D and 3D, I'll need them for later)


### Vector Symbolic Architectures: A New Building Material for Artificial General Intelligence
Simon D. LEVY and Ross GAYLER

Is just a few references to other actually interesting (and long) papers, nothing else


### Independently Controllable Features
Emmanuel Bengio, Valentin Thomas, Joelle Pineau, Doina Precup, Yoshua Bengio
(Submitted on 22 Mar 2017)

This is a great paper!!! might be useful for the selection (or not) of the circular convolution stage (holographic dimensionality reduction)  before the compression algorithm -> this might also be great for the Neural Programmer and Neural Programmer Interpreters papers



### A Roadmap Towards Machine Intelligence
November 25, 2015
Tomas Mikolov, Armand Joulin, Marco Baroni

I like the paper, I agree on most of the points, most of the points argee with the other papers on the subject (by deepmind and by others too) although is more comprehensive in the actual tasks descriptions with toy examples (presents CommaAI-env ideas) -> presents the idea of long term memory (in my idea is a DB like a rocksDB extension) -> Read again in detail (and the other 2 papers on the subject + the ones in neurobiology) and write a paper on those + some simulation with the commaAI-env, the billiards task and the bAbI tasks


### NEURAL RANDOM-ACCESS MACHINES
Karol Kurach  & Marcin Andrychowicz & Ilya Sutskever
2016

It seems there are issues with the optimization algorithms: difficult to train. Nice idea on how the external memory is handled, constant access time for other memories. Interesting way of defining fuzzy pointers that can point memory. Is interesting that as CPUs it handles program, data and pointers as all data. Is interesting that the number of times-steps is NOT given by the programmer but decided by the network. Check further but might be possible to merge this approach (the modules) with the Neural Programmer, although I don't know how will it be possible to "save" and "load" programs with an approach like this one as it seems that the controller is the one learning the algorithm, there is no multi-task shown in the paper.


### One-shot Learning with Memory-Augmented Neural Networks
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap
(Submitted on 19 May 2016)

Interesting simplification of NTMs to do one-shot learning. Issues with changing tasks if the memory is not cleaned up


### Gradient Episodic Memory for Continual Learning
David Lopez-Paz and Marc’Aurelio Ranzato
2017

Interesting that the "Task Descriptors" are similar to what I call  "context". There is no zero-shot learning tackled in the paper, although is named as a posibility.

GREAT NEW THING: It allows for positive backwards transfer during learning, improving already learned tasks when learning new tasks. -> this implies having to have memory for previous predictions AND having to recompute the gradients for each of those before making a gradient update (which is slower). I don't see how to easilly mix it with approaches like "one-shot learning with Memory-Augmented Networks" Santoro et al. 2016


### Grid Long Short-Term Memory
Nal Kalchbrenner, Ivo Danihelka, Alex Graves
(Submitted on 6 Jul 2015 (v1), last revised 7 Jan 2016 (this version, v3))

Do the math to actually get it. Seems an interesting and powerful concept, might be useful as a controller for a DNC or an NTM or something else for example, although the number of parameters scares me and there is no reference to smaller sets and training difficulty.


### The Recurrent Temporal Restricted Boltzmann Machine  (the billiards task)
Part of: Advances in Neural Information Processing Systems 21 (NIPS 2008)
Ilya Sutskever Geoffrey E. Hinton Graham W. Taylor

Overviewed -> I just read enough to understand what was the billiards task, nothing else.


### An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
Shaojie Bai, J. Zico Kolter, Vladlen Koltun
(Submitted on 4 Mar 2018 (v1), last revised 19 Apr 2018 (this version, v2))
Great paper, shows how to do sequence modeling with convolutional attentive networks, should use it for sequence modeling and prediction
This papers introduces the Temporal Convolutional Networks (TCNs) and there is a reference to the source code (already cloned in my github)


### What is consciousness, and could machines have it?
Dehaene et al. (2017).

Read (an incomplete version) -> well, seems to agree with the rest of the literature, no new thing, but a compiled short version of it. Good read.



### Model-Free Episodic Control
Charles Blundell, Benigno Uria, Alexander Pritzel, Yazhe Li, Avraham Ruderman, Joel Z Leibo, Jack Rae, Daan Wierstra, Demis Hassabis
(Submitted on 14 Jun 2016)

Interesting, good as model, has a nice parallel with One-Shot Learning with Memory-Augmented NNs (basically simplified NTMs), does not have an exploitation of my idea on hierarchical experience memory DB, interesting overview of dimensionality reduction in Representations.
Encouraging results although it lacks a complexity and training difficulty analysis.
Episodic Control Buffer = 1M entries (much bigger than the Simplified NTMs).
Much more data efficient than other state of the art (in 2016).
I think that is overpassed by Gradient Episodic Memory for Continual Learning


### Compositional Attention Networks for Machine Reasoning
Drew A. Hudson, Christopher D. Manning
(Submitted on 8 Mar 2018 (v1), last revised 24 Apr 2018 (this version, v2))

https://www.facebook.com/iclr.cc/videos/2127071060655282/
(I love the way Manning explains and presents, except for the ammount of text in slides)
Limitation in what it will answer, it's not "independent" there is a big step missing

I find the architecture really good, but I don't see how to use it as a "generic module", this is due to the nature of the control that is given. Also I don't understand how to generalize the READ part, should it be done differently for each type of input sensory information? should be done by a hierarchical level architecture?


### Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
(Submitted on 12 Jun 2017 (v1), last revised 6 Dec 2017 (this version, v5))
Interesting concept taking out convolutions and recurrence.
The idea of adding positional encoding with sinusoidals seems quite similar to my idea of time dependence encoding and relates to grid-neurons (deepmind 2018) and space and directional coding tuning neurons (V1)


### Biologically Plausible, Human-scale Knowledge Representation (35th Annual Conference of the Cognitive Science Society, 2013)
Eric Crawford, Matthew Gingerich, Chris Eliasmith

Interesting, and explains a simple cleanup memory implementation, although all is built in instead of learned. Interesting demonstration of the depth at which this works (up to 20 and really good performance). The architecture can encode and decode complex interconnected knowledge and answer questions.


### Performance RNN: Generating Music with Expressive Timing and Dynamics.
Ian Simon and Sageev Oore.
Magenta Blog, 2017.
https://magenta.tensorflow.org/performance-rnn

* Read website description, interesting project, although I do have the same issue as presented with the long-term structure. In my oppinion the issue with long-term structure generation is the lack of a hierarchical structure in the concept generation (and prediction) level joined with the extra difficulty given by mixing the time shift as another type of event instead of generating an encoding that could help the network separate timing from actual notes.


### Learning to Remember Rare Events
Łukasz Kaiser, Ofir Nachum, Aurko Roy, Samy Bengio
(Submitted on 9 Mar 2017)

Is great that it doesn't need to reset during training. Can be added to any part of a supervised neural network. Life long learning, memory is key-value pairs. The Values being INTs might limit funcitonality for more complex representations?? Is for SUPERVISED classification tasks -> Could it be used as a cleanup memory???!!!
https://github.com/RUSH-LAB/LSH_Memory


### Matching Networks for One Shot Learning
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, Daan Wierstra

Interesting about the concepts, although I need to dig much deeper to understand all the behind the scenes (sequence 2 sequence embeddings and treating the inputs as sets) from previous papers to get in in more depth

TODO read references and come back to this one


### Towards deep learning with segregated dendrites
Jordan Guerguiev,1,2 Timothy P Lillicrap,3 and Blake A Richards1,2,4
https://www.ncbi.nlm.nih.gov/pubmed/29205151
https://elifesciences.org/articles/22901

Amazing paper, they provide a new neuron model that is not only more biologically inspired but also provides SoTA results without the need to do backpropagation in the current way but using a local update learning rule. A comparison is done with backpropagation too.
The neuron model provides a similar implementation of what a neuron would look like in 3 modules:
1. the apical dendrites (feedback)
2. the somatical body
3. the basal dendrites


### Neural Turing Machines
Alex Graves, Greg Wayne, Ivo Danihelka
(Submitted on 20 Oct 2014 (v1), last revised 10 Dec 2014 (this version, v2))

Great paper introducing external memory, was superseeded by Differentiable Neural Computer


### Hybrid computing using a neural network with dynamic external memory
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu & Demis Hassabis

Impressive paper on extending capabilities and taking out limitations of the NTMs showing results in different tasks that are beyond any previous work.


### Implementation and Optimization of Differentiable Neural Computers
Carol Hsin
https://web.stanford.edu/class/cs224n/reports/2753780.pdf

More details and studies on DNCs (previous paper)


### One-Shot Generalization in Deep Generative Models
Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, Daan Wierstra
(Submitted on 16 Mar 2016 (v1), last revised 25 May 2016 (this version, v2))

Presents a class of sequential generative models that are built on the principles of feedback and attention
Has great attention and iteration

Lacks self control in # of steps, maybe due to the fact that it lacks compositionality and causality?
Needs a lot of data


### Neural Episodic Control
Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adrià Puigdomènech, Oriol Vinyals, Demis Hassabis, Daan Wierstra, Charles Blundell
(Submitted on 6 Mar 2017)

Allows for fast assimilation of new experiences (few-shot?), the most interesting concept is the introduction of the  **Differentiable Neural Dictionary**

### Rainbow: Combining Improvements in Deep Reinforcement Learning
Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver
(Submitted on 6 Oct 2017)

Compares different RL techniques and then merges them into one model. It shows that no single model wins, but combining them makes the model have a much better results and faster (orders of magnitude) trainig performance.
Complex to implement such a system


### Reinforcement Learning Neural Turing Machines - Revised
Wojciech Zaremba, Ilya Sutskever
(Submitted on 4 May 2015 (v1), last revised 12 Jan 2016 (this version, v3))

Explores the possibility to use RL to add different non-differentiable discrete interfaces to the NTM. It seems feasible but learning memory access patterns with the REINFORCE algorithm is difficult.


### A neural model of hierarchical reinforcement learning (PLoS ONE, 2017)
Daniel Rasmussen, Aaron R. Voelker, Chris Eliasmith

Impressive paper that shows how to implementreinforcement learning with a biologically plausible architecture showing SoTA and biologically comparable results. Seems quite interesting architecture for an intelligent agent as it shows a modular architecture that could also be implemented with current SoTA techniques in Deep Learning

### Distral: Robust Multitask Reinforcement Learning
Yee Whye Teh, Victor Bapst, Wojciech Marian Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, Razvan Pascanu
(Submitted on 13 Jul 2017)

Tries to tackle the issues that arise when doing multi-task learning, although I did not dig too much on it.

### Understanding deep learning requires rethinking generalization
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
(Submitted on 10 Nov 2016 (v1), last revised 26 Feb 2017 (this version, v2))

IMPRESSIVE PAPER -> shows that our current intuitions about regularization and generalization might be (are) wrong ... we don't actually know what's happening


### A neural model of the development of expertise.
Travis DeWolf, Chris Eliasmith
http://compneuro.uwaterloo.ca/files/publications/aubin.2016a.pdf

Relates the fact that learniing expertise is also known as developing *automaticity* and how this might be created and somehow *standardized* in the cortex and basal ganglia. I might have to read it again and dig deeper.


### Learning to select data for transfer learning with Bayesian Optimization
Sebastian Ruder, Barbara Plank
(Submitted on 17 Jul 2017)

Uses bayesian learning to try to create a model-independent approach to learn how to select data to be transfered. Is not my main interest but might be useful.


### Kickstarting Deep Reinforcement Learning
Simon Schmitt, Jonathan J. Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M. Czarnecki, Joel Z. Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, S. M. Ali Eslami
(Submitted on 10 Mar 2018)

Interesting paper taking a more closer approach the way humans learn from examples
Makes easier to iterate and improve learned tasks using a teacher-student approach where the teacher does NOT show every detail (so is not imitation) of a recorded experience.
The main idea is to employ an auxiliary loss function which encourages the student policy to be close to the teacher policy on trajectories sampled by the student

### Transforming Auto-encoders
G. E. Hinton, A. Krizhevsky & S. D. Wang

Introduces the idea of capsules in a computer vision setup that allow for better learning of a classification task, including ones with superimposed examples. Introduces more complex MNIST variations.

Problems: really slow to train, needs to have all the categories know as it can not deal correctly with "everything else"


### Dynamic Routing Between Capsules
Sara Sabour, Nicholas Frosst, Geoffrey E Hinton
(Submitted on 26 Oct 2017 (v1), last revised 7 Nov 2017 (this version, v2))

Continuation of Transforming Auto-encoders introducing the novel concept of routing between capsules that allows it to predict what the higher layers will need making this a recursive algorithm. This concept is validated by "Towards deep learning with segregated dendrites" Guerguiev et al.


# Papers yet to read for Few-Shot-Music-Generation ....

### Dynamic Evaluation of Neural Sequence Models
Ben Krause, Emmanuel Kahembwe, Iain Murray, Steve Renals
(Submitted on 21 Sep 2017 (v1), last revised 25 Oct 2017 (this version, v2))



# Papers yet to read ....

Visualizing the Loss Landscape of Neural Nets
http://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf

Deep Learning: A Critical Appraisal
Gary Marcus
(Submitted on 2 Jan 2018)

Deep Clustering for Unsupervised Learning of Visual Features
Mathilde Caron, Piotr Bojanowski, Armand Joulin, Matthijs Douze
(Submitted on 15 Jul 2018)

Long-term Recurrent Convolutional Networks for Visual Recognition and Description
Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio Guadarrama, Kate Saenko, Trevor Darrell
(Submitted on 17 Nov 2014 (v1), last revised 31 May 2016 (this version, v4))


Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo
(Submitted on 13 Jun 2015 (v1), last revised 19 Sep 2015 (this version, v2))

Analyzing Uncertainty in Neural Machine Translation
Myle Ott, Michael Auli, David Grangier, Marc'Aurelio Ranzato
(Submitted on 28 Feb 2018 (v1), last revised 12 Jun 2018 (this

Differentiable plasticity: training plastic neural networks with backpropagation
Thomas Miconi, Jeff Clune, Kenneth O. Stanley
(Submitted on 6 Apr 2018 (v1), last revised 19 Jun 2018 (this version, v2))
https://github.com/uber-research/differentiable-plasticity

# Papers yet to read ....


Deep Learning: A Critical Appraisal
Gary Marcus
(Submitted on 2 Jan 2018)

Probabilistic Model-Agnostic Meta-Learning
Chelsea Finn, Kelvin Xu, Sergey Levine
(Submitted on 7 Jun 2018)


Adaptive Engagement of Cognitive Control in Context-Dependent Decision Making
Michael L. Waskom  Michael C. Frank  Anthony D. Wagner
Cerebral Cortex, Volume 27, Issue 2, 1 February 2017, Pages 1270–1284, https://doi.org/10.1093/cercor/bhv333
Published: 04 January 2016
https://academic.oup.com/cercor/article/27/2/1270/3056315?keytype=ref&ijkey=5hjFprzQ7miiYZ4


A Hierarchical Recurrent Neural Network for Symbolic Melody Generation
Jian Wu, Changran Hu, Yulong Wang, Xiaolin Hu, Jun Zhu
(Submitted on 14 Dec 2017)


Learning a Latent Space of Multitrack Measures
Ian Simon, Adam Roberts, Colin Raffel, Jesse Engel, Curtis Hawthorne, Douglas Eck
(Submitted on 1 Jun 2018)


Convolutional Sequence to Sequence Learning
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin
(Submitted on 8 May 2017 (v1), last revised 25 Jul 2017 (this version, v3))

Learning Disentangled Representations with Semi-Supervised Deep Generative Models
N. Siddharth, Brooks Paige, Jan-Willem van de Meent, Alban Desmaison, Noah D. Goodman, Pushmeet Kohli, Frank Wood, Philip H.S. Torr
(Submitted on 1 Jun 2017 (v1), last revised 13 Nov 2017 (this version, v2))


Unsupervised Predictive Memory in a Goal-Directed Agent
Greg Wayne, Chia-Chun Hung, David Amos, Mehdi Mirza, Arun Ahuja, Agnieszka Grabska-Barwinska, Jack Rae, Piotr Mirowski, Joel Z. Leibo, Adam Santoro, Mevlana Gemici, Malcolm Reynolds, Tim Harley, Josh Abramson, Shakir Mohamed, Danilo Rezende, David Saxton, Adam Cain, Chloe Hillier, David Silver, Koray Kavukcuoglu, Matt Botvinick, Demis Hassabis, Timothy Lillicrap


Convolutional Sequence to Sequence Learning
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin
(Submitted on 8 May 2017 (v1), last revised 25 Jul 2017 (this version, v3))



Edge Attention-based Multi-Relational Graph Convolutional Networks
Chao Shang, Qinqing Liu, Ko-Shin Chen, Jiangwen Sun, Jin Lu, Jinfeng Yi, Jinbo Bi
(Submitted on 14 Feb 2018 (v1), last revised 20 May 2018 (this version, v2))


Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning
Qimai Li, Zhichao Han, Xiao-Ming Wu
(Submitted on 22 Jan 2018)
https://tkipf.github.io/graph-convolutional-networks/

FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling
Jie Chen, Tengfei Ma, Cao Xiao
(Submitted on 30 Jan 2018)


Convolutional Sequence to Sequence Learning
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin
(Submitted on 8 May 2017 (v1), last revised 25 Jul 2017 (this version, v3))


Rotational Unit of Memory
Rumen Dangovski, Li Jing, Marin Soljacic
(Submitted on 26 Oct 2017)


A Clockwork RNN
Jan Koutník, Klaus Greff, Faustino Gomez, Jürgen Schmidhuber
(Submitted on 14 Feb 2014)

OptNet: Differentiable Optimization as a Layer in Neural Networks
Brandon Amos, J. Zico Kolter
(Submitted on 1 Mar 2017 (v1), last revised 12 Jan 2018 (this version, v3))

Input Convex Neural Networks
Brandon Amos, Lei Xu, J. Zico Kolter
(Submitted on 22 Sep 2016 (v1), last revised 14 Jun 2017 (this version, v3))

Where Do Rewards Come From?
Satinder Singh Richard L. Lewis Andrew G. Barto


Evolved Policy Gradients
Rein Houthooft Richard Y. Chen Phillip Isola  Bradly C. Stadie Jonathan Ho Pieter Abbeel


Associative Long Short-Term Memory
Ivo Danihelka, Greg Wayne, Benigno Uria, Nal Kalchbrenner, Alex Graves
(Submitted on 9 Feb 2016 (v1), last revised 19 May 2016 (this version, v2))
Complex numbers in Deep Learning ->  checks some ideas on Holographics representations


Unitary Evolution Recurrent Neural Networks
Martin Arjovsky, Amar Shah, Yoshua Bengio
(Submitted on 20 Nov 2015 (v1), last revised 25 May 2016 (this version, v4))
Complex numbers too here



“Two-Stage Synthesis Networks for Transfer Learning in Machine Comprehension”, Microsoft’s AI researchers
http://mrc2018.cipsc.org.cn/
https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/emnlp17_SynNet.pdf
http://tcci.ccf.org.cn/summit/2017/dlinfo/003.pdf
http://mrc2018.cipsc.org.cn/



A Hierarchical Recurrent Neural Network for Symbolic Melody Generation
Jian Wu, Changran Hu, Yulong Wang, Xiaolin Hu, Jun Zhu
(Submitted on 14 Dec 2017)



Few-Shot Learning Through an Information Retrieval Lens
Eleni Triantafillou, Richard Zemel, Raquel Urtasun



Fast Decoding in Sequence Models using Discrete Latent Variables
Łukasz Kaiser, Aurko Roy, Ashish Vaswani, Niki Parmar, Samy Bengio, Jakob Uszkoreit, Noam Shazeer
(Submitted on 9 Mar 2018 (v1), last revised 29 Apr 2018 (this version, v5))


Generative Temporal Models with Memory
Mevlana Gemici, Chia-Chun Hung, Adam Santoro, Greg Wayne, Shakir Mohamed, Danilo J. Rezende, David Amos, Timothy Lillicrap
(Submitted on 15 Feb 2017 (v1), last revised 21 Feb 2017 (this version, v2))


Learning Awareness Models
Brandon Amos, Laurent Dinh, Serkan Cabi, Thomas Rothörl, Sergio Gómez Colmenarejo, Alistair Muldal, Tom Erez, Yuval Tassa, Nando de Freitas, Misha Denil
15 Feb 2018 (modified: 28 Feb 2018)ICLR 2018


World Models ... https://worldmodels.github.io/
David Ha, Jürgen Schmidhuber
(Submitted on 27 Mar 2018 (v1), last revised 9 Apr 2018 (this version, v3))


Memory-based Parameter Adaptation
Pablo Sprechmann, Siddhant M. Jayakumar, Jack W. Rae, Alexander Pritzel, Adrià Puigdomènech Badia, Benigno Uria, Oriol Vinyals, Demis Hassabis, Razvan Pascanu, Charles Blundell
(Submitted on 28 Feb 2018)


Hierarchical Disentangled Representations
Babak Esmaeili, Hao Wu, Sarthak Jain, N. Siddharth, Brooks Paige, Jan-Willem van de Meent
(Submitted on 6 Apr 2018 (v1), last revised 12 Apr 2018 (this version, v2))


Predicting Future Instance Segmentations by Forecasting Convolutional Features
Pauline Luc, Camille Couprie, Yann LeCun, Jakob Verbeek
(Submitted on 30 Mar 2018)

Neural Machine Translation in Linear Time
Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, Koray Kavukcuoglu
(Submitted on 31 Oct 2016 (v1), last revised 15 Mar 2017 (this version, v2))

AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS
Lajanugen Logeswaran & Honglak Lee
University of Michigan, Ann Arbor, MI, USA
Google Brain, Mountain View, CA, USA
{llajan,honglak}@umich.edu,honglak@google.com
2018


Personalization in Goal-Oriented Dialog
Chaitanya K. Joshi, Fei Mi, Boi Faltings
(Submitted on 22 Jun 2017 (v1), last revised 15 Dec 2017 (this version, v3))


Control of Memory, Active Perception, and Action in Minecraft
Junhyuk Oh, Valliappa Chockalingam, Satinder Singh, Honglak Lee
(Submitted on 30 May 2016)


Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets
Joulin & Mikolov


Pointer Networks
Oriol Vinyals
Google Brain
Meire Fortunato
Department of Mathematics, UC Berkeley
Navdeep Jaitly
Google Brain
2017


Learning Efficient Algorithms with Hierarchical Attentive Memory
Marcin Andrychowicz
Karol Kurach
2017


Convolutional Gated Recurrent Networks for Video Segmentation
Mennatullah Siam, Sepehr Valipour, Martin Jagersand, Nilanjan Ray
(Submitted on 16 Nov 2016 (v1), last revised 21 Nov 2016 (this version, v2))


( about  Memory & Hipocampus -> read to be able to understand one-shot learning on hipocampus)
Learning of Chunking Sequences in Cognition and Behavior
Jordi Fonollosa , Emre Neftci  , Mikhail Rabinovich
Published: November 19, 2015https://doi.org/10.1371/journal.pcbi.1004592


Cortex. 2013 Sep;49(8):2001-6. doi: 10.1016/j.cortex.2013.02.012. Epub 2013 Mar 13.
Implicit sequence learning and working memory: correlated or complicated?
Janacsek K1, Nemeth D.


Front Hum Neurosci. 2011; 5: 168.
Published online 2011 Dec 19. doi:  10.3389/fnhum.2011.00168
PMCID: PMC3242327
PMID: 22194719
How Does Hippocampus Contribute to Working Memory Processing?
Marcin Leszczynski1


Convolutional Gated Recurrent Neural Network Incorporating Spatial Features for Audio Tagging
Yong Xu, Qiuqiang Kong, Qiang Huang, Wenwu Wang, Mark D. Plumbley
(Submitted on 24 Feb 2017)


An Overview of Multi-Task Learning in Deep Neural Networks
Sebastian Ruder
(Submitted on 15 Jun 2017)


Personalization in Goal-Oriented Dialog
Chaitanya K. Joshi, Fei Mi, Boi Faltings
(Submitted on 22 Jun 2017 (v1), last revised 15 Dec 2017 (this version, v3))


Convolutional neural random fields for action recognition
Caihua Liu, Jie Liu, Zhicheng He, Yujia Zhai, Qinghua Huc, Yalou Huang


Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis
Chuan Li, Michael Wand
(Submitted on 18 Jan 2016)


Backpropagation through the Void: Optimizing control variates for black-box gradient estimation
Will Grathwohl, Dami Choi, Yuhuai Wu, Geoffrey Roeder, David Duvenaud
(Submitted on 31 Oct 2017 (v1), last revised 23 Feb 2018 (this version, v3))


What is consciousness, and could machines have it?
Stanislas Dehaene1,2,*, Hakwan Lau3,4, Sid Kouider5
Science  27 Oct 2017:
Vol. 358, Issue 6362, pp. 486-492
DOI: 10.1126/science.aan8871


Synthesizing Programs for Images using Reinforced Adversarial Learning
Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S.M. Ali Eslami, Oriol Vinyals
(Submitted on 3 Apr 2018)


Sparse Overcomplete Word Vector Representations
Manaal Faruqui Yulia Tsvetkov Dani Yogatama Chris Dyer Noah A. Smith
Language Technologies Institute
Carnegie Mellon University
Pittsburgh, PA, 15213, USA


Reducing explicit semantic representation vectors using Latent Dirichlet Allocation
Abdulgbar Saif, Mohd Juzaiddin, Ab Aziz, NazliaOmar
https://doi.org/10.1016/j.knosys.2016.03.002


Sparse Binary Polynomial Hashing and the CRM114 Discriminator
William S. Yerazunis, PhD.1
Mitsubishi Electric Research Laboratories2
Cambridge, MA 02139 USA
Version 2003-01-20


Sparse similarity-preserving hashing
Jonathan Masci, Alex M. Bronstein, Michael M. Bronstein, Pablo Sprechmann, Guillermo Sapiro
(Submitted on 19 Dec 2013 (v1), last revised 16 Feb 2014 (this version, v3))


Cognitive Dynamics-Dynamic Cognition?
R Ferber - schoen.waers.de


Vector Symbolic Architectures
http://home.wlu.edu/~levys/vsa.html
https://www.tu-chemnitz.de/etit/proaut/en/research/vsa.html


Jones, M. N., & Mewhort, D. J. K. (2007).
Representing word meaning and order information in a composite holographic lexicon.
Psychological Review, 114, 1-37.
https://www.researchgate.net/publication/6575617_Representing_word_meaning_and_order_information_composite_holographic_lexicon


Eliasmith, C. & P. Thagard. (2001)
Integrating Structure and Meaning: A Distributed Model of Analogical Mapping.
Cognitive Science.
http://watarts.uwaterloo.ca/~celiasmi/Papers/ce.pt.2001.drama.cogsci.html
http://csjarchive.cogsci.rpi.edu/2001v25/i02/p0245p0286/00000048.pdf



Deep unsupervised perceptual grouping.
Greff, Klaus, Rasmus, Antti, Berglund, Mathias, Hao, Tele, Valpola, Harri, and Schmidhuber, Juergen. Tagger
In Advances in Neural Information Processing Systems, pp. 4484–4492, 2016
Tagger: Deep Unsupervised Perceptual Grouping
Klaus Greff, Antti Rasmus, Mathias Berglund, Tele Hotloo Hao, Jürgen Schmidhuber, Harri Valpola
(Submitted on 21 Jun 2016 (v1), last revised 28 Nov 2016 (this version, v2))


Understand the Solomonoff Induction (1964)


Energy-based Generative Adversarial Network
Junbo Zhao, Michael Mathieu, Yann LeCun
(Submitted on 11 Sep 2016 (v1), last revised 6 Mar 2017 (this version, v4))


Calibrating Energy-based Generative Adversarial Networks
Zihang Dai, Amjad Almahairi, Philip Bachman, Eduard Hovy, Aaron Courville
(Submitted on 6 Feb 2017 (v1), last revised 24 Feb 2017 (this version, v2))


iCaRL: Incremental Classifier and Representation Learning
Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
(Submitted on 23 Nov 2016 (v1), last revised 14 Apr 2017 (this version, v2))


The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables
Chris J. Maddison, Andriy Mnih, Yee Whye Teh
(Submitted on 2 Nov 2016 (v1), last revised 5 Mar 2017 (this version, v3))

Object-Oriented Deep Learning
Qianli Liao and Tomaso Poggio
Center for Brains, Minds, and Machines, McGovern Institute for Brain Research,
Massachusetts Institute of Technology, Cambridge, MA, 02139.
October 2017


Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
Chelsea Finn, Pieter Abbeel, Sergey Levine
(Submitted on 9 Mar 2017 (v1), last revised 18 Jul 2017 (this version, v3))


A Hierarchical Recurrent Neural Network for Symbolic Melody Generation
Jian Wu, Changran Hu, Yulong Wang, Xiaolin Hu, Jun Zhu
(Submitted on 14 Dec 2017)


Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory
Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, Bing Liu
(Submitted on 4 Apr 2017 (v1), last revised 14 Sep 2017 (this version, v3))


Consciousness as a State of Matter
Max Tegmark (MIT)
(Submitted on 6 Jan 2014 (v1), last revised 18 Mar 2015 (this version, v3))


Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio
(Submitted on 11 Dec 2014)





Reading tasks for the Consciousness Prior


Goodfellow et al. (2016). Deep Learning Esp. Ch 15.
Bahdanau et al. (2014). Neural Machine Translation by Jointly Learning to Align and Translate
Goodfellow et al. (2014). Generative Adversarial Networks
Brakel and Bengio. (2017). Learning Independent Features with Adversarial Nets for Non-linear ICA
Jaderberg et al. (2016). Reinforcement Learning with Unsupervised Auxiliary Tasks
Grathwohl et al. (2017). Backpropagation Through the Void


The research of constructing dynamic cognition model based on brain network
Fang Chunying,a,b Li Haifeng,a,⁎ and Ma Lina
Author information ► Article notes ► Copyright and License information ► Disclaimer


Learning what to learn in a neural program
Richard Shin, Dawn Song
15 Feb 2018ICLR 2018

MEMORY, LEARNING, AND EMOTION: THE HIPPOCAMPUS
http://psycheducation.org/brain-tours/memory-learning-and-emotion-the-hippocampus/


Behav Neural Biol. 1993 Jul;60(1):9-26.
On the role of the hippocampus in learning and memory in the rat.
https://www.ncbi.nlm.nih.gov/pubmed/8216164

Interplay of hippocampus and prefrontal cortex in memory
Alison R. Preston1 and Howard Eichenbaum2
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3789138/


Associative Learning and the Hippocampus
Wendy A. Suzuki, PhD
http://www.apa.org/science/about/psa/2005/02/suzuki.aspx


Personalization in Goal-oriented Dialog
Poster at NIPS 2017 Conversational AI Workshop
https://chaitjo.github.io/personalization-in-dialog/


ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
Adam Paszke, Abhishek Chaurasia, Sangpil Kim, Eugenio Culurciello
(Submitted on 7 Jun 2016)


Systematic evaluation of CNN advances on the ImageNet
Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas
(Submitted on 7 Jun 2016 (v1), last revised 13 Jun 2016 (this version, v2))f


Overcoming catastrophic forgetting in neural networks
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell
PNAS March 28, 2017. 114 (13) 3521-3526; published ahead of print March 14, 2017. https://doi.org/10.1073/pnas.1611835114


Attention and Augmented Recurrent Neural Networks
https://distill.pub/2016/augmented-rnns/


https://blog.heuritech.com/2016/01/20/attention-mechanism/


Neural Network Architectures
Eugenio Culurciello
https://towardsdatascience.com/neural-network-architectures-156e5bad51ba


Navigating the Unsupervised Learning Landscape
Eugenio Culurciello
May 4, 2017


https://continuousai.com/research/


Universal Sentence Encoder
Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil
(Submitted on 29 Mar 2018 (v1), last revised 12 Apr 2018 (this version, v2))


Can Neural Networks Understand Logical Entailment?
Evans Saxton Amos Kohli & Grtefenstette
ICLR 2018


Neural Module Networks
Jacob Andreas, Marcus Rohrbach, Trevor Darrell, Dan Klein
(Submitted on 9 Nov 2015 (v1), last revised 24 Jul 2017 (this version, v4))


FiLM: Visual Reasoning with a General Conditioning Layer
Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville
(Submitted on 22 Sep 2017 (v1), last revised 18 Dec 2017 (this version, v2))


Learning to Reason: End-to-End Module Networks for Visual Question Answering
Ronghang Hu, Jacob Andreas, Marcus Rohrbach, Trevor Darrell, Kate Saenko
(Submitted on 18 Apr 2017 (v1), last revised 11 Sep 2017 (this version, v3))


Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions
Scott Reed, Yutian Chen, Thomas Paine, Aäron van den Oord, S. M. Ali Eslami, Danilo Rezende, Oriol Vinyals, Nando de Freitas
(Submitted on 27 Oct 2017 (v1), last revised 28 Feb 2018 (this version, v4))


One-Shot Generalization in Deep Generative Models
Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, Daan Wierstra
(Submitted on 16 Mar 2016 (v1), last revised 25 May 2016 (this version, v2))


A Sensorimotor Circuit in Mouse Cortex for Visual Flow Predictions
Marcus Leinweber, Daniel R. Ward, Jan M. Sobczak3, Alexander Attinger, Georg B. Keller4


Vector-based navigation using grid-like representations in artificial agents
Andrea Banino1,2,3,5*, Caswell Barry2,5*, Benigno Uria1, Charles Blundell1, Timothy Lillicrap1, Piotr Mirowski1, Alexander Pritzel1, Martin J. Chadwick1, Thomas Degris1, Joseph Modayil1, Greg Wayne1, Hubert Soyer1, Fabio Viola1, Brian Zhang1, Ross Goroshin1, Neil Rabinowitz1, Razvan Pascanu1, Charlie Beattie1, Stig Petersen1, Amir Sadik1, Stephen Gaffney1, Helen King1,  Koray Kavukcuoglu1, Demis Hassabis1,4, Raia Hadsell1 & Dharshan Kumaran1,3*
https://deepmind.com/blog/grid-cells/


Learned Deformation Stability in Convolutional Neural Networks
Avraham Ruderman, Neil Rabinowitz, Ari S. Morcos, Daniel Zoran
(Submitted on 12 Apr 2018)


Synthesizing Programs for Images using Reinforced Adversarial Learning
Yaroslav Ganin, Tejas Kulkarni, Igor Babuschkin, S.M. Ali Eslami, Oriol Vinyals
(Submitted on 3 Apr 2018)

#  Learning to Learn

Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
Chelsea Finn, Pieter Abbeel, Sergey Levine
(Submitted on 9 Mar 2017 (v1), last revised 18 Jul 2017 (this version, v3))


On First-Order Meta-Learning Algorithms
Alex Nichol, Joshua Achiam, John Schulman
(Submitted on 8 Mar 2018 (v1), last revised 4 Apr 2018 (this version, v2))
https://blog.openai.com/reptile/

Reptile: a Scalable Metalearning Algorithm
Alex Nichol and John Schulman
OpenAI {alex, joschu}@openai.com

Optimization as a Model for Few-Shot Learning
Sachin Ravi, Hugo Larochelle
04 Nov 2016 (modified: 01 Mar 2017)


Zero-Shot Visual Imitation
Deepak Pathak, Parsa Mahmoudieh, Guanghao Luo, Pulkit Agrawal, Dian Chen, Yide Shentu, Evan Shelhamer, Jitendra Malik, Alexei A. Efros, Trevor Darrell
15 Feb 2018 (modified: 20 Apr 2018)


Learning to learn by gradient descent by gradient descent
Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, Nando de Freitas
(Submitted on 14 Jun 2016 (v1), last revised 30 Nov 2016 (this version, v2))
http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/



# Covariance Shift problem:

Understanding covariate shift in model performance
[version 1; referees: 2 approved with reservations]
Georgia McGaughey, W. Patrick Walters, Brian Goldman
https://f1000research.com/articles/5-597/v1#ref-2

https://www.quora.com/What-is-Covariate-shift
http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html
https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/
https://www.youtube.com/watch?v=YzvlBz_nV2k


AI safety via debate
Geoffrey Irving, Paul Christiano, Dario Amodei
(Submitted on 2 May 2018)


Prefrontal cortex as a meta-reinforcement learning system
Jane X. Wang, Zeb Kurth-Nelson, Dharshan Kumaran, Dhruva Tirumala, Hubert Soyer, Joel Z. Leibo, Demis Hassabis & Matthew Botvinick


Dense Associative Memory for Pattern Recognition
Dmitry Krotov, John J Hopfield
(Submitted on 3 Jun 2016 (v1), last revised 27 Sep 2016 (this version, v2))


Learning to Compare: Relation Network for Few-Shot Learning
Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H.S. Torr, Timothy M. Hospedales
(Submitted on 16 Nov 2017 (v1), last revised 27 Mar 2018 (this version, v2))
https://www.arxiv-vanity.com/papers/1711.06025/


Can Active Memory Replace Attention?
Łukasz Kaiser, Samy Bengio
(Submitted on 27 Oct 2016 (v1), last revised 7 Mar 2017 (this version, v2))



Scalable and Sustainable Deep Learning via Randomized Hashing
Ryan Spring, Anshumali Shrivastava
(Submitted on 26 Feb 2016 (v1), last revised 5 Dec 2016 (this version, v2))
https://github.com/RUSH-LAB/LSH_DeepLearning
GOOOD in energy and training cost reduction
Check other algorithm here too: https://github.com/RUSH-LAB/Flash


Order Matters: Sequence to sequence for sets
Oriol Vinyals, Samy Bengio, Manjunath Kudlur
(Submitted on 19 Nov 2015 (v1), last revised 23 Feb 2016 (this version, v4))


Neighbourhood Components Analysis
Jacob Goldberger, Sam Roweis, Geoff Hinton, Ruslan Salakhutdinov
Department of Computer Science, University of Toronto
{jacob,roweis,hinton,rsalakhu}@cs.toronto.edu


Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure
Ruslan Salakhutdinov, Geoff Hinton ; Proceedings of the Eleventh International Conference on Artificial Intelligence and Statistics, PMLR 2:412-419, 2007.



The LAMBADA dataset: Word prediction requiring a broad discourse context
Denis Paperno (1), Germán Kruszewski (1), Angeliki Lazaridou (1), Quan Ngoc Pham (1), Raffaella Bernardi (1), Sandro Pezzelle (1), Marco Baroni (1), Gemma Boleda (1), Raquel Fernández (2) ((1) CIMeC - Center for Mind/Brain Sciences, University of Trento, (2) Institute for Logic, Language & Computation, University of Amsterdam)


Winner's Curse? On Pace, Progress, and Empirical Rigor
D. Sculley, Jasper Snoek, Alex Wiltschko, Ali Rahimi


A Clockwork RNN
Jan Koutník, Klaus Greff, Faustino Gomez, Jürgen Schmidhuber
(Submitted on 14 Feb 2014)


Complex Numbers and AI:

Unitary Evolution Recurrent Neural Networks
Martin Arjovsky Amar Shah Yoshua Bengio


Gated Orthogonal Recurrent Units:
On Learning to Forget
Li Jing 1∗ , Caglar Gulcehre 2∗ , John Peurifoy 1 , Yichen Shen 1 ,
Max Tegmark 1 , Marin Soljačić 1 , Yoshua Bengio

https://en.wikipedia.org/wiki/Wirtinger_derivatives


Mathematical foundations of matrix syntax
Roman Orus, Roger Martin, Juan Uriagereka
(Submitted on 1 Oct 2017)


Mathematical foundations of matrix syntax
Roman Orus, Roger Martin, Juan Uriagereka
(Submitted on 1 Oct 2017)


Quantum Clustering and Gaussian Mixtures
Mahajabin Rahman, Davi Geiger
(Submitted on 29 Dec 2016)

Uncertainty in Deep Learning
(PhD Thesis)
http://mlg.eng.cam.ac.uk/yarin/blog_2248.html


https://www.quantamagazine.org/quantum-theory-rebuilt-from-simple-physical-principles-20170830/

The Holographic Principle: Why Deep Learning Works
https://medium.com/intuitionmachine/the-holographic-principle-and-deep-learning-52c2d6da8d9


A mathematical motivation for complex-valued convolutional networks
Joan Bruna, Soumith Chintala, Yann LeCun, Serkan Piantino, Arthur Szlam, Mark Tygert
(Submitted on 11 Mar 2015 (v1), last revised 12 Dec 2015 (this version, v3))



Associative Long Short-Term Memory
Ivo Danihelka, Greg Wayne, Benigno Uria, Nal Kalchbrenner, Alex Graves
(Submitted on 9 Feb 2016 (v1), last revised 19 May 2016 (this version, v2))


Deep Complex Networks
Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, Joao Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal



Rotational Unit of Memory
Rumen Dangovski, Li Jing, Marin Soljacic
(Submitted on 26 Oct 2017)


Hippocampal mediation of stimulus representation: a computational theory.
Gluck MA1, Myers CE. 1993
https://www.ncbi.nlm.nih.gov/pubmed/8269040


Books:

http://www.psywww.com/intropsych/index.html

Artificial Intelligence: Foundations of Computational Agents,  2nd Edition
David L. Poole and   Alan K. Mackworth
http://artint.info/2e/html/ArtInt2e.html

Bayesian Reinforcement Learning: A Survey

Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, Aviv Tamar
(Submitted on 14 Sep 2016)
https://arxiv.org/abs/1609.04436


Papers:

https://nickbostrom.com/
https://nickbostrom.com/papers/aipolicy.pdf

Disentangled Variational Auto-Encoder for Semi-supervised Learning
Yang Li, Quan Pan, Suhang Wang, Haiyun Peng, Tao Yang, Erik Cambria
(Submitted on 15 Sep 2017)

Artificial Neural Networks Applied to Taxi Destination Prediction
Alexandre de Brébisson, Étienne Simon, Alex Auvolat, Pascal Vincent, Yoshua Bengio
(Submitted on 31 Jul 2015 (v1), last revised 21 Sep 2015 (this version, v2))

Online Learning Rate Adaptation with Hypergradient Descent
Atilim Gunes Baydin, Robert Cornish, David Martinez Rubio, Mark Schmidt, Frank Wood
(Submitted on 14 Mar 2017 (v1), last revised 26 Feb 2018 (this version, v3))

Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates
Leslie N. Smith, Nicholay Topin
(Submitted on 23 Aug 2017 (v1), last revised 13 Dec 2017 (this version, v2))

Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes
Jack W Rae, Jonathan J Hunt, Tim Harley, Ivo Danihelka, Andrew Senior, Greg Wayne, Alex Graves, Timothy P Lillicrap
(Submitted on 27 Oct 2016)

Learning to Learn without Gradient Descent by Gradient Descent
Yutian Chen, Matthew W. Hoffman, Sergio Gomez Colmenarejo, Misha Denil, Timothy P. Lillicrap, Matt Botvinick, Nando de Freitas
(Submitted on 11 Nov 2016 (v1), last revised 12 Jun 2017 (this version, v6))



Solutions that are  More biologically plausible than today's,

AMAZING video here: https://www.youtube.com/watch?v=YUVLgccVi54 If I implement the solution in pytorch this might be a great showcase ;)
Fast Parametric Learning with Activation Memorization
Jack W Rae, Chris Dyer, Peter Dayan, Timothy P Lillicrap
(Submitted on 27 Mar 2018)
https://arxiv.org/abs/1803.10049


Neurons have the right shape for deep learning
https://www.sciencedaily.com/releases/2017/12/171204091150.htm


Random synaptic feedback weights support error backpropagation for deep learning
Timothy P. Lillicrap
, Daniel Cownden
, Douglas B. Tweed
 & Colin J. Akerman

https://www.nature.com/articles/ncomms13276

Psychol Res. 2008 May;72(3):321-30. Epub 2007 Apr 20.
Solving the credit assignment problem: explicit and implicit learning of action sequences with probabilistic outcomes.
Fu WT1, Anderson JR.

https://www.ncbi.nlm.nih.gov/pubmed/17447083

Sparse Bursts Optimize Information Transmission
Richard Naud, Henning Sprekeler

https://www.biorxiv.org/content/early/2017/11/28/143636

Dendritic error backpropagation in deep cortical microcircuits
João Sacramento, Rui Ponte Costa, Yoshua Bengio, Walter Senn
(Submitted on 30 Dec 2017)
https://arxiv.org/abs/1801.00062

Continuous control with deep reinforcement learning
Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra
(Submitted on 9 Sep 2015 (v1), last revised 29 Feb 2016 (this version, v5))
https://arxiv.org/abs/1509.02971


Branching into brains
Adam Shai Matthew E Larkum
https://www.researchgate.net/publication/321539186_Branching_into_brains
https://elifesciences.org/articles/33066

The Persistence and Transience of Memory
Blake A.Richards Paul W.Frankland


There is also THE CODE!!!!
http://linclab.org/code/
Santoro A, Frankland PW, Richards BA, “Memory Transformation Enhances Reinforcement Learning in Dynamic Environments”, Journal of Neuroscience, 36 (48), 12228-12242
http://linclab.org/santoro-a-frankland-pw-richards-ba-memory-transformation-enhances-reinforcement-learning-in-dynamic-environments-journal-of-neuroscience-36-48-12228-12242/


Guerguiev J, Lillicrap TP, Richards BA, “Towards deep learning with segregated dendrites”
http://linclab.org/deep-learning-with-segregated-dendrites/


Richards BA, Frankland PW, “The Persistence and Transience of Memory”, Neuron, 94 (6), 1071-1074
http://linclab.org/richards-ba-frankland-pw-the-persistence-and-transience-of-memory-neuron-94-6-1071-1074/


Yiu AP, Mercaldo V, Yan C, Richards BA, Rashid AJ, Hsiang HL, Pressey J, Mahadevan V, Tran MM, Kushner SA, Woodin MA, Frankland PW and Josselyn SA, “Neurons are recruited to a memory trace based on relative neuronal excitability at the time of training”, Neuron, 83(3), 722-735
http://linclab.org/yiu-ap-mercaldo-v-yan-c-richards-ba-rashid-aj-hsiang-hl-pressey-j-mahadevan-v-tran-mm-kushner-sa-woodin-ma-frankland-pw-and-josselyn-sa-neurons-are-recruited-to-a-memory-trace-based-on/


van Rheede JJ, Richards BA, Akerman CJ, “Sensory-Evoked Spiking Behavior Emerges via an Experience-Dependent Plasticity Mechanism”, Neuron, 87(5), 1050-1062
http://linclab.org/van-rheede-jj-richards-ba-akerman-cj-sensory-evoked-spiking-behavior-emerges-via-an-experience-dependent-plasticity-mechanism-neuron-875-1050-1062/


Richards BA, Xia F, Santoro A, Husse J, Woodin MA, Josselyn SA, Frankland PW, “Patterns across multiple memories are identified over time”, Nature Neuroscience, 17(7), 981-986
http://linclab.org/richards-ba-xia-f-santoro-a-husse-j-woodin-ma-josselyn-sa-frankland-pw-patterns-across-multiple-memories-are-identified-over-time-nature-neuroscience-177-981-986/


Richards BA and Frankland PW, “The conjunctive trace”, Hippocampus, 23(3), 207-212
http://linclab.org/richards-ba-and-frankland-pw-the-conjunctive-trace-hippocampus-233-207-212/


Memory Transformation Enhances Reinforcement Learning in Dynamic Environments
Adam Santoro, Paul W. Frankland and Blake A. Richards
Journal of Neuroscience 30 November 2016, 36 (48) 12228-12242; DOI: https://doi.org/10.1523/JNEUROSCI.0763-16.2016
http://www.jneurosci.org/content/36/48/12228

STDP as presynaptic activity times rate of change of postsynaptic activity
Yoshua Bengio, Thomas Mesnard, Asja Fischer, Saizheng Zhang, Yuhuai Wu
(Submitted on 19 Sep 2015 (v1), last revised 21 Mar 2016 (this version, v2))

Early Inference in Energy-Based Models Approximates Back-Propagation
Yoshua Bengio, Asja Fischer
(Submitted on 9 Oct 2015 (v1), last revised 7 Feb 2016 (this version, v2))

STDP-Compatible Approximation of Backpropagation in an Energy-Based Model
Yoshua Bengio , Thomas Mesnard , Asja Fischer ,Saizheng Zhang and Yuhuai Wu

Dendritic error backpropagation in deep cortical microcircuits
João Sacramento, Rui Ponte Costa, Yoshua Bengio, Walter Senn
(Submitted on 30 Dec 2017)

Equivalence of Equilibrium Propagation and Recurrent Backpropagation
Benjamin Scellier, Yoshua Bengio
(Submitted on 22 Nov 2017)


Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation
Benjamin Scellier, Yoshua Bengio
(Submitted on 16 Feb 2016 (v1), last revised 28 Mar 2017 (this version, v5))

Feedforward Initialization for Fast Inference of Deep Generative Networks is biologically plausible
Yoshua Bengio, Benjamin Scellier, Olexa Bilaniuk, Joao Sacramento, Walter Senn
(Submitted on 6 Jun 2016 (v1), last revised 28 Jun 2016 (this version, v2))

Difference Target Propagation
Dong-Hyun Lee, Saizheng Zhang, Asja Fischer, Yoshua Bengio
(Submitted on 23 Dec 2014 (v1), last revised 25 Nov 2015 (this version, v5))


Tuning of MCMC with Langevin, Hamiltonian, and other stochastic autoregressive proposals
Richard A. Norton, Colin Fox
(Submitted on 3 Oct 2016)


Deep Learning of Representations: Looking Forward
Yoshua Bengio
(Submitted on 2 May 2013 (v1), last revised 7 Jun 2013 (this version, v2))


Disentangling Factors of Variation via Generative Entangling
Guillaume Desjardins, Aaron Courville, Yoshua Bengio
(Submitted on 19 Oct 2012)


Neural Networks with Few Multiplications
Zhouhan Lin, Matthieu Courbariaux, Roland Memisevic, Yoshua Bengio
(Submitted on 11 Oct 2015 (v1), last revised 26 Feb 2016 (this version, v3))



Transform Invariant Auto-encoder
Tadashi Matsuo, Hiroya Fukuhara, Nobutaka Shimada
(Submitted on 12 Sep 2017)



Links:

https://code.facebook.com/posts/1700437286678763/
https://medium.com/onfido-tech/towards-faster-training-and-smaller-generalisation-gaps-in-deep-learning-b9767f68ec23
http://blog.kaggle.com/
https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc?gi=ed620fc2d22
https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e
http://www.fast.ai/


Online Trainings

https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/
http://course.fast.ai/
http://course.fast.ai/part2.html
http://course.fast.ai/lessons/lessons.html
http://data8.org/

Videos:

Word Embeddings https://www.youtube.com/watch?v=25nC0n9ERq4
