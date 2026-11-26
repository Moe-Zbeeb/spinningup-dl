# Key Papers in Deep Learning

```{contents} Table of Contents
:depth: 2
```

What follows is a list of papers in deep learning that are worth knowing: first the twenty most important, then a longer list grouped by topic. It is far from comprehensive, and it leaves out whole fields, but it is a good starting point for someone looking to read the original work behind the ideas in this course. The papers are grouped by topic, and within a topic they are roughly in the order they appeared. Each comes with a line on why it matters.

A good way to use the list is to read a paper right after the part of the course that covers its idea. You will recognize the ideas, and the paper will show you what was hard about them at the time.

## The Top 20

If you only read twenty papers, read these. They were chosen by combining two signals: how often a paper is cited, and whether it started a direction the field still follows. Three of them, ResNet, AlexNet, and U-Net, are among the 25 most cited papers of the 21st century in any field, according to an analysis by *Nature* in 2025. They are listed in the order they appeared. For an overview written by three of the field's founders, start with [Deep Learning](https://doi.org/10.1038/nature14539), LeCun, Bengio, and Hinton, 2015, a review that is itself among the most cited papers of the century.

1. [Learning Representations by Back-Propagating Errors](https://doi.org/10.1038/323533a0), Rumelhart, Hinton, and Williams, 1986. Backpropagation, the algorithm that trains every network.
2. [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Hochreiter and Schmidhuber, 1997. The LSTM, which let networks remember over long sequences.
3. [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791), LeCun, Bottou, Bengio, and Haffner, 1998. LeNet-5 and the convolutional network design.
4. [ImageNet: A Large-Scale Hierarchical Image Database](https://doi.org/10.1109/CVPR.2009.5206848), Deng, Dong, Socher, and others, 2009. The dataset whose benchmark drove the deep learning boom.
5. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), Krizhevsky, Sutskever, and Hinton, 2012. AlexNet, the result that convinced the field.
6. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), Mikolov and others, 2013. Word2Vec, and the idea that words can be learned vectors.
7. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Kingma and Welling, 2013. The variational autoencoder.
8. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), Srivastava and others, 2014. The regularizer used across deep learning.
9. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Goodfellow and others, 2014. Generation as a game between two networks.
10. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau, Cho, and Bengio, 2014. The attention mechanism.
11. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma and Ba, 2014. The optimizer most models are trained with.
12. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), Ioffe and Szegedy, 2015. Normalization inside the network, which made deep models train fast.
13. [Human-Level Control through Deep Reinforcement Learning](https://doi.org/10.1038/nature14236), Mnih and others, 2015. A deep network learning to play Atari games from pixels, which started deep reinforcement learning.
14. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Ronneberger, Fischer, and Brox, 2015. The encoder-decoder behind segmentation and diffusion models.
15. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), Ren and others, 2015. The standard design for object detection.
16. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He and others, 2015. Residual connections, and the most cited paper of the 21st century.
17. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani and others, 2017. The Transformer, the architecture behind today's language models.
18. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Devlin and others, 2018. Pre-training followed by fine-tuning.
19. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), Brown and others, 2020. GPT-3, and what scale makes possible.
20. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Ho, Jain, and Abbeel, 2020. Diffusion models, the basis of modern image generation.

## Foundations of Training

- [Learning Representations by Back-Propagating Errors](https://doi.org/10.1038/323533a0), Rumelhart, Hinton, and Williams, 1986. Popularized backpropagation, the algorithm every network in this course is trained with.
- [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a.html), Glorot and Bengio, 2010. Diagnosed vanishing and exploding signals in deep networks and introduced Xavier initialization.
- [Improving Neural Networks by Preventing Co-adaptation of Feature Detectors](https://arxiv.org/abs/1207.0580), Hinton and others, 2012. Introduced dropout.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), Srivastava and others, 2014. The full study of dropout, and why it works.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma and Ba, 2014. The default optimizer for most of deep learning.
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852), He and others, 2015. Introduced He initialization for ReLU networks.
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), Ioffe and Szegedy, 2015. Made deep networks much easier and faster to train.
- [Layer Normalization](https://arxiv.org/abs/1607.06450), Ba, Kiros, and Hinton, 2016. The normalization used in recurrent networks and Transformers.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740), Micikevicius and others, 2017. Training with 16-bit numbers, now standard for large models.

## Convolutional Networks

- [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791), LeCun, Bottou, Bengio, and Haffner, 1998. LeNet-5, the template for the ConvNets that followed.
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), Krizhevsky, Sutskever, and Hinton, 2012. AlexNet, the result that started the deep learning boom.
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901), Zeiler and Fergus, 2013. Showed what the layers of a ConvNet learn.
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), Simonyan and Zisserman, 2014. VGG: depth from small, uniform $3 \times 3$ convolutions.
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), Szegedy and others, 2014. The inception network and its $1 \times 1$ bottlenecks.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), He and others, 2015. Residual connections, which made networks with hundreds of layers trainable.
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993), Huang and others, 2016. DenseNet: every layer connected to every later one.
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946), Tan and Le, 2019. How to scale depth, width, and resolution together.

## Object Detection and Segmentation

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), Ren and others, 2015. Region proposals computed by the network itself.
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640), Redmon and others, 2015. YOLO: detection in a single pass.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Ronneberger, Fischer, and Brox, 2015. The encoder-decoder with skip connections, now also the backbone of diffusion models.
- [Mask R-CNN](https://arxiv.org/abs/1703.06870), He and others, 2017. Detection and pixel-level segmentation together.

## Sequence Models

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), Hochreiter and Schmidhuber, 1997. The LSTM, and the memory cell that solved vanishing gradients in RNNs.
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Cho and others, 2014. Introduced the GRU and the RNN encoder-decoder.
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), Sutskever, Vinyals, and Le, 2014. Showed that an encoder-decoder LSTM could translate.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau, Cho, and Bengio, 2014. Introduced attention.

## Word Embeddings

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), Mikolov and others, 2013. Word2Vec's skip-gram and CBOW models.
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546), Mikolov and others, 2013. Negative sampling.
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/), Pennington, Socher, and Manning, 2014. Embeddings from word co-occurrence counts.

## Transformers and Language Models

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani and others, 2017. The Transformer, which replaced recurrence with attention.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Devlin and others, 2018. Pre-training a Transformer encoder, then fine-tuning it for many tasks.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford and others, 2019. GPT-2: a larger language model doing tasks it was never trained for.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), Kaplan and others, 2020. How performance improves predictably with model size, data, and compute.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), Brown and others, 2020. GPT-3, and learning a task from a few examples in the prompt.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann and others, 2022. Chinchilla: how to split a compute budget between model size and data.
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155), Ouyang and others, 2022. InstructGPT: aligning a language model with what people ask for.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), Hu and others, 2021. Fine-tuning a large model by training a small number of added parameters.

## Generative Models

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Kingma and Welling, 2013. The variational autoencoder.
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661), Goodfellow and others, 2014. A generator trained against a discriminator.
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), Gatys, Ecker, and Bethge, 2015. Neural style transfer.
- [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803), Dinh, Sohl-Dickstein, and Bengio, 2016. Normalizing flows: invertible networks with exact likelihoods.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Ho, Jain, and Abbeel, 2020. The diffusion model implemented in this course's code.
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), Song and others, 2020. A unified view of diffusion models as differential equations.
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), Rombach and others, 2021. Diffusion in a compressed latent space, the basis of Stable Diffusion.

## Representation Learning

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Hinton, Vinyals, and Dean, 2015. Training a small model to imitate a large one.
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832), Schroff, Kalenichenko, and Philbin, 2015. The triplet loss.
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), Chen and others, 2020. SimCLR: learning image features without labels.
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), Dosovitskiy and others, 2020. The Vision Transformer.
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), Radford and others, 2021. CLIP: images and text in a shared embedding space.
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), He and others, 2021. Self-supervised learning by reconstructing hidden image patches.

## Understanding Deep Learning

- [Intriguing Properties of Neural Networks](https://arxiv.org/abs/1312.6199), Szegedy and others, 2013. The discovery of adversarial examples.
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), Goodfellow, Shlens, and Szegedy, 2014. Why adversarial examples exist, and adversarial training.
- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530), Zhang and others, 2016. Networks can memorize random labels, so why do they generalize?
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635), Frankle and Carbin, 2018. Small subnetworks that train as well as the full network.
- [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292), Nakkiran and others, 2019. Test error can fall, rise, and fall again as models grow.
