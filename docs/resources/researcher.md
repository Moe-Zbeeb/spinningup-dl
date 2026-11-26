# Spinning Up as a Deep Learning Researcher

```{contents} Table of Contents
:depth: 2
```

Deep learning moves fast, and it can look as if research is reserved for people at large labs with enormous budgets. It is not. Many useful contributions come from careful work on small, well-chosen problems. This essay is for students who have worked through the course and want to start doing research: what background to build, how to practice, how to pick a project, and how to do the work so that its results can be trusted.

## The Right Background

You do not need to know everything before you start, but a few foundations make everything else easier.

- **Mathematics.** Linear algebra, calculus, and probability are the language of the field. You should be comfortable with matrix shapes and products, with the chain rule and gradients of functions of vectors, and with expectations, variances, and common distributions. A little optimization helps too. When a paper's equations feel opaque, the gap is almost always in one of these.
- **Programming.** Fluent Python, and fluent PyTorch or a similar library. Fluent means that you can write a training loop from memory, inspect tensor shapes when something breaks, and read other people's research code without getting lost. Comfort with the command line, Git, and running jobs on a remote machine pays off quickly.
- **The core ideas of deep learning.** The five parts of this course: how networks are trained, why they overfit, how to diagnose them, and the main architectures. You should be able to explain, without notes, what each piece of a standard model does and why it is there.
- **The habit of reading papers.** Start with the [Key Papers](keypapers.md) list. Reading a paper well takes several passes: first the abstract, figures, and conclusions, to learn what it claims; then the method, to learn how; and only then the details, if you need to reproduce it.

If one of these is weak, strengthen it alongside the rest rather than waiting until it is perfect. Nothing builds understanding like using it.

## Learn by Doing

The fastest way to learn deep learning is to implement things yourself.

**Write the core models from scratch.** Implement an MLP, a ConvNet, an LSTM, and a Transformer with nothing but basic tensor operations, and train each one on a small dataset. Then compare yours with the implementations in this course's code. Writing a model yourself forces you to understand every shape and every line, and it teaches you the most important skill of all: debugging a network that runs without errors but does not learn.

**Expect silent bugs.** Most bugs in deep learning code do not crash anything. A model with a mistake in it often still trains, just worse than it should. A wrong axis in a softmax, a mask that leaks the future, labels shuffled separately from their inputs: each of these produces a model that learns something, so the bug is easy to miss. Simple habits catch most of them:

- Check the loss at initialization. A classifier with $C$ classes should start with a loss close to $\log C$.
- Train on a single small batch until the loss is nearly zero. A model that cannot memorize 16 examples has a bug.
- Look at the data right before it enters the model, and at the model's outputs, not only at the loss.
- Assert tensor shapes in your code, and use [gradient checking](../intro/improving_networks.md#gradient-checking) when you write a backward pass by hand.

**Reproduce a paper.** Pick a paper with a clear result that is cheap to verify, and try to reproduce its main table or figure. You will discover how many details a paper leaves out, which of them matter, and how sensitive results can be to small choices. When your numbers do not match, finding out why teaches you more than any amount of reading.

**Start small and scale up.** Begin with a tiny model and a small dataset, so that an experiment takes minutes, not days. Get it working, understand it, and only then make it bigger. Fast experiments mean more ideas tested, and more ideas tested means faster learning.

## Developing a Research Project

Once you are comfortable implementing and training models, you can start a project of your own.

**Find a question, not just a method.** A good project starts from something you want to know: why does this model fail on that kind of input, does this trick still help at a larger scale, can this be done with a tenth of the data? A question tells you which experiments to run and when you are done.

**Read the literature around it.** Before investing weeks, search for related work. Someone may have already answered your question, or come close, and their paper will tell you what baselines to compare against. Do not let this discourage you: an idea that others have tried in a different setting is often still open in yours.

**Pick a direction that suits your resources.** You will not train the largest models, but many important questions do not need them. Good directions for a student include:

- **Improving what already exists.** Take a standard method and make it faster, simpler, more robust, or better understood.
- **Understanding.** Explain why something works, or when it fails, through careful experiments. A clear explanation of an existing method is a real contribution.
- **Applications.** Bring deep learning to a problem or a dataset that the field has neglected: a low-resource language, a scientific domain, a local problem that you understand better than anyone else.
- **Evaluation.** Build a benchmark or a test that measures something existing benchmarks miss.

**Avoid two traps.** The first is the idea so ambitious that no single experiment can test it. The second is incremental tuning: trying variation after variation of a model in the hope that one of them scores slightly higher, without a question that explains why it should. Aim for a project where a few weeks of work give a clear answer, whatever the answer turns out to be.

**Work with others.** Find a mentor who can tell you which ideas are worth pursuing, and peers who will read your drafts and ask hard questions. Research is far easier, and far more enjoyable, as a conversation.

## Doing Rigorous Research

The goal of an experiment is to find out what is true, not to produce a good-looking number. A few practices separate results that hold up from results that do not.

**Use strong baselines, and tune them as carefully as your own method.** The most common way to fool yourself is to compare a carefully tuned new method against a carelessly tuned baseline. If your method had a hyperparameter search, the baseline needs one of the same size.

**Run several seeds.** Two runs of the same model with different random seeds can differ by more than the improvement you are claiming. Run at least three, report the mean and the spread, and be suspicious of any gain smaller than the variation between seeds. The [Running Experiments](../user/running.md) page shows how to run a grid of seeds with this course's code.

**Keep the test set untouched.** Make every decision, from the architecture to the stopping point, on the dev set. Evaluate on the test set once, at the end. A test set that has guided your choices no longer measures how the model will do on new data, as [Part 3](../intro/structuring_projects.md) explains.

**Change one thing at a time.** When a new method combines several ideas, run **ablations**: remove each idea in turn and measure what happens. Ablations tell you which parts actually matter, and they often show that a simpler version works as well.

**Look at the outputs.** Numbers summarize; examples explain. Read the model's mistakes, as in [error analysis](../intro/structuring_projects.md#error-analysis), and look at what it gets right for the wrong reasons.

**Record everything.** Save the configuration and the code version of every experiment, so that any result can be traced and reproduced. Months later, you will need to know exactly how a number was produced, and memory is not good enough.

**Report honestly.** Say what did not work as well as what did, report the settings where your method is weaker, and release your code. Negative results, clearly explained, save other people from repeating your mistakes.

## Closing Thoughts

Becoming a researcher is a gradual process. You start by reproducing other people's results, move on to small improvements, and only later find questions of your own that no one has asked. Every step builds the intuition for the next. The most important qualities are not brilliance or resources but persistence, care, and curiosity: the willingness to debug patiently, to check your results honestly, and to keep asking why.

## Other Resources

- [Deep Learning](https://www.deeplearningbook.org/), Goodfellow, Bengio, and Courville. A thorough textbook on the mathematics and ideas, free online.
- [Dive into Deep Learning](https://d2l.ai/), Zhang, Lipton, Li, and Smola. An interactive textbook with runnable code for every chapter.
- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), Andrej Karpathy. A video series that builds neural networks, and then a GPT, from scratch.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), Andrej Karpathy. Practical habits for training networks without fooling yourself.
- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), Google Research. A systematic guide to choosing and tuning hyperparameters.
- [Practical Deep Learning for Coders](https://course.fast.ai/), fast.ai. A hands-on course that starts from working applications.
- [Distill](https://distill.pub/). Clear, interactive explanations of machine learning research.
- [arXiv](https://arxiv.org/). Where almost every new deep learning paper appears first.
