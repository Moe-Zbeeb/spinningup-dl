# Part 3: Structuring Machine Learning Projects

```{contents} Table of Contents
:depth: 2
```

Welcome to Part 3 of our introduction to deep learning! [Part 2](improving_networks.md) gave you a toolbox of ideas for improving a model: more data, regularization, a bigger network, a better optimizer. A real project never has time to try them all, and a wrong guess can cost months. This part is about deciding which idea to try next.[^sefidian] In this part, we aim to acquaint you with

- orthogonalization, the habit of using one tool for one problem,
- how to set a goal: a single evaluation metric and the right dev and test sets,
- how to compare your model with human-level performance,
- error analysis, which tells you which mistakes are worth fixing,
- what to do when your training data comes from a different distribution than your test data,
- and three ways to structure a learning problem: transfer learning, multi-task learning, and end-to-end learning.

## Orthogonalization

Think of how you drive a car. The steering wheel changes your direction, the accelerator your speed, and the brake slows you down. Each control does one thing, so you can correct one problem without creating another. Now imagine a single joystick where pushing forward also turned slightly to the left. You could still drive, but every correction would need a second correction.

**Orthogonalization** is the same idea applied to machine learning: arrange your tools so that each one fixes one problem and leaves the others alone. To succeed, a supervised learning system has to do four things, in order, and each has its own set of tools.

| Goal | Tools that address it |
| --- | --- |
| Fit the training set well on the cost function | A bigger network, a different architecture, a better optimizer such as Adam, training longer |
| Fit the dev set well | Regularization, a bigger training set |
| Fit the test set well | A bigger dev set |
| Perform well in the real world | A different dev set, a different cost function |

When something goes wrong, first find which of the four goals is failing, then reach for the tools in that row. If the training error is already low but the dev error is high, a bigger network will not help; regularization might.

A tool that affects several goals at once is harder to use. [Early stopping](improving_networks.md#early-stopping) is the classic example: stopping earlier makes the fit on the training set worse and the fit on the dev set better at the same time. It is not wrong to use it, but it couples two goals that you would rather adjust separately.

## Single Number Evaluation Metric

Progress in machine learning comes from an empirical loop: have an idea, implement it, run an experiment, look at the result, and have the next idea. The loop runs much faster when one number tells you whether the new model is better than the old one.

For a classifier, two natural numbers often pull in different directions. Suppose the task is to recognize cats.

- **Precision** is the fraction of images the classifier calls cats that really are cats:

$$
P = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}.
$$

- **Recall** is the fraction of real cat images that the classifier finds:

$$
R = \frac{\text{true positives}}{\text{true positives} + \text{false negatives}}.
$$

Here a **true positive** is a cat correctly called a cat, a **false positive** is a non-cat wrongly called a cat, and a **false negative** is a cat the classifier missed. A classifier that labels everything a cat has perfect recall and poor precision. One that only labels the single most obvious cat has perfect precision and poor recall. Usually there is a tradeoff between the two.

With two numbers, it is not clear which classifier is better:

| Classifier | Precision | Recall | Average | $F_1$ score |
| --- | ---: | ---: | ---: | ---: |
| A | 95% | 90% | 92.5% | 92.4% |
| B | 98% | 85% | 91.5% | 91.0% |

The **$F_1$ score** combines them into one number. It is the **harmonic mean** of precision and recall:

$$
F_1 = \frac{2}{\frac{1}{P} + \frac{1}{R}}.
$$

By $F_1$, classifier A is better.

Why the harmonic mean rather than the ordinary average? The harmonic mean is dragged down by whichever number is smaller. A classifier with $P = 100\%$ and $R = 2\%$ is useless, since it misses 98% of the cats, but its ordinary average is a respectable-looking 51%. Its $F_1$ score is 3.9%, which describes it much better. A high $F_1$ requires both numbers to be high.

The same principle applies beyond precision and recall. If you track the error separately for users in four countries, also compute their average, so that you can rank models at a glance.

**Example: Computing Precision, Recall, and $F_1$.**

```python
def precision_recall_f1(y_true, y_pred):
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 / (1 / precision + 1 / recall)
    return precision, recall, f1


y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
print("P = %.2f, R = %.2f, F1 = %.2f" % precision_recall_f1(y_true, y_pred))
```

```text
P = 0.60, R = 0.75, F1 = 0.67
```

The classifier found 3 of the 4 cats, so $R = 0.75$, but 2 of its 5 cat predictions were wrong, so $P = 0.6$.

## Satisficing and Optimizing Metrics

Sometimes you care about several things that cannot sensibly be folded into one formula, such as accuracy and running time. Adding them together with weights, say accuracy minus $0.5 \times$ running time, is artificial. A cleaner way is to split them into two kinds.

- An **optimizing metric** is the one you want to be as good as possible, for example accuracy.
- A **satisficing metric** only has to be good enough, for example a running time of at most 100 milliseconds. Beyond that threshold, you do not care how much faster the model is.

With $N$ metrics, choose one to optimize and make the other $N - 1$ satisficing. Then the best model is the one with the best optimizing metric among those that meet every threshold.

| Model | Accuracy | Running time |
| --- | ---: | ---: |
| A | 90% | 80 ms |
| B | 92% | 95 ms |
| C | 95% | 1500 ms |

C is the most accurate, but it fails the 100 ms threshold, so B is the best acceptable model.

Another example is a voice assistant that listens for a wake word. You want to maximize the chance it wakes up when someone says the word, subject to waking up by mistake at most once every 24 hours.

**Example: Choosing the Best Acceptable Model.**

```python
models = {
    "A": {"accuracy": 0.90, "runtime_ms": 80},
    "B": {"accuracy": 0.92, "runtime_ms": 95},
    "C": {"accuracy": 0.95, "runtime_ms": 1500},
}

acceptable = {name: m for name, m in models.items() if m["runtime_ms"] <= 100}
best = max(acceptable, key=lambda name: acceptable[name]["accuracy"])
print(best)
```

```text
B
```

## Training, Dev, and Test Set Distributions

The dev set and the evaluation metric together define the target your team aims at. Every experiment is judged by them, so if they point at the wrong place, the whole team optimizes toward the wrong place, however good its work.

### Choosing the Dev and Test Sets

Choose a dev set and a test set that reflect the data you expect to see in the future and care about doing well on. And take both from the same distribution. Suppose you build a cat classifier for users in eight regions and use four regions for the dev set and the other four for the test set. Your team could spend months improving performance on the dev regions, only to discover that the test regions look different. Instead, shuffle all eight regions together and split them randomly, so that hitting the dev target also hits the test target.

The sizes follow the guidelines from [Part 2](improving_networks.md#training-dev-cross-validation-and-test-sets): the old 70/30 or 60/20/20 splits suit small datasets, and with a million examples a 98/1/1 split is common. The test set only needs to be big enough to give high confidence in the final performance. Working without a test set is sometimes acceptable, though not recommended.

Mismatched distributions are common in deep learning, because deep networks need a lot of labeled data and the easiest data to collect is often not the data you care about. [Training and Testing on Different Distributions](#training-and-testing-on-different-distributions) returns to this.

### When to Change the Metric

A metric should rank models the way you would rank them. Suppose two cat classifiers have errors of 3% and 5%, but the 3% one occasionally lets through explicit images, labeling them as cats, while the 5% one never does. Your users would clearly prefer the second, yet plain classification error prefers the first. The metric is wrong.

One fix is to weight the errors: count a mistake on an explicit image, say, 10 times as heavily as an ordinary mistake. The **weighted error** is

$$
\text{Error} = \frac{\sum_{i} w^{(i)} \, \mathbb{1}\left\{\hat{y}^{(i)} \neq y^{(i)}\right\}}{\sum_{i} w^{(i)}},
$$

where $w^{(i)} = 10$ for explicit images and 1 otherwise, and $\mathbb{1}\{\cdot\}$ is 1 when the condition holds and 0 otherwise.

**Example: A Weighted Error.** Ten predictions with two mistakes, one of them on an explicit image:

```python
import torch

wrong = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
is_explicit = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

weights = 1 + 9 * is_explicit
plain_error = wrong.mean()
weighted_error = (weights * wrong).sum() / weights.sum()
print(f"plain error {plain_error:.2f}, weighted error {weighted_error:.2f}")
```

```text
plain error 0.20, weighted error 0.58
```

This is orthogonalization again. Defining the right target, the metric, is one job. Hitting that target, by training a model that scores well on it, is a separate job. Do them one at a time.

Change the dev set or the metric as soon as doing well on them stops meaning doing well on your application. If your dev set is full of professional photographs but your users upload blurry phone pictures, a model that wins on the dev set may lose in practice. Change the dev set to look like the users' pictures. Do not keep running for long with a dev set or metric you know is wrong; it is better to set up a rough target quickly and correct it later than to have no target at all.

## Comparing to Human-Level Performance

It is natural to compare machine learning systems with people, and the comparison turns out to be a useful tool.

```{figure} ../_static/img/structuring_projects/human_level.svg
:alt: A curve of model accuracy rising over the time spent on a project. It climbs quickly to a dashed line marking human-level performance, then slows down and creeps toward a higher dotted line marking the best possible accuracy.
:width: 80%

A typical history of a project, drawn schematically. Accuracy rises quickly until it reaches human-level performance, then slows down as it approaches the best possible accuracy.
```

No model can do better than the **Bayes optimal error**, or **Bayes error**: the lowest error any function could achieve on the task. It is usually above zero, because some inputs are genuinely ambiguous, such as an audio clip too noisy for anyone to transcribe, or an image too blurry to classify.

Progress tends to slow down once a model passes human-level performance, for two reasons. First, for many tasks people are already close to the Bayes error, so there is little room left. Second, as long as the model is worse than people, you have tools that stop working once it is better:

- you can ask people to label more data,
- you can look at an example the model got wrong, ask why a person got it right, and gain insight from that manual error analysis,
- and you can do a better analysis of bias and variance, as described next.

### Avoidable Bias

In [Part 2](improving_networks.md#bias-and-variance), we judged the training error against a base error. For tasks that people do well, such as recognizing images or speech, human-level error is a good estimate of the Bayes error. The gap between it and the training error is the **avoidable bias**, the part of the bias you could still remove. The gap between the training error and the dev error is the variance.

| | Scenario A | Scenario B |
| --- | ---: | ---: |
| Human-level error | 2% | 8.5% |
| Training error | 9% | 9% |
| Dev error | 11% | 11% |
| Avoidable bias | 7% | 0.5% |
| Variance | 2% | 2% |

The model's errors are identical in both scenarios, but the right next step is not. In scenario A, the training error is far above the 2% people achieve, so the avoidable bias of 7% is the bigger problem: train a bigger network or train longer. In scenario B, people make 8.5% errors themselves, so the model is already close to the best possible on the training set. The variance of 2% is the bigger problem: try regularization or a bigger training set.

### Defining Human-Level Error

"Human-level" can mean different things. On a medical imaging task, the error might be

| Who | Error |
| --- | ---: |
| A typical person | 3% |
| A typical doctor | 1% |
| An experienced doctor | 0.7% |
| A team of experienced doctors | 0.5% |

When the purpose is to estimate the Bayes error, use the best human performance, here 0.5%: the Bayes error can be no higher than that. The choice matters most late in a project, when the avoidable bias and the variance are both small and you need a precise reference to tell which is bigger. Early on, when the training error is 5%, the precise choice makes no difference.

### Surpassing Human-Level Performance

Once a model beats the best human performance, you lose your estimate of the Bayes error, so you can no longer tell how much avoidable bias is left. Progress becomes harder, though not impossible.

Machine learning already far surpasses people on many problems with **structured data**, data that comes as tables and databases rather than as images or sound, such as online advertising, product recommendations, logistics, and loan approval. A model sees far more examples than any person could. On **natural perception** tasks, which people are very good at, surpassing them is harder, but it has happened for some speech recognition, some image recognition, and some medical tasks such as reading ECGs and spotting certain cancers.

### Improving Your Model

Putting it all together, compare the three errors and work on the biggest gap.

- **To reduce avoidable bias:** train a bigger model, train longer or with a better optimizer such as momentum, RMSprop, or Adam, or search for a better architecture and better hyperparameters.
- **To reduce variance:** get more data, add regularization such as L2, dropout, or data augmentation, or search for a better architecture and better hyperparameters.

## Error Analysis

Suppose your cat classifier has 10% dev error, and a teammate suggests making it better on dogs, since some dogs get mistaken for cats. That could take months. Before starting, spend an hour on **error analysis**: look at the mistakes yourself and count what kinds they are.

1. Take a sample of about 100 misclassified dev examples.
2. Look at each one and note which categories it falls into, in a spreadsheet with one column per idea: dogs, great cats such as lions, blurry images, and so on. An example can fall into several categories.
3. Add a column for anything unexpected you notice, and turn it into a new category if it keeps coming up.
4. Count the percentage of mistakes in each category.

```{figure} ../_static/img/structuring_projects/error_analysis.svg
:alt: A horizontal bar chart of four error categories among 100 misclassified dev examples. Dogs are 8%, great cats 43%, blurry images 61%, and incorrectly labeled examples 6%. Each bar is labeled with the dev error that would remain if every mistake in that category were fixed.
:width: 90%

An error analysis of 100 misclassified examples, with the dev error that would remain if every mistake in a category were fixed.
```

The percentages put a ceiling on what each idea can achieve. If only 8% of the mistakes are dogs, then even a perfect fix for dogs only lowers the dev error from 10% to 9.2%. Fixing blurry images, which make up 61% of the mistakes, could lower it to 3.9%. An hour of counting can save months of work on the wrong idea. If several categories look promising, different people on the team can take one each.

**Example: Tallying an Error Analysis.** Once the examples are tagged, counting is easy:

```python
from collections import Counter

reviewed = [
    {"dog"}, {"blurry"}, {"great cat", "blurry"}, {"great cat"}, {"blurry", "mislabeled"},
    {"great cat"}, {"blurry"}, {"great cat", "blurry"}, {"dog", "blurry"}, {"great cat"},
]

counts = Counter(tag for tags in reviewed for tag in tags)
for tag, n in counts.most_common():
    print(f"{tag:<11}{100 * n / len(reviewed):.0f}%")
```

```text
blurry     60%
great cat  50%
dog        20%
mislabeled 10%
```

### Incorrectly Labeled Examples

Sometimes the label itself is wrong: a person labeled a dog image "cat".

In the training set, deep learning is quite robust to random labeling mistakes. If a few percent of the labels are wrong at random, it is usually not worth fixing them by hand. It is much less robust to **systematic** mistakes, such as a labeler who calls every white dog a cat, because the model will learn that rule.

In the dev and test sets, add a column for incorrect labels to your error analysis and treat it like any other category. If 6% of the mistakes come from wrong labels and the dev error is 10%, wrong labels account for 0.6% of it, and there are probably better things to work on. But if the dev error falls to 2%, the same 0.6% is now 30% of the mistakes, and the dev set can no longer reliably tell two models apart. Then it is worth fixing.

```{admonition} You Should Know
When you fix labels in the dev set, follow three rules.

- Apply the same process to the test set, so that the two still come from the same distribution.
- Look at the examples the model got right as well as the ones it got wrong. Otherwise you only fix mistakes that hurt the model, and your estimate of its error becomes biased in its favor.
- The training set does not need the same treatment. It may end up from a slightly different distribution than the dev and test sets, and that is usually fine.
```

## Build Your First System Quickly, Then Iterate

For a new application, do not spend weeks designing the perfect system. Instead:

1. Set up a dev set, a test set, and a metric, so that you have a target.
2. Build a first system quickly, even a crude one.
3. Use bias and variance analysis and error analysis to decide what to work on next.

The reason is that you cannot know in advance which problems will matter. A speech recognizer might struggle with noisy backgrounds, accented speech, far-away speakers, or children's voices, and each could be a project of its own. Only a working system and an error analysis tell you which ones actually cost you accuracy. The first system's job is to show you where to go. This advice matters less when the problem is already well studied, so that the research literature tells you what works.

## Training and Testing on Different Distributions

Deep learning needs a lot of training data, so teams often train on whatever data they can get, even when it differs from the data the system will face.

Suppose you are building a cat classifier for a mobile app. You have 200,000 high-resolution cat images collected from web pages, distribution A, and only 10,000 lower-resolution images uploaded through the app, distribution B. The app images are what the system must classify well. How should you split the data?

```{figure} ../_static/img/structuring_projects/data_split.svg
:alt: Two ways of splitting 200 thousand web images and 10 thousand app images into a training set of 205 thousand, a dev set of 2.5 thousand, and a test set of 2.5 thousand. In option 1, every set is mostly web images. In option 2, the dev and test sets are entirely app images, and the training set holds the web images plus 5 thousand app images.
:width: 100%

Two ways to split the data. Option 2 aims the dev and test sets at the distribution you care about.
```

- **Option 1: shuffle everything together** and split it 205,000 / 2,500 / 2,500. All three sets then come from the same mixture, but only about 5% of the mixture is app images. The dev set has only about 120 app images, so your team spends almost all its effort optimizing for web images. This aims at the wrong target.
- **Option 2: build the dev and test sets from the target distribution.** Put 2,500 app images in the dev set and 2,500 in the test set, and train on the 200,000 web images plus the remaining 5,000 app images. The training set now differs from the dev set, but the target is the right one. This is the better choice.

### The Training-Dev Set

Option 2 creates a new difficulty. Suppose the training error is 1% and the dev error is 10%. With matching distributions, that would be a clear case of high variance. But now two things change between the training set and the dev set: the model has not seen the dev examples, and the dev examples come from a different distribution. Either could cause the jump.

To tell them apart, hold out a **training-dev set**: a random slice of the training data that has the same distribution as the training set but is not used for training. Now there are four errors to compare.

| | Scenario A | Scenario B | Scenario C | Scenario D |
| --- | ---: | ---: | ---: | ---: |
| Human-level error | 0% | 0% | 0% | 0% |
| Training error | 1% | 1% | 10% | 10% |
| Training-dev error | 9% | 1.5% | 11% | 11% |
| Dev error | 10% | 10% | 12% | 20% |
| Problem | Variance | Data mismatch | Avoidable bias | Bias and mismatch |

- In **scenario A**, the error jumps between the training set and the training-dev set, which come from the same distribution. The model fails on unseen data even from the distribution it was trained on: a **variance** problem.
- In **scenario B**, the model does fine on unseen data from the training distribution but badly on the dev distribution. This is a **data mismatch** problem.
- In **scenario C**, the training error itself is far above human level: an **avoidable bias** problem.
- In **scenario D**, there is both avoidable bias and a large data mismatch.

The gaps between consecutive errors each name a different problem:

```{figure} ../_static/img/structuring_projects/error_gaps.svg
:alt: Five bars for human-level error 4 percent, training error 7 percent, training-dev error 10 percent, dev error 12 percent, and test error 12.5 percent. Arrows between consecutive bars are labeled avoidable bias, variance, data mismatch, and overfitting to the dev set.
:width: 90%

Each gap between consecutive errors points to a different problem, and so to a different set of tools.
```

| Gap | Problem |
| --- | --- |
| Human-level error to training error | Avoidable bias |
| Training error to training-dev error | Variance |
| Training-dev error to dev error | Data mismatch |
| Dev error to test error | Overfitting to the dev set |

A large gap between the dev error and the test error means you have tuned so much on the dev set that you have started to overfit it. The cure is a bigger dev set. Never make development decisions based on the test set, or the test error stops being an unbiased estimate.

**Example: Building a Training-Dev Set.** This builds the option 2 split, then holds out 5,000 training examples as a training-dev set:

```python
import torch
from torch.utils.data import ConcatDataset, TensorDataset, random_split

web = TensorDataset(torch.randn(200_000, 8), torch.randint(0, 2, (200_000,)))
app = TensorDataset(torch.randn(10_000, 8), torch.randint(0, 2, (10_000,)))
g = torch.Generator().manual_seed(0)

app_train, dev, test = random_split(app, [5_000, 2_500, 2_500], generator=g)
pool = ConcatDataset([web, app_train])
train, train_dev = random_split(pool, [200_000, 5_000], generator=g)
print(len(train), len(train_dev), len(dev), len(test))
```

```text
200000 5000 2500 2500
```

### Addressing Data Mismatch

There is no systematic cure for data mismatch, but two steps usually help.

1. Do a manual error analysis to understand how the training data differs from the dev data. You might find that the dev set of a car's voice assistant has a lot of road noise in the background.
2. Make the training data more like the dev data, or collect more data like the dev data. One way to make it more similar is **artificial data synthesis**: for example, record road noise and add it to clean speech recordings, producing noisy, in-car-sounding training data.

```{admonition} You Should Know
Synthesized data can fool you. Suppose you have 10,000 hours of clean speech but only one hour of road noise, and you loop that hour over all of it. To your ears the result sounds fine, but the network may overfit to that particular hour of noise, a tiny subset of all the road noise it will meet. Make the synthesized data as varied as the real data, for example by recording many hours of noise.
```

## Transfer Learning

**Transfer learning** takes a network trained on one task and reuses it for another. Suppose you have trained an image recognizer on a million everyday photos, task A, and now want to diagnose X-rays, task B, from only a thousand labeled examples. The knowledge the first network gained about images, how to detect edges, curves, and textures, is useful for X-rays too.

```{figure} ../_static/img/structuring_projects/transfer_learning.svg
:alt: A network with four layers pre-trained on task A, inside a dashed box, followed by a new output layer drawn in a different color that is trained on task B.
:width: 90%

Transfer learning keeps the layers learned on task A and replaces the output layer with a new one for task B.
```

To transfer, delete the network's output layer and replace it with a new output layer, or several new layers, with randomly initialized weights for task B. Then train on the new data. How much of the network you retrain depends on how much data you have.

- With little data for task B, retrain only the new layers, and keep the rest **frozen**.
- With enough data, retrain the whole network. Then the original training on task A is called **pre-training**, and the training on task B is called **fine-tuning**.

Transfer learning from task A to task B makes sense when

- tasks A and B have the same kind of input $x$, for example both take images,
- you have a lot more data for task A than for task B,
- and low-level features learned on A could help with B.

The second condition matters most. One example of task A is worth less than one example of task B, so transfer pays off when A has far more of them.

**Example: Transfer Learning in PyTorch.** Here `pretrained` stands for a network trained on task A; in practice you would load one, for example from `torchvision.models`. Freezing its parameters and replacing its output layer leaves only the new layer to train:

```python
import torch
import torch.nn as nn

pretrained = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 1000),
)

for p in pretrained.parameters():
    p.requires_grad = False
pretrained[-1] = nn.Linear(128, 3)

optimizer = torch.optim.Adam([p for p in pretrained.parameters() if p.requires_grad], lr=1e-3)
trainable = sum(p.numel() for p in pretrained.parameters() if p.requires_grad)
print("trainable parameters:", trainable)
```

```text
trainable parameters: 387
```

The new layer, with $128 \times 3$ weights and 3 biases, is the only part that trains. To fine-tune the whole network instead, skip the freezing loop.

## Multi-Task Learning

In transfer learning, you learn task A first and then task B. In **multi-task learning**, one network learns several tasks at the same time. A self-driving car, for example, needs to know whether each image contains a pedestrian, a car, a stop sign, and a traffic light.

```{figure} ../_static/img/structuring_projects/multi_task.svg
:alt: A network whose shared hidden layers, inside a dashed box, feed four sigmoid output units labeled pedestrian, car, stop sign, and traffic light.
:width: 90%

One network with four outputs. The hidden layers are shared by all four tasks.
```

The label $y^{(i)}$ is now a vector with one entry per task, 1 if that object is in the image and 0 if not, and the output layer has one sigmoid unit per task. The cost sums the logistic loss over all $c$ tasks:

$$
J = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{c} \L\left(\hat{y}^{(i)}_j, y^{(i)}_j\right),
$$

where $\L$ is the logistic loss from Part 1. Unlike [softmax](improving_networks.md#multi-class-classification), which picks exactly one class, multi-task learning lets one image have several labels: an image can contain both a car and a stop sign.

Some labels may be unknown: a labeler might have marked pedestrians and cars but not traffic lights. Multi-task learning handles this naturally. Just leave the unknown entries out of the inner sum.

Multi-task learning makes sense when

- the tasks could benefit from shared low-level features, such as edges and shapes for all four road objects,
- the amount of data for each task is usually similar, so that the other tasks together provide much more data than any one task alone,
- and you can train a network big enough to do well on all the tasks. A network that is too small does worse on multi-task learning than separate networks would.

In practice, transfer learning is used much more often than multi-task learning. Computer vision, and object detection in particular, is the main exception.

**Example: A Multi-Task Loss with Missing Labels.** Unknown labels are marked with `nan` and masked out of the cost:

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, -1.0, 0.5, -3.0],
                       [-0.5, 1.5, -2.0, 0.0]])
labels = torch.tensor([[1.0, 0.0, float("nan"), 0.0],
                       [0.0, 1.0, 0.0, float("nan")]])

known = ~torch.isnan(labels)
targets = torch.nan_to_num(labels)
losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
cost = (losses * known).sum() / labels.shape[0]
print(f"cost {cost:.4f} from {int(known.sum())} known labels")
```

```text
cost 0.6456 from 6 known labels
```

## End-to-End Learning

Many systems used to be built as **pipelines** of hand-designed stages. A speech recognizer, for example, would turn audio into hand-designed features, the features into phonemes, the basic units of sound, the phonemes into words, and the words into a transcript. **End-to-end learning** replaces the whole pipeline with one neural network that maps the input directly to the output.

```{figure} ../_static/img/structuring_projects/end_to_end.svg
:alt: Top, a pipeline in which audio goes through hand-designed features, then phonemes, then words, then the transcript. Bottom, one neural network maps audio directly to the transcript.
:width: 100%

A pipeline of hand-designed stages (top) and an end-to-end network (bottom) for speech recognition.
```

Which works better depends mostly on how much data you have. With a small dataset, the traditional pipeline usually works better, because each stage builds in human knowledge the data cannot supply. With a very large dataset, an end-to-end network usually wins.

The pros of end-to-end learning:

- **It lets the data speak.** The network learns whatever representation works best, instead of being forced through human ideas such as phonemes, which may not be the most useful units for a computer.
- **It needs less hand-design.** There are fewer components to build and maintain.

The cons:

- **It may need a large amount of data**, pairs of $x$ and $y$ for the whole mapping, which are often much harder to collect than data for the individual stages.
- **It excludes potentially useful hand-designed components.** When data is limited, carefully designed components are how human knowledge gets into the system.

### Choosing What to Learn

The key question is: do you have enough data to learn a function as complex as the mapping from $x$ to $y$?

If not, you can still use deep learning for the individual stages. Consider a turnstile that recognizes employees from a camera. Going straight from the camera image to the person's identity is hard, because the person can stand anywhere in the frame and there are few such images of each employee. Splitting it into two steps works better: first find and crop the face, then compare the cropped face with the employees' photos. Each step is a simpler problem with plenty of data available, and each can be a neural network. Choose the mappings $x \to y$ you learn according to the tasks you can get data for.

## Recap

In this part, you learned how to steer a machine learning project. You set a target with a single evaluation metric, satisficing metrics for the constraints, and dev and test sets that reflect the data you care about, and you change that target when it stops matching your application. You compare the training error with human-level error to separate avoidable bias from variance, and use a training-dev set to separate variance from data mismatch. You use error analysis to find out which mistakes are worth fixing before you spend time on them. Finally, you saw three ways to structure a learning problem: transfer learning, multi-task learning, and end-to-end learning. Part 4 turns to convolutional neural networks, the architecture behind modern computer vision.

[^sefidian]: The topics and their order follow Amir Masoud Sefidian, [Coursera Deep Learning Specialization Notes: Structuring Machine Learning Projects](https://www.sefidian.com/wp-content/uploads/2023/03/Structuring-Machine-Learning-Projects.pdf), 2023, which summarizes the third course of Andrew Ng's Deep Learning Specialization.
