---
title: "99.46% valid accurary digit recognition CNN with resnet architecture from scratch"
date: 2019-05-27
permalink: /mnist-resnet/
tags: [fastai, pytorch, cnn, resnet, mnist]
excerpt: "Building resnet architecture on top of CNN"
mathjax: "true"
---

## MNIST CNN

In the previous blogpost, we talked a lot about convolutions and convolution neural network (CNN), this blogpost we will try to build a ResNet architecture on top of the CNN to create a fairly modern deep learning architecture from scratch, but not exactly from scratch because we do not want to waste time re-implementing what we already know, instead we would just rely on pre-existing PyTorch models.


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
# this line creates data & models folder and functionalities integration (e.g. untar_data, model.save)  
  !curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    Done.



```python
from fastai.vision import *
```

### Data


```python
path = untar_data(URLs.MNIST)
```


```python
path.ls()
```




    [PosixPath('/content/data/mnist_png/training'),
     PosixPath('/content/data/mnist_png/testing'),
     PosixPath('/content/data/mnist_png/models')]



As usual, we want to put our data into an item list, so that we can do any processing upon it before turning it into a `databunch`, since MNIST is an image dataset, and the source we download it from already put the train and test data in its respective folder, we can just call `ImageList.from_folder`, we will use **Pillow (PIL)** `convert_mode = 'L'` since the image is grayscale, to learn how convert mode in **PIL** work with fastai, click [here](https://docs.fast.ai/vision.image.html).


```python
il = ImageList.from_folder(path, convert_mode='L')
```


```python
il
```




    ImageList (70000 items)
    Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28)
    Path: /content/data/mnist_png




```python
il.items[0]
```




    PosixPath('/content/data/mnist_png/training/6/51370.png')




```python
il[0].show()
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_12_0.png" alt="">


We want to set the default colourmap to `binary`, because our image is in grayscale, so it would make sense for us to display them in grayscale.


```python
defaults.cmap='binary'
```


```python
il[0].show()
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_15_0.png" alt="">


Because our item list consist of all the images from `path`, we want to split it into train and valid set, note that even though the folder name is `testing`, it does have label, and so we should put it as a validation set, we follow the parlance that Kaggle use for train, valid, and test set.


```python
sd = il.split_by_folder(train='training', valid='testing')
```


```python
sd
```




    ItemLists;

    Train: ImageList (60000 items)
    Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28)
    Path: /content/data/mnist_png;

    Valid: ImageList (10000 items)
    Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28)
    Path: /content/data/mnist_png;

    Test: None



Notice that our data in the `ItemLists` do not yet have labels, conveniently, we know that the name of the file represents the label as shown below.


```python
(path/'training').ls()
```




    [PosixPath('/content/data/mnist_png/training/6'),
     PosixPath('/content/data/mnist_png/training/0'),
     PosixPath('/content/data/mnist_png/training/3'),
     PosixPath('/content/data/mnist_png/training/5'),
     PosixPath('/content/data/mnist_png/training/9'),
     PosixPath('/content/data/mnist_png/training/7'),
     PosixPath('/content/data/mnist_png/training/4'),
     PosixPath('/content/data/mnist_png/training/8'),
     PosixPath('/content/data/mnist_png/training/2'),
     PosixPath('/content/data/mnist_png/training/1')]



So we can just auto generate the labels from the folder names by calling `sd.label_from_folder()`, our `ItemLists` is now `LabelLists`.


```python
ll = sd.label_from_folder()
```


```python
ll
```




    LabelLists;

    Train: LabelList (60000 items)
    x: ImageList
    Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28)
    y: CategoryList
    6,6,6,6,6
    Path: /content/data/mnist_png;

    Valid: LabelList (10000 items)
    x: ImageList
    Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28)
    y: CategoryList
    6,6,6,6,6
    Path: /content/data/mnist_png;

    Test: None




```python
ll.train
```




    LabelList (60000 items)
    x: ImageList
    Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28),Image (1, 28, 28)
    y: CategoryList
    6,6,6,6,6
    Path: /content/data/mnist_png




```python
x,y = ll.train[0]
```


```python
x.show()
print(y,x.shape)
```

    6 torch.Size([1, 28, 28])



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_26_1.png" alt="">


Next we are doing some transformations, but we are not using `get_transforms` in our previous blogpost, because we are doing digit recognition and it does not make sense for us to flip or rotate the photo in any direction, so generally for small picture like this, we just add random padding `rand_pad`.


```python
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
```

`rand_pad` returns 2 transforms, the padding transform and the cropping transform, the use of * means that we want to put both these transforms in the list, the empty array `[]` signals that we won't be doing any sort of transformation our validation set.


```python
ll = ll.transform(tfms)
```


```python
bs = 128
```

Turning our `LabelLists` into a `DataBunch` object...


```python
# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize()
```

By default, `normalize()` will take a random batch and takes it stats as the stats to use for normalization, this is what we generally do when we do not use a pre-trained model.


```python
x,y = data.train_ds[0]
```


```python
x.show()
print(y)
```

    6



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_36_1.png" alt="">


Our training set now has data argumentation (namely `tfms` that we did for it), we can visually check this process.

Below, we define `_plot` which basically grabs a single image form our training set, and `plot_multi` basically takes in our function, and plot it in a 3 x 3 grid.

Everytime we grab an image from the training set, it's going to load it from disk and transform on the fly, this is why the 3 x 3 plot below shows a slight variations of the same image.


```python
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_38_0.png" alt="">


To grab a batchsize of data, we simply call `.one_batch()`


```python
xb,yb = data.one_batch()
xb.shape,yb.shape
```




    (torch.Size([128, 1, 28, 28]), torch.Size([128]))




```python
data.show_batch(rows=3, figsize=(5,5))
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_41_0.png" alt="">


Let's create a simple CNN now!

### Basic CNN with batchnorm

We know that for this particular CNN, all of our kernel size would be 3, and we want to use stride 2 convolutions, and padding of size 1, so instead of calling `nn.Conv2d` and retyping all of that, we would just define a new function `conv` which sets all that as defaults.


```python
def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
```

Now we have to specify how our hidden layers look like, so a gentle reminder that our input is a rank 4 tensor with a shape of [128, 1, 14, 14], 128 being the batchsize, 1 channel because the image is grayscale, 28 x 28 being the gridsize of the image.


```python
model = nn.Sequential(
    conv(1, 8), # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)
```

Note that we put `stride = 2`, so each time we go through a convolution, the grid size will be reduced by half, as explained in the [previous blogpost](https://cheeloong.github.io/rossmann/#), the `#` in the code above updates the grid size, and it's apparent that after each convolution, the gridsize reduces by half.

`conv(1, 8)` represents the first stride 2 convolutions in our network, the 1 represents the input channel which is 1 because the input is a grayscale image, the output channel is really depending on how many filters we want, so let's say we want 8 for the first convolution. *(Recall that In the case of a fully connected net, the output would be the width of our parameter matrix)*

After the first conv, our grayscale image (1x28x28) is now a feature map with rank 3 tensor of activations (8x14x14), which will go through batch normalization and ReLU.

We then repeat the process, but as we can see, initially as we go deeper into the network, we double the amount of filters we used from the previous layer, after that we start to reduce the amount of filters by half, and then reduce the amount of filters again to the same number of classes of the problem (in this case, we have 10 digits).

In the last conv, we have a feature map with rank 3 tensor of activations (10x1x1),  because the loss function expects a vector of 10 instead of a rank 3 tensor, which is why we call `Flatten()` to turn the rank 3 tensor to a vector.

We now have `data` which is `ImageDataBunch`, CNN `model`, we then put that in `Learner` and choose the respective loss function, specify to print out `accuracy` as the metrics.


```python
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
```


```python
print(learn.summary())
```

    ======================================================================
    Layer (type)         Output Shape         Param #    Trainable
    ======================================================================
    Conv2d               [8, 14, 14]          80         True      
    ______________________________________________________________________
    BatchNorm2d          [8, 14, 14]          16         True      
    ______________________________________________________________________
    ReLU                 [8, 14, 14]          0          False     
    ______________________________________________________________________
    Conv2d               [16, 7, 7]           1,168      True      
    ______________________________________________________________________
    BatchNorm2d          [16, 7, 7]           32         True      
    ______________________________________________________________________
    ReLU                 [16, 7, 7]           0          False     
    ______________________________________________________________________
    Conv2d               [32, 4, 4]           4,640      True      
    ______________________________________________________________________
    BatchNorm2d          [32, 4, 4]           64         True      
    ______________________________________________________________________
    ReLU                 [32, 4, 4]           0          False     
    ______________________________________________________________________
    Conv2d               [16, 2, 2]           4,624      True      
    ______________________________________________________________________
    BatchNorm2d          [16, 2, 2]           32         True      
    ______________________________________________________________________
    ReLU                 [16, 2, 2]           0          False     
    ______________________________________________________________________
    Conv2d               [10, 1, 1]           1,450      True      
    ______________________________________________________________________
    BatchNorm2d          [10, 1, 1]           20         True      
    ______________________________________________________________________
    Flatten              [10]                 0          False     
    ______________________________________________________________________

    Total params: 12,126
    Total trainable params: 12,126
    Total non-trainable params: 0



Let's put 1 batchsize of our data to GPU to make sure everything works out okay.


```python
xb = xb.cuda()
```

Any PyTorch module can be used a function, so we will use our CNN `model` that we have defined earlier on the 1 mini-batch that we popped on the GPU server.


```python
model(xb).shape
```




    torch.Size([128, 10])




```python
learn.lr_find(end_lr=100)
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



```python
learn.recorder.plot()
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_56_0.png" alt="">



```python
learn.fit_one_cycle(3, max_lr=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.222902</td>
      <td>0.151807</td>
      <td>0.949800</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.127626</td>
      <td>0.085388</td>
      <td>0.973800</td>
      <td>00:31</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.070649</td>
      <td>0.041023</td>
      <td>0.987300</td>
      <td>00:30</td>
    </tr>
  </tbody>
</table>


### Refactor

Previously, we defined `conv` which utilizes PyTorch module `nn.Conv2d` and we created a very simple CNN, there's a more sophisticated way to do it, which is to use `conv_layer` which does batch normalization and ReLU behind the scene.

Therefore, we can redefine our stride2 conv using `conv_layer` as below:


```python
def conv2(ni,nf): return conv_layer(ni,nf,stride=2)
```


```python
model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)
```


```python
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
```

Let's fit the learner 10 epochs this time..


```python
learn.fit_one_cycle(10, max_lr=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.244508</td>
      <td>0.174048</td>
      <td>0.945400</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.196057</td>
      <td>0.255560</td>
      <td>0.917900</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.164657</td>
      <td>0.174953</td>
      <td>0.943700</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.139749</td>
      <td>0.089181</td>
      <td>0.971200</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.122364</td>
      <td>0.095143</td>
      <td>0.970100</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.102708</td>
      <td>0.067537</td>
      <td>0.977000</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.083754</td>
      <td>0.053965</td>
      <td>0.982600</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.064042</td>
      <td>0.040444</td>
      <td>0.987900</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.049300</td>
      <td>0.029500</td>
      <td>0.990500</td>
      <td>00:31</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.048261</td>
      <td>0.028291</td>
      <td>0.991300</td>
      <td>00:28</td>
    </tr>
  </tbody>
</table>



```python
print(learn.summary())
```

    ======================================================================
    Layer (type)         Output Shape         Param #    Trainable
    ======================================================================
    Conv2d               [8, 14, 14]          72         True      
    ______________________________________________________________________
    ReLU                 [8, 14, 14]          0          False     
    ______________________________________________________________________
    BatchNorm2d          [8, 14, 14]          16         True      
    ______________________________________________________________________
    Conv2d               [16, 7, 7]           1,152      True      
    ______________________________________________________________________
    ReLU                 [16, 7, 7]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 7, 7]           32         True      
    ______________________________________________________________________
    Conv2d               [32, 4, 4]           4,608      True      
    ______________________________________________________________________
    ReLU                 [32, 4, 4]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [32, 4, 4]           64         True      
    ______________________________________________________________________
    Conv2d               [16, 2, 2]           4,608      True      
    ______________________________________________________________________
    ReLU                 [16, 2, 2]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 2, 2]           32         True      
    ______________________________________________________________________
    Conv2d               [10, 1, 1]           1,440      True      
    ______________________________________________________________________
    ReLU                 [10, 1, 1]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [10, 1, 1]           20         True      
    ______________________________________________________________________
    Flatten              [10]                 0          False     
    ______________________________________________________________________

    Total params: 12,044
    Total trainable params: 12,044
    Total non-trainable params: 0



One thing worth of mentioning is that if we look at the CNN that we built earlier without refactor, we do batch normalization before ReLU activation function, that is consistent with the original BatchNorm paper. In fastai however, the batch normalization comes after ReLU, which I presume yield better results, there is also a [reddit discussion](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/) about this topic.

### Resnet-ish

We have already achieved 99.1% accuracy, that's good and all, but how can we get even better result than that?

Easy, create a deeper network would probably get us a better result, and an easy way to do that is to add a stride 1 conv after each stride 2 conv, since it does not change the feature map size.

There is a problem with this approach which is addressed by this [paper](https://arxiv.org/abs/1512.03385), and here's the key takeaway.

<img src="https://i.imgur.com/a3WDMGm.png" width="500">

Kaiming et al. 2015 attempted to fit a 20-layer CNN with 3x3 conv kernel and without batch normalization on the training set, and another CNN with similar archetecture except that it has 56-layer which has all the stride 1 conv after stride 2 conv.

The 56-layer was expected to overfit and should zip down to low training error because it has more parameters. Surprisingly, that is not the case, it's performing worse than the 20-layer in terms of training error, as shown in the figure above.

<br>

**What actually happened?**

Basically, as we have more layers of convolutions, ReLU, and BatchNorm, accuracy of the model would most probably improve, however, beyond a certain number of layers, the accuracy would start to diminish with increasing number of layers, this is due to what is known as **Vanishing Gradient**, I found a [youtube video](https://www.youtube.com/watch?v=qO_NLVjD6zE) explaining such concept clearly:

That is the reason why the 56-layer was performing poorly than the 20-layer even in terms of training error.

<br>

#### Skip Connection

This problem was addressed by Kaiming et al. 2015 by introducing residual connection (a.k.a identity connection, skip connection),  which is just a simple term to describe connection between the output of previous layers to the output of new layers. The general idea is to add the input to the result of every two convolutions, as shown in the diagram below.

<img src="https://i.imgur.com/LTK5FhE.png" width="400">

The reason why this will be at least as good as the 20-layer is because conv2 and conv1 weights could be set to 0 except for the 20-layers and this will go through a skip connection or known as identity function from the diagram.  

Another [paper](https://arxiv.org/abs/1712.09913) also published the visualization of loss function with and without skip connections

 <img src="https://i.imgur.com/BEsIR5Y.png" width="700">

Here's how we do it in Python, we can build a resblock by using PyTorch models `nn.Module` as follows..


```python
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        self.conv2 = conv_layer(nf,nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))
```

Or we can simply use the `res_block` from the fastai library and pass in how many filters we want to use


```python
help(res_block)
```

    Help on function res_block in module fastai.layers:

    res_block(nf, dense:bool=False, norm_type:Union[fastai.layers.NormType, NoneType]=<NormType.Batch: 1>, bottle:bool=False, **conv_kwargs)
        Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.



Let's try to implement this resblock by calling fastai library `res_block` in our CNN.


```python
model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)
```

Since we are using `res_block` so many times, we can also refractor that one more time by going...


```python
def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))
```


```python
model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)
```


```python
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
```


```python
learn.lr_find(end_lr=100)
learn.recorder.plot()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/mnist-resnet/output_79_2.png" alt="">



```python
learn.fit_one_cycle(12, max_lr=0.05)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.262617</td>
      <td>0.165254</td>
      <td>0.955400</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.130820</td>
      <td>0.118797</td>
      <td>0.964200</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.116942</td>
      <td>0.184853</td>
      <td>0.942000</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.103353</td>
      <td>0.160178</td>
      <td>0.951700</td>
      <td>00:31</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.079369</td>
      <td>0.079555</td>
      <td>0.975400</td>
      <td>00:31</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.063511</td>
      <td>0.035730</td>
      <td>0.987900</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.048071</td>
      <td>0.060546</td>
      <td>0.981400</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.044386</td>
      <td>0.024461</td>
      <td>0.991000</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.036555</td>
      <td>0.029766</td>
      <td>0.991600</td>
      <td>00:31</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024889</td>
      <td>0.020684</td>
      <td>0.994300</td>
      <td>00:31</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.021405</td>
      <td>0.017585</td>
      <td>0.994600</td>
      <td>00:32</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.016599</td>
      <td>0.017346</td>
      <td>0.994600</td>
      <td>00:31</td>
    </tr>
  </tbody>
</table>


99.46% accuracy, this Resnet architecture probably does better digit recognition than me doing it manually.

Let's dig a little on the source code for `res_block` and see if it aligns well with what we know.

<img src="https://i.imgur.com/U3Ksr8t.png" width="800">


As we can see from the source code, the source code for `res_block` is largely the same, except that it's `SequentialEx` model which is almost the same as `Sequential` but with `MergeLayer(dense)` part of the code, `dense` takes on a boolean value, if `dense = False`, then the code will run with the ResNet architecture, otherwise, it will run a **DenseNet**, which is almost the same as ResNet architecture, but instead of adding the input to the result of the 2 convs, it's concatenating them.

So as we go deeper in the network, we still have the original input pixel, the original layer 1 pixel, original layer 2 pixel, and so forth. Consequently, this is very memory intensive, but they have very few parameters, so it often works well with small dataset and especially segmentation tasks.






```python
print(learn.summary())
```

    ======================================================================
    Layer (type)         Output Shape         Param #    Trainable
    ======================================================================
    Conv2d               [8, 14, 14]          72         True      
    ______________________________________________________________________
    ReLU                 [8, 14, 14]          0          False     
    ______________________________________________________________________
    BatchNorm2d          [8, 14, 14]          16         True      
    ______________________________________________________________________
    Conv2d               [8, 14, 14]          576        True      
    ______________________________________________________________________
    ReLU                 [8, 14, 14]          0          False     
    ______________________________________________________________________
    BatchNorm2d          [8, 14, 14]          16         True      
    ______________________________________________________________________
    Conv2d               [8, 14, 14]          576        True      
    ______________________________________________________________________
    ReLU                 [8, 14, 14]          0          False     
    ______________________________________________________________________
    BatchNorm2d          [8, 14, 14]          16         True      
    ______________________________________________________________________
    MergeLayer           [8, 14, 14]          0          False     
    ______________________________________________________________________
    Conv2d               [16, 7, 7]           1,152      True      
    ______________________________________________________________________
    ReLU                 [16, 7, 7]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 7, 7]           32         True      
    ______________________________________________________________________
    Conv2d               [16, 7, 7]           2,304      True      
    ______________________________________________________________________
    ReLU                 [16, 7, 7]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 7, 7]           32         True      
    ______________________________________________________________________
    Conv2d               [16, 7, 7]           2,304      True      
    ______________________________________________________________________
    ReLU                 [16, 7, 7]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 7, 7]           32         True      
    ______________________________________________________________________
    MergeLayer           [16, 7, 7]           0          False     
    ______________________________________________________________________
    Conv2d               [32, 4, 4]           4,608      True      
    ______________________________________________________________________
    ReLU                 [32, 4, 4]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [32, 4, 4]           64         True      
    ______________________________________________________________________
    Conv2d               [32, 4, 4]           9,216      True      
    ______________________________________________________________________
    ReLU                 [32, 4, 4]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [32, 4, 4]           64         True      
    ______________________________________________________________________
    Conv2d               [32, 4, 4]           9,216      True      
    ______________________________________________________________________
    ReLU                 [32, 4, 4]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [32, 4, 4]           64         True      
    ______________________________________________________________________
    MergeLayer           [32, 4, 4]           0          False     
    ______________________________________________________________________
    Conv2d               [16, 2, 2]           4,608      True      
    ______________________________________________________________________
    ReLU                 [16, 2, 2]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 2, 2]           32         True      
    ______________________________________________________________________
    Conv2d               [16, 2, 2]           2,304      True      
    ______________________________________________________________________
    ReLU                 [16, 2, 2]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 2, 2]           32         True      
    ______________________________________________________________________
    Conv2d               [16, 2, 2]           2,304      True      
    ______________________________________________________________________
    ReLU                 [16, 2, 2]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [16, 2, 2]           32         True      
    ______________________________________________________________________
    MergeLayer           [16, 2, 2]           0          False     
    ______________________________________________________________________
    Conv2d               [10, 1, 1]           1,440      True      
    ______________________________________________________________________
    ReLU                 [10, 1, 1]           0          False     
    ______________________________________________________________________
    BatchNorm2d          [10, 1, 1]           20         True      
    ______________________________________________________________________
    Flatten              [10]                 0          False     
    ______________________________________________________________________

    Total params: 41,132
    Total trainable params: 41,132
    Total non-trainable params: 0



That would be all for this blogpost, in the next one we will explore U-Net on camvid dataset that we have used previously in earlier blogpost because we have some unfinished business with U-Net, we have used it without really understanding the details, so that is what we will be looking at next blogpost!
