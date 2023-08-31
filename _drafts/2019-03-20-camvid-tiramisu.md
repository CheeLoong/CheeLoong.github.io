---
title: "Semantic Segmentation on Cambridge-driving Labeled Video"
date: 2019-03-20
permalink: /camvid-tiramisu/
tags: [fastai, deep learning, cnn, u-net, camvid-tiramisu, semantic segmentation]
excerpt: "Semantic Segmentation with U-Net Convolutional Neural Network"
mathjax: "true"
published: false
---

# Image segmentation with CamVid

![https://towardsdatascience.com/detection-and-segmentation-through-convnets-47aa42de27ea](https://cdn-images-1.medium.com/max/1200/1*SNvD04dEFIDwNAqSXLQC_g.jpeg)


In this blogpost, we will be doing semantic segmentation and that's different from object detection! Why aren't we doing object detection yet? Well cuz my brain isn't large enough yet.

**So what is semantic segmentation?**

The basic idea is to do classification for every pixel in an image and our goal is to have a different colours for different things.

For this to be made possible, we need dataset where someone has labeled every pixel in an image for our model to learn.

Before, we talk about the dataset, as usual, we have to rent a GPU..

# Colab Cloud Server VM setup & FastAI Configurations

First we need to use GPU for deep learning applications, so instead of buying, we can rent it from cloud servers. There are many other server options, but basically I chose Colab because issa FREE.

To learn how to set up a Colab Server which supports fastai library and its applications, click [here](https://course.fast.ai/start_colab.html).

NB: This is a free service that may not always be available, and requires extra steps to ensure your work is saved. Be sure to read the docs on the Colab web-site to ensure you understand the limitations of the system.


```python
# Permit collaboratory instance to read/write to Google Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3/'
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code

    Enter your authorization code:
    ··········
    Mounted at /content/gdrive



```python
  !curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    [31mfeaturetools 0.4.1 has requirement pandas>=0.23.0, but you'll have pandas 0.22.0 which is incompatible.[0m
    [31mdatascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.[0m
    [31malbumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.8 which is incompatible.[0m
    Done.



```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
from fastai.vision import *
from fastai.callbacks.hooks import *
```

# Let's talk Data

The  [The One Hundred Layer Tiramisu paper](https://arxiv.org/abs/1611.09326) used a modified version of Camvid, with smaller images and few classes. You can get it from the CamVid directory of this repo:

    git clone https://github.com/alexgkendall/SegNet-Tutorial.git

## Getting the Data


```python
# create the data directory
path = Config.data_path()/'camvid-tiramisu'
path.mkdir(parents=True, exist_ok=True)
path
```




    PosixPath('/root/.fastai/data/camvid-tiramisu')




```python
# clone camvid directory to designated folder
!git clone https://github.com/alexgkendall/SegNet-Tutorial.git /content/data/camvid-tiramisu
```

    Cloning into '/content/data/camvid-tiramisu'...
    remote: Enumerating objects: 2785, done.[K
    remote: Total 2785 (delta 0), reused 0 (delta 0), pack-reused 2785[K
    Receiving objects: 100% (2785/2785), 340.84 MiB | 41.48 MiB/s, done.
    Resolving deltas: 100% (81/81), done.



```python
# check files from git clone
path.ls()
```




    [PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/Example_Models'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/.gitattributes'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/.gitignore'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/Models'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/docker'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/.git'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/README.md'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/Scripts')]



## Understanding the Data

Let's have a look and see what's underneath our `val` folder, which represent our validation set of our image data. On the other hand, `valannot` folder represents the segment mask of our image data.


```python
# reassign path
path = Config.data_path()/'camvid-tiramisu/CamVid'
```


```python
# check files under 'CamVid' folder
path.ls()
```




    [PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/test.txt'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/valannot'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/val.txt'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/train'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/trainannot'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/val'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/train.txt'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/testannot'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/test')]




```python
# images under 'val' folder
fnames = get_image_files(path/'val')
fnames[:3]
```




    [PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/val/0016E5_08159.png'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/val/0016E5_08025.png'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/val/0016E5_08017.png')]




```python
# segment mask under `valannot` folder
lbl_names = get_image_files(path/'valannot')
lbl_names[:3]
```




    [PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/valannot/0016E5_08159.png'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/valannot/0016E5_08025.png'),
     PosixPath('/root/.fastai/data/camvid-tiramisu/CamVid/valannot/0016E5_08017.png')]




```python
# this is the first image file under `val` folder
img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(8,8))
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_20_0.png" alt="">


Note that this image is from the `val` folder. Now we need to find a way to navigate to the segment mask of this image in the `valannot` folder which stores image files that contains integer for each pixel on the image.

We will define a function `get_y_fn` below to do that.


```python
def get_y_fn(x): return Path(str(x.parent)+'annot')/x.name
```

After we manage to find a way to link the image name in the `val` folder to the image name in the `valannot` folder, we use `open_mask` to get the intergers stored in the pixels of the image, then use `.show` to display the colour coded image.


```python
# we use open_mask because this is an image file that contains integer
mask = open_mask(get_y_fn(img_f))
mask.show(figsize = (8,8), alpha=1)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_24_0.png" alt="">


We would like to know what each colour in the image represents, the following line is given along with this dataset:


```python
codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
     'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
```


```python
src_size = np.array(mask.shape[1:])
print(src_size)
print(mask.data)
```

    [360 480]
    tensor([[[1, 1, 1,  ..., 5, 5, 5],
             [1, 1, 1,  ..., 5, 5, 5],
             [1, 1, 1,  ..., 5, 5, 5],
             ...,
             [3, 3, 3,  ..., 3, 3, 3],
             [3, 3, 3,  ..., 3, 3, 3],
             [3, 3, 3,  ..., 3, 3, 3]]])


When we look at `mask.data`, we are looking at the tensor which contains the numerical representation of the objects.

For example, in the top left of the tensor, we see 1's (i.e. Building from `codes`), bottom right of the tensor are 3's (i.e. Road from `codes`), and so on..

## Creating `DataBunch`

This step is very important, regardless of what kind of dataset we are using, in order to do modelling with fastai, we first need to convert the dataset to a `DataBunch` object.

Different types of datasets will use slightly different functions to convert them to `DataBunch`, so its a good idea to check out [data block API](https://docs.fast.ai/data_block.html).

For this case, we need to use `SegmentationItemList`. This will make sure the model created has the proper loss function to deal with the multiple classes.


```python
bs = 4
size = src_size//2
```


```python
src = (SegmentationItemList.from_folder(path)
       # Where to find the data? -> in path and its subfolders
       .split_by_folder(valid='val')
       # How to split in train/valid? -> use the folders
       .label_from_func(get_y_fn, classes=codes))
       # How to label? -> use the label function on the file name of the data
```

Here we need to use `tfm_y=True` in the transform call because we need the same transforms to be applied to the target mask as were applied to the image.


```python
data = (src.transform(get_transforms(), tfm_y=True)
       # Data augmentation? -> use tfms with a size of 128, also transform the label images
        .databunch(bs=bs)
       # convert to databunch with bs of 4
        .normalize(imagenet_stats))
       # normalize the databunch
```


```python
data
```




    ImageDataBunch;

    Train: LabelList (367 items)
    x: SegmentationItemList
    Image (3, 360, 480),Image (3, 360, 480),Image (3, 360, 480),Image (3, 360, 480),Image (3, 360, 480)
    y: SegmentationLabelList
    ImageSegment (1, 360, 480),ImageSegment (1, 360, 480),ImageSegment (1, 360, 480),ImageSegment (1, 360, 480),ImageSegment (1, 360, 480)
    Path: /root/.fastai/data/camvid-tiramisu/CamVid;

    Valid: LabelList (101 items)
    x: SegmentationItemList
    Image (3, 360, 480),Image (3, 360, 480),Image (3, 360, 480),Image (3, 360, 480),Image (3, 360, 480)
    y: SegmentationLabelList
    ImageSegment (1, 360, 480),ImageSegment (1, 360, 480),ImageSegment (1, 360, 480),ImageSegment (1, 360, 480),ImageSegment (1, 360, 480)
    Path: /root/.fastai/data/camvid-tiramisu/CamVid;

    Test: None




```python
data.show_batch(2, figsize=(10,7))
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_36_0.png" alt="">


# Model

Note that in `codes`, there's label called `Void`, we are not sure about what this label means in a pixel, but we do know it's not part of the object that we are interested in.

Thus, we need a define an accuracy metric that excludes `Void` pixel, as follows:


```python
# this creates a dictionary which has the object of interest as the key and an integer as its code
name2id = {v:k for k,v in enumerate(codes)}
# this looks for the code which has 'Void' as its key
void_code = name2id['Void']

# def custom accuracy for camvid dataset
def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
```

We also defined a weight decay term `wd`, which helps avoid overfitting, and potentially help the learner to train. (Will get back to this in future blogpost)


```python
metrics=acc_camvid
wd=1e-2
```

We will not be using the standard `cnn_learner` architecture, instead we will use  a `unet_learner` because U-Net  is the recommended architecture for image segmentation model.

Click [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) to check out the U-Net paper.


```python
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, bottle=True)
```

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.torch/models/resnet34-333f7ec4.pth
    87306240it [00:01, 78943877.36it/s]



```python
lr_find(learn)
learn.recorder.plot()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_44_2.png" alt="">


`1e-4` looks like the steepest slope in the plot before learning rate spikes up, so we are going to pick that as the learning rate.


```python
lr=1e-4
```

`pct_start` is the Percentage of total number of iterations when learning rate rises during one cycle.

In our case, we set `pct_start = 0.8`, this means that for 80% of our iterations in 1 epoch, the learning rate will increase up to `lr=2e-3`, then decreases for the rest of the 20% iterations.

Read more [here](https://forums.fast.ai/t/what-is-the-pct-start-mean/26168/7).

**But why do we want the learning rate to increase at first, and then decrease towards the end?**

Practically speaking, the loss function curve is almost always bumpy, and to set the learning rate high at first is really to jump over the bumps that will gives local minimum solutions, and that's why we want learning rate to behave this way, this is also known as [**learning rate annealing**](https://www.jeremyjordan.me/nn-learning-rate/).


```python
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_01_0.png" alt="">




```python
# pct_start = 0.8 plot
learn.recorder.plot_lr()
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_49_0.png" alt="">



```python
learn.save('stage-1')
```


```python
learn.load('stage-1');
```

Let's unfreeze the model to see we can improve the learner.


```python
learn.unfreeze()
```


```python
lr_find(learn)
learn.recorder.plot()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_54_2.png" alt="">



```python
lrs = slice(lr/100,lr)
```


```python
learn.fit_one_cycle(6, lrs, pct_start=0.8)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_02_0.png" alt="">



`learn.recorder` keep tracks of what is going on during training, `learn.recorder.plot_losses()` plots the training loss and validation loss.


```python
learn.recorder.plot_losses()
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_58_0.png" alt="">



```python
learn.save('stage-2');
```

# Transfer Learning

In my previous post, I talked about how transfer learning can help improve the learner

click [here](https://cheeloong.github.io/planets/#) to read my previous blogpost.


```python
learn=None
gc.collect()
```




    8355



You may have to restart your kernel and come back to this stage if you run out of memory, and may also need to decrease `bs`.


```python
size = src_size
bs=2
```


```python
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
```


```python
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, bottle=True).load('stage-2');
```

So now we know that this learner has the same weights as `'stage-2'` but utilizing a larger screen size data.



```python
lr_find(learn)
learn.recorder.plot()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_68_2.png" alt="">



```python
lr=1e-5
```


```python
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_03_0.png" alt="">




```python
learn.save('stage-1-big')
```


```python
learn.load('stage-1-big');
```


```python
learn.unfreeze()
```


```python
lrs = slice(lr/1000,lr/10)
```


```python
learn.fit_one_cycle(10, lrs)
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_04_0.png" alt="">



92.57% validation accuracy (not accounting for `Void` pixels), this is pretty decent!


```python
learn.save('stage-2-big')
```


```python
learn.load('stage-2-big');
```


```python
learn.show_results(rows=3, figsize=(9,11))
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/camvid-tiramisu/output_79_0.png" alt="">




```python
# export the learner to be used in production
learn.export()
```

We know `92.57%` is good, but like how good? [The One Hundred Layer Tiramisu paper](https://arxiv.org/abs/1611.09326) first published in 28th November 2016 achieved an accuracy of `91.5%`, which was the state-of-the-art performance at that time, and they probably spent weeks or even months trying to get there.


![100layer](https://i.imgur.com/W4D5SjYg.png)

So really, I just want to say...

![gladiator](https://cdn.shopify.com/s/files/1/0824/6367/products/gladiator_trump_with_text-mockup_grande.png?v=1547717744)

Thank you for reading, and as usual, credits to [fast.ai](https://www.fast.ai/) for most of the resources.
