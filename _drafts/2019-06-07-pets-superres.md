---
title: "Perceptual Losses for Super-Resolution on Oxford Pets dataset"
date: 2019-06-07
permalink: /pets-superres/
tags: [fastai, pytorch, super-resolution, perceptual losses, pets, deep learning]
excerpt: "Using Perceptual Losses as the loss function for super-resolution task with fastai library"
mathjax: "true"
published: false
---

In this blogpost, we will revisit the oxford pets dataset just like in the previous blogpost, but with a little tweak, when we try to crappify our images, we are not going to write a random number on the image, we are just simply going to make the resolution worse since we have learnt how to remove a number on an image from the previous blogpost.

Previously we used **GAN** and we got pretty decent quality of generated images, but it's still losing some features that we care about, like having an eyeball, fine detail of the fur, etc. So how can we tackle this problem?

The simple answer is to use a better loss function, we want to use a loss function that not only can it identify high quality images, but also the kind of images that it supposed to look like, what I mean by that is if we have a bad quality cat image, we want to not only turn it into higher quality, but at the same time, the output should still show a cat.

That better loss function is called **Perceptual Losses**, and the source is from this [paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf). In fastai library, it is called **Feature Losses**, but they are referring to the same thing.

<img src="https://i.imgur.com/eT0oRP6.png" width="800">

So let's look at the procedure, they put image through something called **Image Transform Net** which really is a U-Net generator, because when they published the paper, U-Net wasn't a thing yet, so they called it **Image Transform Net**, just like U-Net, it has downsampling part and upsampling part, the downsampling part is also known as **encoder**, while the upsampling part is called **decoder**.

After the input went through the Image Transform Net, they have generated predictions $\hat y$, they then chuck it through a pre-trained ImageNet network known as VGG, it is kind of old now, but it's still usable for this application. The output is whatever class that the model think the generated image is, a cat, a dog, a ghost, etc.

But in the process of getting to that final classification, it goes through lots of different layers. In this case, they've color-coded all the layers with the same grid size and the feature map with the same color. So every time the layer switched colors, it means switching grid size. So there was a stride 2 conv or in VGG's case they used to use some maxpooling layers which is a similar idea.

Here's the interesting bit, instead of getting the final output of the VGG model, they instead take the activations in the middle layers, those activations might be a feature map of say 256 channels by 28 by 28, and each grid in the 28 by 28 would outline things like does this look furry, does this look circular, does it look like an eyeball, etc.

They also chuck the real target $y$ (the real images) through the same pre-trained VGG network, and pulled out the activations of the same layer, then a **Mean Squared Error (MSE)** comparison between the generated image with the target image, so something like in grid (1,1) of the 28 by 28 real image feature map is something furry and white, and that in grid (1,1) of the 28 by 28 generated image feature map does not even show any fur, then **MSE** will be large. That is what is known as **Feature Losses** in fastai or **Perceptual Losses** from the paper.

Apparently, this can help the model to generate better quality images, let's start to get our hands dirty!

## Super resolution

Let's first import all the libraries that we will be using for this task.


```python
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn
```


```python
# this line creates data & models folder and functionalities integration (e.g. untar_data, model.save)  
  !curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    Done.


Download the images using built-in function `untar_data`, nothing new so far.


```python
path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'small-96'
path_mr = path/'small-256'
```


```python
il = ImageList.from_folder(path_hr)
```

Similar to previous blogpost, we try to reduce the resolution of the image, but in the process of worsen the quality of the image, we won't be adding a number on top of the image, and the JPEG quality is always 60 instead of a random number in the previous blogpost.


```python
def crappify(fn, i, path, size):
    dest = path/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)
```

Now let's *crappify* the images, set the `path` and the desired `size` to store our resized *crappfied* images.


```python
# create smaller image sets the first time this nb is run
sets = [(path_lr, 96), (path_mr, 256)]
for p,size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(crappify, path=p, size=size), il.items)
```

    resizing to 96 into /root/.fastai/data/oxford-iiit-pet/small-96




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='7390' class='' max='7390', style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [7390/7390 00:32<00:00]
    </div>



    resizing to 256 into /root/.fastai/data/oxford-iiit-pet/small-256




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='7390' class='' max='7390', style='width:300px; height:20px; vertical-align: middle;'></progress>
      100.00% [7390/7390 00:51<00:00]
    </div>



Okay we got the resized images in their respective folders, now it's time to create `DataBunch` object for the data, and we set batchsize, size of the image, convert to `ImageImageLists` to `LabelList`, some basic data augmentation, converting to `DataBunch`, normalize because we are using a pre-trained model, yada yada.


```python
bs,size=32,128
arch = models.resnet34

src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
```


```python
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
```


```python
data = get_data(bs,size)
```


```python
data
```




    ImageDataBunch;

    Train: LabelList (6651 items)
    x: ImageImageList
    Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128)
    y: ImageList
    Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128)
    Path: /root/.fastai/data/oxford-iiit-pet/small-96;

    Valid: LabelList (739 items)
    x: ImageImageList
    Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128)
    y: ImageList
    Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128),Image (3, 128, 128)
    Path: /root/.fastai/data/oxford-iiit-pet/small-96;

    Test: None




```python
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_17_0.png" alt="">


Here we have got the original higher res photo that we download from oxford pets dataset (on the right) and the same photo that went through the *crappfication* (on the left), now we want to create a loss function that was discussed in the beginning of the blogpost, the **Perceptual Loss** a.k.a **Feature Loss** in fast.ai library.



## Feature loss


```python
t = data.valid_ds[0][1].data
t = torch.stack([t,t])
```


```python
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)
```


```python
gram_matrix(t)
```




    tensor([[[0.1386, 0.1384, 0.1066],
             [0.1384, 0.1395, 0.1063],
             [0.1066, 0.1063, 0.0908]],

            [[0.1386, 0.1384, 0.1066],
             [0.1384, 0.1395, 0.1063],
             [0.1066, 0.1063, 0.0908]]])



Here we define a base loss function which is used to compare the pixels and the features, the main choice is either **Mean Squared Error (MSE)** or **L1**, it's almost the same, so let's be cool and pick a version we have never used before.


```python
base_loss = F.l1_loss
```

Next, we create a VGG model, we will want to use the pre-trained model, which is why we set `True` in `vgg16_vn`, `.features` contains the convolutional part of the model, as discussed previously, we are not interested in the final output of the layer, but rather the intermediate activations.'

We chuck the model on the GPU, and put the model into `.eval()`  mode, because we are not training it, and we turn off `requires_grad` since we will not be updating the weights of the model, we are just using it for inference, for the loss function.


```python
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
```

    Downloading: "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth" to /root/.cache/torch/checkpoints/vgg16_bn-6c64b313.pth
    100%|██████████| 553507836/553507836 [00:23<00:00, 23536503.35it/s]


Enumerate through all the children of the VGG model, find all the max pooling layers because in VGG, that is where the grid size changes, and we want to grab features everytime just before the grid size changes, `i` is the layer where the grid size changes, so we grab `i-1`.


```python
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]
```




    ([5, 12, 22, 32, 42],
     [ReLU(inplace), ReLU(inplace), ReLU(inplace), ReLU(inplace), ReLU(inplace)])



We obtained a list of layer numbers `[5, 12, 22, 32, 42]` and we name it as `blocks`, they are all just before the max pooling layers, and they are all ReLU's, which makes sense because after ReLU's is where `Conv2D` happens, where grid size changes, at least in the case of VGG.

Now we are ready to define `FeatureLoss` class, subclassing from `nn.Module`.


```python
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()
```

Let's try to understanding a little on what `FeatureLoss` class does;

- `m_feat` refers to a pre-trained model which contains the features we want our `FeatureLoss` on,

- we grab all the layers from that network that we want the features for, to create the losses with ` [self.m_feat[i] for i in layer_ids]`.

- to grab intermediate layers in PyTorch, we use `hook_outputs`. `hooks.stored` contains our hooked outputs.

now the `forward`;

- we call `make_features`, which goes through all the stored activations from the VGG model and grab a copy of them.

- we will `make_features` for the `target` which is our actual image, and we call in `out_feat`; we do the same for our `input` which is the generated image, and we call it `in_feat`, this is where we have got the activations from the intermediate layers of the VGG network for both the actual image and the generated image.

- next, we calculate the L1 loss between the `input` pixels and the `target` pixels with `base_loss(input, target)` because we still want to calculate Pixel Loss. On top of that, we also add the L1 loss between the intermediate layer activations of the real image `out_feat` and the generated image `in_feat`, we do this on each of the `block` that we stored earlier. All this ends up in a list called `feat_losses`, then sum it all up. The reason why we use this as a list is because we've got this callback that if we put them into `.metrics` in the loss function, it will print out all the separate layer loss amounts.




```python
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
```

## Train

Now we can train a U-Net using a pre-trained `resnet34`, passing a loss function which is using a pre-trained VGG model. `callback_fns` is that callback we mentioned why we made `feat_losses`a list in the `FeatureLoss` class. `blur` and `norm_type` is beyond the scope of this blogpost, just remember to put them in.


```python
wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();
```

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth
    100%|██████████| 87306240/87306240 [00:03<00:00, 27791573.26it/s]



```python
learn.lr_find()
learn.recorder.plot()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_36_2.png" alt="">


```python
lr = 1e-3
```

Let's refactor what we will be doing with our model `learn`, we know we want to fit it, save the result, and then show the result, so we define `do_fit` to do that.


```python
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)
```


```python
do_fit('1a', slice(lr*10))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>pixel</th>
      <th>feat_0</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>gram_0</th>
      <th>gram_1</th>
      <th>gram_2</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.886171</td>
      <td>3.792608</td>
      <td>0.144805</td>
      <td>0.231925</td>
      <td>0.319104</td>
      <td>0.225272</td>
      <td>0.568750</td>
      <td>1.232286</td>
      <td>1.070467</td>
      <td>02:12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.766050</td>
      <td>3.650140</td>
      <td>0.151221</td>
      <td>0.229633</td>
      <td>0.311545</td>
      <td>0.216603</td>
      <td>0.529802</td>
      <td>1.180650</td>
      <td>1.030684</td>
      <td>02:12</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.676016</td>
      <td>3.565730</td>
      <td>0.145754</td>
      <td>0.226765</td>
      <td>0.304995</td>
      <td>0.209882</td>
      <td>0.525366</td>
      <td>1.153265</td>
      <td>0.999705</td>
      <td>02:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.641587</td>
      <td>3.458262</td>
      <td>0.147482</td>
      <td>0.222755</td>
      <td>0.296970</td>
      <td>0.204589</td>
      <td>0.495623</td>
      <td>1.111829</td>
      <td>0.979015</td>
      <td>02:11</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.566939</td>
      <td>3.406335</td>
      <td>0.154068</td>
      <td>0.220950</td>
      <td>0.292872</td>
      <td>0.201263</td>
      <td>0.478660</td>
      <td>1.098038</td>
      <td>0.960483</td>
      <td>02:12</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.563146</td>
      <td>3.429187</td>
      <td>0.148542</td>
      <td>0.219867</td>
      <td>0.294611</td>
      <td>0.203530</td>
      <td>0.489307</td>
      <td>1.098856</td>
      <td>0.974474</td>
      <td>02:12</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.473958</td>
      <td>3.357378</td>
      <td>0.139740</td>
      <td>0.219882</td>
      <td>0.289128</td>
      <td>0.199748</td>
      <td>0.470852</td>
      <td>1.082224</td>
      <td>0.955805</td>
      <td>02:12</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.479109</td>
      <td>3.298060</td>
      <td>0.139252</td>
      <td>0.218445</td>
      <td>0.285587</td>
      <td>0.195888</td>
      <td>0.456176</td>
      <td>1.060188</td>
      <td>0.942524</td>
      <td>02:14</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.431122</td>
      <td>3.247562</td>
      <td>0.135472</td>
      <td>0.215652</td>
      <td>0.282397</td>
      <td>0.193139</td>
      <td>0.449652</td>
      <td>1.046846</td>
      <td>0.924405</td>
      <td>02:13</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.333428</td>
      <td>3.163835</td>
      <td>0.134335</td>
      <td>0.214761</td>
      <td>0.277888</td>
      <td>0.187503</td>
      <td>0.429132</td>
      <td>1.020663</td>
      <td>0.899554</td>
      <td>02:13</td>
    </tr>
  </tbody>
</table>



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_40_1.png" alt="">

Since we are using a pre-trained network in our U-Net, by default fast.ai start with frozen layers for the downsampling part, so as per usual we unfreeze and train some more.


```python
learn.unfreeze()
```


```python
do_fit('1b', slice(1e-5,lr))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>pixel</th>
      <th>feat_0</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>gram_0</th>
      <th>gram_1</th>
      <th>gram_2</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.305859</td>
      <td>3.156691</td>
      <td>0.133818</td>
      <td>0.214156</td>
      <td>0.277167</td>
      <td>0.187051</td>
      <td>0.428372</td>
      <td>1.018768</td>
      <td>0.897359</td>
      <td>02:06</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.290506</td>
      <td>3.150845</td>
      <td>0.133536</td>
      <td>0.214124</td>
      <td>0.276997</td>
      <td>0.186716</td>
      <td>0.427197</td>
      <td>1.017001</td>
      <td>0.895275</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.299270</td>
      <td>3.147583</td>
      <td>0.133517</td>
      <td>0.214064</td>
      <td>0.276767</td>
      <td>0.186421</td>
      <td>0.426787</td>
      <td>1.016645</td>
      <td>0.893381</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.288661</td>
      <td>3.138078</td>
      <td>0.133636</td>
      <td>0.213800</td>
      <td>0.276131</td>
      <td>0.186008</td>
      <td>0.424317</td>
      <td>1.012763</td>
      <td>0.891425</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.298153</td>
      <td>3.132771</td>
      <td>0.134050</td>
      <td>0.213418</td>
      <td>0.275638</td>
      <td>0.185731</td>
      <td>0.423932</td>
      <td>1.010383</td>
      <td>0.889619</td>
      <td>02:10</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.272896</td>
      <td>3.121851</td>
      <td>0.133716</td>
      <td>0.213121</td>
      <td>0.275185</td>
      <td>0.185173</td>
      <td>0.420191</td>
      <td>1.007193</td>
      <td>0.887272</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.270102</td>
      <td>3.122237</td>
      <td>0.133771</td>
      <td>0.213074</td>
      <td>0.275168</td>
      <td>0.184694</td>
      <td>0.420764</td>
      <td>1.009157</td>
      <td>0.885609</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.286982</td>
      <td>3.130420</td>
      <td>0.133968</td>
      <td>0.213186</td>
      <td>0.274885</td>
      <td>0.185579</td>
      <td>0.423203</td>
      <td>1.010425</td>
      <td>0.889175</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>8</td>
      <td>3.262553</td>
      <td>3.110287</td>
      <td>0.134226</td>
      <td>0.211993</td>
      <td>0.274293</td>
      <td>0.184476</td>
      <td>0.418150</td>
      <td>1.003001</td>
      <td>0.884149</td>
      <td>02:09</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3.259192</td>
      <td>3.098200</td>
      <td>0.133077</td>
      <td>0.212246</td>
      <td>0.273374</td>
      <td>0.183324</td>
      <td>0.416459</td>
      <td>1.001015</td>
      <td>0.878706</td>
      <td>02:09</td>
    </tr>
  </tbody>
</table>



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_43_1.png" alt="">

Looks like it's better already! Let's tweak the size by doubling it, and halve the batch size to avoid insufficient GPU memory problem, freeze it, and train.


```python
data = get_data(12,size*2)
```


```python
learn.data = data
learn.freeze()
gc.collect()
```




    19102




```python
learn.load('1b');
```

    /usr/local/lib/python3.6/dist-packages/torch/serialization.py:256: UserWarning: Couldn't retrieve source code for container of type FeatureLoss. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "



```python
do_fit('2a')
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>pixel</th>
      <th>feat_0</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>gram_0</th>
      <th>gram_1</th>
      <th>gram_2</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.225376</td>
      <td>2.185290</td>
      <td>0.161320</td>
      <td>0.257686</td>
      <td>0.292701</td>
      <td>0.153935</td>
      <td>0.379293</td>
      <td>0.576543</td>
      <td>0.363811</td>
      <td>08:18</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.197070</td>
      <td>2.154776</td>
      <td>0.161691</td>
      <td>0.258125</td>
      <td>0.291580</td>
      <td>0.153081</td>
      <td>0.364252</td>
      <td>0.566762</td>
      <td>0.359284</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.186401</td>
      <td>2.143717</td>
      <td>0.164252</td>
      <td>0.258314</td>
      <td>0.290389</td>
      <td>0.152067</td>
      <td>0.359591</td>
      <td>0.562303</td>
      <td>0.356800</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.157262</td>
      <td>2.126101</td>
      <td>0.163328</td>
      <td>0.257844</td>
      <td>0.288987</td>
      <td>0.151266</td>
      <td>0.352374</td>
      <td>0.558364</td>
      <td>0.353937</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.155367</td>
      <td>2.111059</td>
      <td>0.164497</td>
      <td>0.257444</td>
      <td>0.287402</td>
      <td>0.150576</td>
      <td>0.344391</td>
      <td>0.554151</td>
      <td>0.352598</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.122349</td>
      <td>2.102529</td>
      <td>0.164177</td>
      <td>0.257280</td>
      <td>0.286493</td>
      <td>0.149182</td>
      <td>0.341692</td>
      <td>0.553759</td>
      <td>0.349946</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.112471</td>
      <td>2.095867</td>
      <td>0.164917</td>
      <td>0.257617</td>
      <td>0.285726</td>
      <td>0.149298</td>
      <td>0.339237</td>
      <td>0.550531</td>
      <td>0.348541</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.104733</td>
      <td>2.089682</td>
      <td>0.165018</td>
      <td>0.254993</td>
      <td>0.283424</td>
      <td>0.147400</td>
      <td>0.341948</td>
      <td>0.551457</td>
      <td>0.345442</td>
      <td>08:04</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.108611</td>
      <td>2.076551</td>
      <td>0.166236</td>
      <td>0.255857</td>
      <td>0.282911</td>
      <td>0.146739</td>
      <td>0.334595</td>
      <td>0.546347</td>
      <td>0.343866</td>
      <td>08:03</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.073607</td>
      <td>2.057561</td>
      <td>0.164193</td>
      <td>0.255132</td>
      <td>0.281260</td>
      <td>0.146045</td>
      <td>0.329636</td>
      <td>0.539821</td>
      <td>0.341474</td>
      <td>08:03</td>
    </tr>
  </tbody>
</table>



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_48_1.png" alt="">


```python
learn.unfreeze()
```


```python
do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>pixel</th>
      <th>feat_0</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>gram_0</th>
      <th>gram_1</th>
      <th>gram_2</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.088794</td>
      <td>2.058649</td>
      <td>0.164308</td>
      <td>0.255304</td>
      <td>0.281285</td>
      <td>0.146020</td>
      <td>0.329518</td>
      <td>0.540691</td>
      <td>0.341523</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.065200</td>
      <td>2.058934</td>
      <td>0.164693</td>
      <td>0.255722</td>
      <td>0.281721</td>
      <td>0.146050</td>
      <td>0.328886</td>
      <td>0.540392</td>
      <td>0.341470</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.078276</td>
      <td>2.054307</td>
      <td>0.164148</td>
      <td>0.254858</td>
      <td>0.280948</td>
      <td>0.145622</td>
      <td>0.328883</td>
      <td>0.539401</td>
      <td>0.340447</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.075890</td>
      <td>2.053493</td>
      <td>0.164240</td>
      <td>0.255246</td>
      <td>0.280984</td>
      <td>0.145528</td>
      <td>0.327753</td>
      <td>0.539249</td>
      <td>0.340493</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.061490</td>
      <td>2.048820</td>
      <td>0.164339</td>
      <td>0.254802</td>
      <td>0.280882</td>
      <td>0.145360</td>
      <td>0.325720</td>
      <td>0.537912</td>
      <td>0.339806</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.039664</td>
      <td>2.048193</td>
      <td>0.163467</td>
      <td>0.254878</td>
      <td>0.280731</td>
      <td>0.145162</td>
      <td>0.326799</td>
      <td>0.537569</td>
      <td>0.339587</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.040556</td>
      <td>2.046022</td>
      <td>0.163391</td>
      <td>0.254259</td>
      <td>0.280130</td>
      <td>0.144817</td>
      <td>0.327564</td>
      <td>0.536867</td>
      <td>0.338995</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.050709</td>
      <td>2.045427</td>
      <td>0.164144</td>
      <td>0.254700</td>
      <td>0.280263</td>
      <td>0.144923</td>
      <td>0.325844</td>
      <td>0.536693</td>
      <td>0.338860</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.051194</td>
      <td>2.044928</td>
      <td>0.163854</td>
      <td>0.254556</td>
      <td>0.280258</td>
      <td>0.144970</td>
      <td>0.325792</td>
      <td>0.536503</td>
      <td>0.338995</td>
      <td>08:20</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.055931</td>
      <td>2.045213</td>
      <td>0.163834</td>
      <td>0.254476</td>
      <td>0.280212</td>
      <td>0.144805</td>
      <td>0.326275</td>
      <td>0.536823</td>
      <td>0.338788</td>
      <td>08:20</td>
    </tr>
  </tbody>
</table>



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_50_1.png" alt="">

Look at the quality difference, we can really tell that the model has made an effort to outline the eyes and nose, not the tongue because after we did the crappfication, the tongue pixel is completely cut off, can't really blame that on the model.

The goal was to generate an image that is not only of a higher quality, but also outline the key features of the image, we can safely assume that we have done it for the Oxford pets data in this case.

## Test

Let's see how this model would perform on the medium resolution images.


```python
learn = None
gc.collect();
```


```python
256/320*1024
```




    819.2




```python
256/320*1600
```




    1280.0




```python
free = gpu_mem_get_free_no_cache()
# the max size of the test image depends on the available GPU RAM
if free > 8000: size=(1280, 1600) # >  8GB RAM
else:           size=( 820, 1024) # <= 8GB RAM
print(f"using size={size}, have {free}MB of GPU RAM free")
```

    using size=(1280, 1600), have 14163MB of GPU RAM free



```python
learn = unet_learner(data, arch, loss_func=F.l1_loss, blur=True, norm_type=NormType.Weight)
```


```python
data_mr = (ImageImageList.from_folder(path_mr).split_by_rand_pct(0.1, seed=42)
          .label_from_func(lambda x: path_hr/x.name)
          .transform(get_transforms(), size=size, tfm_y=True)
          .databunch(bs=1).normalize(imagenet_stats, do_y=True))
data_mr.c = 3
```


```python
learn.load('2b');
```


```python
learn.data = data_mr
```


```python
fn = data_mr.valid_ds.x.items[0]; fn
```




    PosixPath('/root/.fastai/data/oxford-iiit-pet/small-256/newfoundland_50.jpg')



Alright, let's assign the mid-res image to variable name `img`.




```python
img = open_image(fn); img.shape
```




    torch.Size([3, 255, 306])



Use the model that we have built to predict/generate on the mid-res image, and call it `img_hr`.


```python
p,img_hr,b = learn.predict(img)
```


```python
show_image(img, figsize=(18,15), interpolation='nearest');
```


<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_67_0.png" alt="">

If we zoom in on the mid-res image above, we will very clearly see the pixel block by block of the image, and that tells us how bad the resolution of the image is, so after putting that image into our model and generate a new one...


```python
Image(img_hr).show(figsize=(18,15))
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



<img src="{{ site.url }}{{ site.baseurl }}/assets/images/pets-superres/output_69_1.png" alt="">

Zoom in and see, you will see that the picture is in a much finer detail and quality, and all that is done by using a **U-Net** and a **Perceptual Loss** CNN.

## Style Transfer

In addition to that, there is also another application for using this model, that is style transfer, yes it is exactly what it sounded like, you can transfer the style of an image to another image, so there is a deep learning project that revolves around this called [DeOldify](https://github.com/jantic/DeOldify), it's basically trying to colour grayscale images, let's see some amazing work of this project.

On the left we have the input, and on the right, we have the model prediction (generated images).

<img src="https://i.imgur.com/lNXObKY.png" width="900">

<div align="center">
<i> Thanksgiving Maskers (1911) </i>
</div>

<br>

<img src="https://i.imgur.com/RSyFf1o.png" width="900">

<div align="center">
<i> Terrasse de café, Paris (1925) </i>
</div>

<br>

<img src="https://i.imgur.com/2GuQsvc.png" width="900">

<div align="center">
<i> Chinese Opium Smokers (1980) </i>
</div>

Impressive isn't it? If you ever want to build your own image filter for your Instagram images (because you are so cool), make sure you use [DeOldify](https://github.com/jantic/DeOldify)!

Alright, enough photoshop magic for now, thank you for reading, have a nice one!
