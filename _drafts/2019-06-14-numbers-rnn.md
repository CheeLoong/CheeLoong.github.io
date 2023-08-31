---
title: "Recurrent Neural Network on Human numbers dataset"
date: 2019-06-14
permalink: /numbers-rnn/
tags: [fastai, pytorch, rnn, human numbers, deep learning]
excerpt: "Exploring RNN on a toy example with fastai library "
mathjax: "true"
published: false
---

In this blogpost, let's learn what a Recurrent Neural Network (RNN) actually is, we will be using a toy example from fast.ai library to demonstrate this deep learning model.

# Human numbers


```python
from fastai.text import *
```


```python
  !curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    Done.


## Data

Here we are using fastai version human numbers dataset as a toy example to demonstrate creation of a language model. This dataset consist of all the numbers from 1 to 9999 written in English.

From the source we will see 2 portion of the dataset, In the training set, we have 1-7999, and the validation set starts from 8000-9999.



```python
path = untar_data(URLs.HUMAN_NUMBERS)
path.ls()
```




    [PosixPath('/root/.fastai/data/human_numbers/valid.txt'),
     PosixPath('/root/.fastai/data/human_numbers/train.txt')]




```python
def readnums(d): return [', '.join(o.strip() for o in open(path/d).readlines())]
```


```python
train_txt = readnums('train.txt'); train_txt[0][:80]
```




    'one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirt'




```python
valid_txt = readnums('valid.txt'); valid_txt[0][-80:]
```




    ' nine thousand nine hundred ninety eight, nine thousand nine hundred ninety nine'




```python
# set batchsize
bs = 64
```

When we set batchsize `bs = 64`, it doesn't mean each batch is of length 64, but rather we have 64 roughly equally sized batch.

As per usual dealing with fastai, we need to convert our dataset to `DataBunch`. In this case, we start by converting them to `TextList`, then put them together as `ItemLists` and finally as `DataBunch` with `bs=64`


```python
train = TextList(train_txt, path=path)
valid = TextList(valid_txt, path=path)

src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
data = src.databunch(bs=bs)
```


```python
train[0].text[:80]
```




    'xxbos one , two , three , four , five , six , seven , eight , nine , ten , eleve'




```python
valid[0].text[-80:]
```




    'nine thousand nine hundred ninety eight , nine thousand nine hundred ninety nine'




```python
len(data.train_ds[0][0].data)
```




    50079




```python
len(data.valid_ds[0][0].data)
```




    13017



there are `50079` tokens in the train set and `13017` tokens in the valid set.


```python
data.bptt, len(data.valid_dl)
```




    (70, 3)



`data.bptt` refers to *backprops through time*, which is the sequence length, that means for each of the 64 batches from the document, we then split each of the 64 batches into pieces of `70`.


```python
13017/70/bs
```




    2.905580357142857



which works out to be about 3 batches.


```python
it = iter(data.valid_dl)
x1,y1 = next(it)
x2,y2 = next(it)
x3,y3 = next(it)
it.close()
```


```python
x1.numel()+x2.numel()+x3.numel()
```




    13440



Notice that in the validation set, we had `13017 tokens`, but we now have `13440` tokens, this is fastai library customization to ensure that each of the 3 batches have the same batch size and *back props through time*.

We will soon see why we have more tokens after we split it into 3 batches.


```python
print(x1.shape,y1.shape)
print(x2.shape,y2.shape)
print(x3.shape,y3.shape)
```

    torch.Size([64, 70]) torch.Size([64, 70])
    torch.Size([64, 70]) torch.Size([64, 70])
    torch.Size([64, 70]) torch.Size([64, 70])



```python
print(x1)
print(x2)
print(x3)
```

    tensor([[ 2, 19, 11,  ..., 36,  9, 19],
            [ 9, 19, 11,  ..., 24, 20,  9],
            [11, 27, 18,  ...,  9, 19, 11],
            ...,
            [20, 11, 20,  ..., 11, 20, 10],
            [20, 11, 20,  ..., 24,  9, 20],
            [20, 10, 26,  ..., 20, 11, 20]], device='cuda:0')
    tensor([[11, 37,  9,  ..., 22, 13,  9],
            [19, 11, 25,  ...,  9, 19, 11],
            [12, 10, 12,  ..., 10, 31,  9],
            ...,
            [21, 12,  9,  ..., 20, 10, 22],
            [11, 20, 10,  ...,  9, 20, 11],
            [10, 27,  9,  ..., 11, 20, 10]], device='cuda:0')
    tensor([[19, 11, 22,  ..., 17,  9, 19],
            [26, 15,  9,  ..., 19, 11, 27],
            [19, 11, 12,  ..., 14,  9, 19],
            ...,
            [12,  9, 20,  ..., 10, 23, 12],
            [20, 10, 25,  ..., 20, 11, 20],
            [28,  9, 20,  ..., 12,  9, 19]], device='cuda:0')



```python
print(y1)
print(y2)
print(y3)
```

    tensor([[19, 11, 12,  ...,  9, 19, 11],
            [19, 11, 23,  ..., 20,  9, 19],
            [27, 18,  9,  ..., 19, 11, 12],
            ...,
            [11, 20, 10,  ..., 20, 10, 21],
            [11, 20, 10,  ...,  9, 20, 11],
            [10, 26,  9,  ..., 11, 20, 10]], device='cuda:0')
    tensor([[37,  9, 19,  ..., 13,  9, 19],
            [11, 25,  9,  ..., 19, 11, 26],
            [10, 12,  9,  ..., 31,  9, 19],
            ...,
            [12,  9, 20,  ..., 10, 22, 12],
            [20, 10, 24,  ..., 20, 11, 20],
            [27,  9, 20,  ..., 20, 10, 28]], device='cuda:0')
    tensor([[11, 22, 14,  ...,  9, 19, 11],
            [15,  9, 19,  ..., 11, 27, 19],
            [11, 12, 10,  ...,  9, 19, 11],
            ...,
            [ 9, 20, 11,  ..., 23, 12,  9],
            [10, 25, 12,  ..., 11, 20, 10],
            [ 9, 20, 11,  ...,  9, 19, 11]], device='cuda:0')


Let's examine closer by looking at the very first mini-batch of `x1` and `y1`


```python
x1[0]
```




    tensor([ 2, 19, 11, 12,  9, 19, 11, 13,  9, 19, 11, 14,  9, 19, 11, 15,  9, 19,
            11, 16,  9, 19, 11, 17,  9, 19, 11, 18,  9, 19, 11, 19,  9, 19, 11, 20,
             9, 19, 11, 29,  9, 19, 11, 30,  9, 19, 11, 31,  9, 19, 11, 32,  9, 19,
            11, 33,  9, 19, 11, 34,  9, 19, 11, 35,  9, 19, 11, 36,  9, 19],
           device='cuda:0')




```python
y1[0]
```




    tensor([19, 11, 12,  9, 19, 11, 13,  9, 19, 11, 14,  9, 19, 11, 15,  9, 19, 11,
            16,  9, 19, 11, 17,  9, 19, 11, 18,  9, 19, 11, 19,  9, 19, 11, 20,  9,
            19, 11, 29,  9, 19, 11, 30,  9, 19, 11, 31,  9, 19, 11, 32,  9, 19, 11,
            33,  9, 19, 11, 34,  9, 19, 11, 35,  9, 19, 11, 36,  9, 19, 11],
           device='cuda:0')



`y1` is basically `x1` offset by 1, because in a language model we want to predict the next word.

We can also grab the `.vocab` of the validation set, to make use of `.textify` to look at our batches in text.


```python
v = data.valid_ds.vocab
```


```python
# first mini-batch of x1
v.textify(x1[0])
```




    'xxbos eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve , eight thousand thirteen , eight thousand fourteen , eight thousand fifteen , eight thousand sixteen , eight thousand seventeen , eight'




```python
# first mini-batch of y1
v.textify(y1[0])
```




    'eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve , eight thousand thirteen , eight thousand fourteen , eight thousand fifteen , eight thousand sixteen , eight thousand seventeen , eight thousand'




```python
# first mini-batch of x2
v.textify(x2[0])
```




    'thousand eighteen , eight thousand nineteen , eight thousand twenty , eight thousand twenty one , eight thousand twenty two , eight thousand twenty three , eight thousand twenty four , eight thousand twenty five , eight thousand twenty six , eight thousand twenty seven , eight thousand twenty eight , eight thousand twenty nine , eight thousand thirty , eight thousand thirty one , eight thousand thirty two ,'




```python
# first mini-batch of x3
v.textify(x3[0])
```




    'eight thousand thirty three , eight thousand thirty four , eight thousand thirty five , eight thousand thirty six , eight thousand thirty seven , eight thousand thirty eight , eight thousand thirty nine , eight thousand forty , eight thousand forty one , eight thousand forty two , eight thousand forty three , eight thousand forty four , eight thousand forty five , eight thousand forty six , eight'



Okay, let's put everything here to see how the 3 batches of `x` were composed;

x1[0]: 8001 ~ 8017

x2[0]: 8018 ~ 8032

x3[0]: 8033 ~ 8046

Let us guess, x1[1] should start from 8047?




```python
# second mini-batch of x1
v.textify(x1[1])
```




    ', eight thousand forty six , eight thousand forty seven , eight thousand forty eight , eight thousand forty nine , eight thousand fifty , eight thousand fifty one , eight thousand fifty two , eight thousand fifty three , eight thousand fifty four , eight thousand fifty five , eight thousand fifty six , eight thousand fifty seven , eight thousand fifty eight , eight thousand fifty nine ,'



Not quite, it was 8046, although it already appeared in x3[0], why does this appear twice? This is to ensure that each of the batch of `x` are linked, without doing this, x3[0] might stop at 8046, but x3[1] would start from 8050 or something, this is the reason why we have more tokens here than the original validation set.


```python
v.textify(x2[1])
```




    'eight thousand sixty , eight thousand sixty one , eight thousand sixty two , eight thousand sixty three , eight thousand sixty four , eight thousand sixty five , eight thousand sixty six , eight thousand sixty seven , eight thousand sixty eight , eight thousand sixty nine , eight thousand seventy , eight thousand seventy one , eight thousand seventy two , eight thousand seventy three , eight thousand'




```python
v.textify(x3[1])
```




    'seventy four , eight thousand seventy five , eight thousand seventy six , eight thousand seventy seven , eight thousand seventy eight , eight thousand seventy nine , eight thousand eighty , eight thousand eighty one , eight thousand eighty two , eight thousand eighty three , eight thousand eighty four , eight thousand eighty five , eight thousand eighty six , eight thousand eighty seven , eight thousand eighty'



x1[1] : 8046 ~ 8059

x2[1] : 8060 ~ 8073

x3[1] : 8074 ~ 8087

and one last check on the very last mini-batch of the last batch;

x3[-1] : 9991 ~ 9999


```python
v.textify(x3[-1])
```




    'ninety , nine thousand nine hundred ninety one , nine thousand nine hundred ninety two , nine thousand nine hundred ninety three , nine thousand nine hundred ninety four , nine thousand nine hundred ninety five , nine thousand nine hundred ninety six , nine thousand nine hundred ninety seven , nine thousand nine hundred ninety eight , nine thousand nine hundred ninety nine xxbos eight thousand one , eight'




```python
data.show_batch(ds_type=DatasetType.Valid)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>idx</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>thousand forty seven , eight thousand forty eight , eight thousand forty nine , eight thousand fifty , eight thousand fifty one , eight thousand fifty two , eight thousand fifty three , eight thousand fifty four , eight thousand fifty five , eight thousand fifty six , eight thousand fifty seven , eight thousand fifty eight , eight thousand fifty nine , eight thousand sixty , eight thousand sixty</td>
    </tr>
    <tr>
      <td>1</td>
      <td>eight , eight thousand eighty nine , eight thousand ninety , eight thousand ninety one , eight thousand ninety two , eight thousand ninety three , eight thousand ninety four , eight thousand ninety five , eight thousand ninety six , eight thousand ninety seven , eight thousand ninety eight , eight thousand ninety nine , eight thousand one hundred , eight thousand one hundred one , eight thousand one</td>
    </tr>
    <tr>
      <td>2</td>
      <td>thousand one hundred twenty four , eight thousand one hundred twenty five , eight thousand one hundred twenty six , eight thousand one hundred twenty seven , eight thousand one hundred twenty eight , eight thousand one hundred twenty nine , eight thousand one hundred thirty , eight thousand one hundred thirty one , eight thousand one hundred thirty two , eight thousand one hundred thirty three , eight thousand</td>
    </tr>
    <tr>
      <td>3</td>
      <td>three , eight thousand one hundred fifty four , eight thousand one hundred fifty five , eight thousand one hundred fifty six , eight thousand one hundred fifty seven , eight thousand one hundred fifty eight , eight thousand one hundred fifty nine , eight thousand one hundred sixty , eight thousand one hundred sixty one , eight thousand one hundred sixty two , eight thousand one hundred sixty three</td>
    </tr>
    <tr>
      <td>4</td>
      <td>thousand one hundred eighty three , eight thousand one hundred eighty four , eight thousand one hundred eighty five , eight thousand one hundred eighty six , eight thousand one hundred eighty seven , eight thousand one hundred eighty eight , eight thousand one hundred eighty nine , eight thousand one hundred ninety , eight thousand one hundred ninety one , eight thousand one hundred ninety two , eight thousand</td>
    </tr>
  </tbody>
</table>


## Single fully connected model

In the previous case, we use default `bptt = 70`, and we had 3 batches of `[64, 70]` tensor which makes up the entire dataset, now we are changing `bptt = 3` and taking only one batch


```python
# Let's change backprop through time to 3
data = src.databunch(bs=bs, bptt=3)
```


```python
x,y = data.one_batch()
x.shape,y.shape
```




    (torch.Size([64, 3]), torch.Size([64, 3]))




```python
x[0:5]
```




    tensor([[13,  9, 14],
            [13, 10, 30],
            [10, 26, 17],
            [ 9, 16, 10],
            [18, 10, 20]])




```python
x[:,0]
```




    tensor([13, 13, 10,  9, 18,  9, 11, 11, 13, 19, 16, 23, 24,  9, 12,  9, 13, 14,
            15, 11, 10, 22, 15,  9, 10, 14, 11, 16, 10, 28, 11,  9, 20,  9, 15, 15,
            11, 18, 10, 28, 23, 24,  9, 16, 10, 16, 19, 20, 12, 10, 22, 16, 17, 17,
            17, 11, 24, 10,  9, 15, 16,  9, 18, 11])




```python
data.show_batch(ds_type=DatasetType.Valid)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>idx</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>xxbos eight thousand</td>
    </tr>
    <tr>
      <td>1</td>
      <td>, eight thousand</td>
    </tr>
    <tr>
      <td>2</td>
      <td>thousand eighty seven</td>
    </tr>
    <tr>
      <td>3</td>
      <td>one hundred twenty</td>
    </tr>
    <tr>
      <td>4</td>
      <td>two , eight</td>
    </tr>
  </tbody>
</table>



```python
# each of the token is numericalized with respect to an id
for id, str in enumerate(v.itos):
  print(id, str)
```

    0 xxunk
    1 xxpad
    2 xxbos
    3 xxeos
    4 xxfld
    5 xxmaj
    6 xxup
    7 xxrep
    8 xxwrep
    9 ,
    10 hundred
    11 thousand
    12 one
    13 two
    14 three
    15 four
    16 five
    17 six
    18 seven
    19 eight
    20 nine
    21 twenty
    22 thirty
    23 forty
    24 fifty
    25 sixty
    26 seventy
    27 eighty
    28 ninety
    29 ten
    30 eleven
    31 twelve
    32 thirteen
    33 fourteen
    34 fifteen
    35 sixteen
    36 seventeen
    37 eighteen
    38 nineteen



```python
# total number of vocabs in this document
nv = len(v.itos); nv
```




    39




```python
nh=64
```


```python
def loss4(input,target): return F.cross_entropy(input, target[:,-1])
def acc4 (input,target): return accuracy(input, target[:,-1])
```

<img src="https://i.imgur.com/tIuHmd2.png" width="700">

<center><p><i>Figure 1: Predicting words 4 using words [1, 2 and 3]</i></p>
.</center>



```python
class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)  # green arrow
        self.h_h = nn.Linear(nh,nh)     # brown arrow
        self.h_o = nn.Linear(nh,nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = self.bn(F.relu(self.h_h(self.i_h(x[:,0]))))
        if x.shape[1]>1:
            h = h + self.i_h(x[:,1])
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1]>2:
            h = h + self.i_h(x[:,2])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)
```

The diagram above is a visualization of the code above;

Each coloured arrow has a single matrix, we have got input to hidden layers, hidden to hidden layers, hidden to output layers, and then a batchnorm.

---

Remember:

In our case, the `x` is a rank 2 tensor of size `[64, 3]`, that's 64 rows by 3 columns, which means 64 batches with 3 bbpt.

---

With that in mind, let's look at the `forward` code;


```
h = self.bn(F.relu(self.h_h(self.i_h(x[:,0]))))
```



x[:, 0] takes all the row and the first column in the tensor, this means we are taking all the word 1 input, chuck it through `self.i_h` (input to hidden - *green arrow*) and then `self.h_h` (hidden to hidden - *orange arrow*), followed by `relu` & `bn`.

```
        if x.shape[1]>1:
            h = h + self.i_h(x[:,1])
            h = self.bn(F.relu(self.h_h(h)))
```
If our `x` has more than 1 word input (i.e. `x.shape[1] > 1`), we will take word 2 input, chuck through `self.i_h` (input to hidden - *green arrow*) and then `self.h_h` (hidden to hidden - *orange arrow*), followed by `relu` & `bn`.

```
        if x.shape[1]>2:
            h = h + self.i_h(x[:,2])
            h = self.bn(F.relu(self.h_h(h)))
```

If `x` has more than 2 word inputs (i.e. `x.shape[1] > 2`), we do another round like above and so forth..and finally through `self.h_o` (hidden to output - *blue arrow*)

Let's train and see..






```python
learn = Learner(data, Model0(), loss_func=loss4, metrics=acc4)
```


```python
learn.fit_one_cycle(6, 1e-4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>acc4</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.598977</td>
      <td>3.563497</td>
      <td>0.096737</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.030272</td>
      <td>3.041378</td>
      <td>0.381204</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.420745</td>
      <td>2.521163</td>
      <td>0.454963</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.108515</td>
      <td>2.292764</td>
      <td>0.466682</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.989044</td>
      <td>2.216823</td>
      <td>0.467142</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.964190</td>
      <td>2.206099</td>
      <td>0.467601</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


## Same thing with a loop

If we look at how we define `class Model0` earlier, we can see duplicate codes, which we can then refactor into a loop, which is shown in the diagram and code below:


<img src="https://i.imgur.com/a5Y85Lo.png" width="700">

<center><p><i>Figure 2: Predicting words [n] using words [1 to n-1]</i></p>
.</center>



```python
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)  # green arrow
        self.h_h = nn.Linear(nh,nh)     # brown arrow
        self.h_o = nn.Linear(nh,nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)
```

And you know what is crazy? This is actually RNN, it's just refactoring.

`Model1` and `Model0` is the same, except that `Model1` is refactoring the duplicates code, and also it can now take on any arbitrary number of word inputs to predict the next word, not just 2 word inputs.


```python
# reset learn
del learn
```


```python
learn = Learner(data, Model1(), loss_func=loss4, metrics=acc4)
```


```python
learn.fit_one_cycle(6, 1e-4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>acc4</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.629552</td>
      <td>3.516481</td>
      <td>0.116039</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.059436</td>
      <td>2.995898</td>
      <td>0.429458</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.488669</td>
      <td>2.517447</td>
      <td>0.466682</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2.166466</td>
      <td>2.269247</td>
      <td>0.468290</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2.040354</td>
      <td>2.184184</td>
      <td>0.468980</td>
      <td>00:01</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2.014172</td>
      <td>2.172092</td>
      <td>0.468980</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>


and we get the same result as before.

## Multi fully connected model

Let's increase `bptt = 20` since we have refactored with a loop to predict any number of word inputs.


```python
data = src.databunch(bs=bs, bptt=20)
```


```python
x,y = data.one_batch()
x.shape,y.shape
```




    (torch.Size([64, 20]), torch.Size([64, 20]))



A little more context here, earlier in the loss function, we were looking at the result of the model to the last word of the sequence (*i.e. predicting for word n-th* with words *n-1*, now that we have a 20 words sequence, we can try to compare every word in X to every word in Y, instead of looking at only the last word of the sequence.



<img src="https://i.imgur.com/1XGtm3p.png" width="700">

<center><p><i>Figure 3: Predicting words [2 to n] using words [1 to n-1]</i></p>
.</center>




```python
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = [] # create array to append every time a loop is run
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)
```

`Model2` takes in *n inputs* and spits out *n outputs* because its predicting after every word.


```python
learn = Learner(data, Model2(), metrics=accuracy)
```


```python
learn.fit_one_cycle(10, 1e-4, pct_start=0.1)
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
      <td>3.741133</td>
      <td>3.761891</td>
      <td>0.020028</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.636408</td>
      <td>3.617405</td>
      <td>0.026634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.502551</td>
      <td>3.472571</td>
      <td>0.129190</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.364337</td>
      <td>3.345006</td>
      <td>0.184446</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.235704</td>
      <td>3.243609</td>
      <td>0.214773</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>3.127701</td>
      <td>3.172788</td>
      <td>0.238210</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>3.045259</td>
      <td>3.127753</td>
      <td>0.256747</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2.988367</td>
      <td>3.103684</td>
      <td>0.268040</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2.953725</td>
      <td>3.094491</td>
      <td>0.271236</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2.935733</td>
      <td>3.093130</td>
      <td>0.271875</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


Notice that the accuracy is worse than before? This is because we have limited state, so say if we are looking to predict for the 2nd word, we only have the 1st word of state as input, and when we are prediciting for the 3rd word, we only have the first two words of state to use.

## Maintain state


```python
torch.zeros(2,3)
```




    tensor([[0., 0., 0.],
            [0., 0., 0.]])



In `Model2` we have:



```
h = torch.zeros(x.shape[0], nh).to(device=x.device)
```
This shows that we are resetting state to zero everytime we run a BPTT sequence, instead of doing that, we will keep `h`, so we put that in the constructor.





```python
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).cuda()

    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = h.detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res
```


```python
learn = Learner(data, Model3(), metrics=accuracy)
```


```python
learn.fit_one_cycle(20, 3e-3)
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
      <td>3.617205</td>
      <td>3.560725</td>
      <td>0.058878</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.250525</td>
      <td>2.995812</td>
      <td>0.450497</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.570915</td>
      <td>2.085349</td>
      <td>0.468253</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.993188</td>
      <td>2.016490</td>
      <td>0.315554</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.697359</td>
      <td>2.022782</td>
      <td>0.318608</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.513505</td>
      <td>1.810204</td>
      <td>0.427770</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.327509</td>
      <td>1.707576</td>
      <td>0.450355</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.155951</td>
      <td>1.660483</td>
      <td>0.503125</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.009661</td>
      <td>1.610945</td>
      <td>0.490909</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.885818</td>
      <td>1.654948</td>
      <td>0.517330</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.778724</td>
      <td>1.622910</td>
      <td>0.520739</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.690318</td>
      <td>1.652979</td>
      <td>0.525071</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.620285</td>
      <td>1.615985</td>
      <td>0.534020</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.566535</td>
      <td>1.591181</td>
      <td>0.543608</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.522966</td>
      <td>1.716132</td>
      <td>0.545881</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.490084</td>
      <td>1.694666</td>
      <td>0.542401</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.464625</td>
      <td>1.680378</td>
      <td>0.549787</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.446226</td>
      <td>1.689312</td>
      <td>0.541974</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.434207</td>
      <td>1.695729</td>
      <td>0.536932</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.427399</td>
      <td>1.694964</td>
      <td>0.539915</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## nn.RNN

We can further improve the model by using PyTorch `nn.RNN`, the idea is as follows:

<img src="https://i.imgur.com/fo2okTe.png" width="700">

<center><p><i>Figure 4: Predicting words [2 to n] using words [1 to n-1] with stacked RNNs</i></p>
.</center>

At the end of every loop, we put the result to another RNN, so it's basically an RNN going into another RNN, this is basically `Model3` with more refactoring, and we are doing the loop with `nn.RNN`, everything else is the same, embeddings, output, batchnorm, initialization of h, we can also specify how many layers we want.


```python
class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.rnn = nn.RNN(nh,nh, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(1, bs, nh).cuda()

    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))
```


```python
learn = Learner(data, Model4(), metrics=accuracy)
```


```python
learn.fit_one_cycle(20, 3e-3)
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
      <td>3.464345</td>
      <td>3.295450</td>
      <td>0.283097</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.041520</td>
      <td>2.523367</td>
      <td>0.464276</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.385418</td>
      <td>1.919809</td>
      <td>0.467401</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.896835</td>
      <td>1.947503</td>
      <td>0.316051</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.622708</td>
      <td>1.748680</td>
      <td>0.475994</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.417991</td>
      <td>1.537231</td>
      <td>0.518040</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.232437</td>
      <td>1.507867</td>
      <td>0.503977</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.067902</td>
      <td>1.477712</td>
      <td>0.528622</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.930517</td>
      <td>1.590528</td>
      <td>0.525710</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.817504</td>
      <td>1.629914</td>
      <td>0.543395</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.721151</td>
      <td>1.717034</td>
      <td>0.551421</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.635008</td>
      <td>1.768327</td>
      <td>0.565980</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.561895</td>
      <td>1.764571</td>
      <td>0.573438</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.500028</td>
      <td>1.773710</td>
      <td>0.565980</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.451725</td>
      <td>1.622007</td>
      <td>0.579688</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.414054</td>
      <td>1.640178</td>
      <td>0.583523</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.386020</td>
      <td>1.658984</td>
      <td>0.578338</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.365691</td>
      <td>1.688034</td>
      <td>0.569957</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.352249</td>
      <td>1.675685</td>
      <td>0.575497</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.344502</td>
      <td>1.688391</td>
      <td>0.574219</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## 2-layer GRU

If we visualize the network without loop, it'd look like this:


<img src="https://i.imgur.com/gau3bG4.png" width="700">

If we have say a `bptt = 20`, there'd be 20 layers instead of 3 on the above figure, it will be pretty much impossible to train, to get around that, skip connections is one of the method to do it, but more commonly, instead of adding the green arrow and the orange arrow together into a hidden layer, we use a mini neural net to decide how much of the green arrow and orange arrow to keep, that kind of approach is either called GRU or LSTM depending on the details of that mini neural network. (I gotta admit that I don't exactly know the theoretical part of this, but we will eventually learn this)

So we can now say let's create a GRU instead. It's just like what we had before, but it'll handle longer sequences in deeper networks. Let's use two layers.


```python
class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.rnn = nn.GRU(nh, nh, 2, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(2, bs, nh).cuda()

    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))
```


```python
learn = Learner(data, Model5(), metrics=accuracy)
```


```python
learn.fit_one_cycle(10, 1e-2)
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
      <td>3.081594</td>
      <td>2.520581</td>
      <td>0.441193</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.885889</td>
      <td>1.642362</td>
      <td>0.514844</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.964905</td>
      <td>1.070687</td>
      <td>0.798438</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.468619</td>
      <td>0.928140</td>
      <td>0.832955</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.232655</td>
      <td>0.982595</td>
      <td>0.836008</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.120602</td>
      <td>1.003022</td>
      <td>0.837642</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.066430</td>
      <td>0.999758</td>
      <td>0.838921</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.038846</td>
      <td>1.072193</td>
      <td>0.837500</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.024711</td>
      <td>1.048354</td>
      <td>0.839134</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.017584</td>
      <td>1.053584</td>
      <td>0.838992</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


Look at that sweet performance of the RNN with 2-layer GRU, that is all for this toy example, but this is also commonly used for [sequence labeling task](https://en.wikipedia.org/wiki/Sequence_labeling) which is beyond the scope of this blogpost. Thank you for reading!
