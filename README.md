# Ruby Autoencoder

Toy autoencoder built with [torch-rb](https://github.com/ankane/torch.rb).

## Setup
[Setup torch rb](https://github.com/ankane/torch.rb?tab=readme-ov-file#installation) then
`bundle`.

## Train
Trains on the fashion mnist dataset.

```
bundle exec ruby train.rb
```

## Combine images
```
bundle exec ruby combine.rb pullover shirt
```

Available labels

- t_shirt
- trouser
- pullover
- dress
- coat
- sandal
- shirt
- sneaker
- bag
- ankle_boot

## Results
With 5 layers in encoder / decoder  
784 -> 512 -> 256 -> 128 -> 64 -> 32

### Combining embeddings
#### shirt + t_shirt
**shirt**

<img src="docs/bafpjy--shirt-a.png" alt="shirt" width="50%">

**t_shirt**

<img src="docs/bafpjy--t_shirt-b.png" alt="t_shirt" width="50%">

**combined**

<img src="docs/bafpjy--combined-shirt-t_shirt.png" alt="combined" width="50%">

### sandal + sneaker
**sandal**

<img src="docs/ienzlu--sandal-a.png" alt="sandal" width="50%">

**sneaker**

<img src="docs/ienzlu--sneaker-b.png" alt="sneaker" width="50%">

**combined**

<img src="docs/ienzlu--combined-sandal-sneaker.png" alt="combined" width="50%">

### trouser + pullover
**trouser**

<img src="docs/wkrzbs--trouser-a.png" alt="trouser" width="50%">

**pullover**

<img src="docs/wkrzbs--pullover-b.png" alt="pullover" width="50%">

**combined**

<img src="docs/wkrzbs--combined-trouser-pullover.png" alt="combined" width="50%">

## Reconstruction
### Shirt
**original**

<img src="docs/umcwvt--original_shirt.png" alt="original" width="50%" />

**reconstructed**

<img src="docs/umcwvt--reconstructed_shirt.png" alt="reconstructed" width="50%" />

## TODO
Try out a convolutional network for the auto encoder, should improve output
quality.
