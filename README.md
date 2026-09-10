# Disparity-Network

A small neural network that learns stereo disparity from a pair of image rows,
written in C++ with no libraries.

The network is 10 inputs, 10 hidden units, 1 output. Each training example takes
five neighbouring pixels from the left image and the five that correspond to
them in the right image, and the single output is the disparity between them.
Images arrive as `image1.csv` and `image2.csv`, one row per image.

Everything under the network is written out by hand: `NN` holds the neuron and
layer types, `Network` does the feed-forward and backward passes and owns the
weight update, and `Model` wraps topology setup so a task can subclass it.
`DisparityNet` is that subclass.

```
Network -> feedForward, backPropagate, UpdateWeights, NormalizeWeights
Model   -> SetTopology, InitializeTopology
DisparityNet -> readCSV, getInputVector, Train, write_csv
```

Weights are normalised at construction and again during training, which matters
here because the input is raw pixel intensity rather than anything scaled.

## Run

```bash
g++ -std=c++17 main.cpp NN.cpp -o disparityNet
./disparityNet
```

100 epochs by default, set by `NUM_EPOCHS` in `main.cpp`. Output goes to
`outputs.csv`.

See [Iterative-Disparity-Network](https://github.com/godofecht/Iterative-Disparity-Network)
for the version that replaces gradient descent with conjugate gradient and adds
an R reference implementation to check against.
