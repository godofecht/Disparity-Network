# Disparity-Network

A stereo disparity network for depth from binocular image pairs.

## What this is

Your two eyes see the world from slightly different places. A near object shifts more between the left and right views than a far one. That shift is called disparity, and your brain uses it to judge depth.

This project is a small neural network (a set of connected artificial neurons that turns inputs into outputs), written in C++ with no libraries, that tries to learn disparity from one row of a left image and the same row of a right image. Nobody gives it the right answers. It is trained to find an output that changes smoothly from one place to the next along the row, while still varying across the row as a whole. The idea is that disparity behaves this way on real surfaces.

When it runs, it trains for 100 rounds, prints one number per round, and writes its 1,000 outputs to `outputs.csv`. The training rule has a bug at present (see [Project status](#project-status)), so the outputs are not yet a trained disparity map.

## What you can learn from it

- Biology and Physics: binocular vision, and how disparity relates to depth.
- Computer Science: learning without labels (unsupervised learning), where the network gets a goal to aim for instead of right answers.
- Computer Science: how a neural network is built from neurons, weights and layers, in plain C++.
- Maths: weighted moving averages with an exponential kernel, and variance.
- Computer Science: how a type such as `long` can silently throw away the fractional part of a number, and the bugs that causes.

## Run it

### On your own computer

**You will need:** a C++17 compiler: Xcode Command Line Tools on macOS, `g++` from `build-essential` on Ubuntu, or [MinGW-w64](https://www.mingw-w64.org/) or Visual Studio on Windows. Tested on macOS for this guide.

1. Download the code with [Git](https://git-scm.com/downloads):

   ```bash
   git clone https://github.com/godofecht/Disparity-Network.git
   cd Disparity-Network
   ```

2. Compile it. Only `main.cpp` is passed to the compiler, because the other source files are pulled in with `#include`:

   ```bash
   g++ -std=c++17 main.cpp -o disparityNet
   ```

3. Run it from the same folder, so it can find `image1.csv` and `image2.csv`:

   ```bash
   ./disparityNet
   ```

It finishes in a few seconds. It prints 100 lines, one per training round, and overwrites `outputs.csv` with a column headed `Values` and 1,000 numbers. With the current code every printed line is `0`, which is the bug described in Project status. The numbers in `outputs.csv` change from run to run because the starting weights are random.

## How it works

**The input.** `image1.csv` and `image2.csv` each hold one row of 7,000 pixel values, from the left and right images. The program cuts the row into 1,000 windows. Window number `a` takes 5 pixels from the left row and the 5 pixels at the same positions in the right row, starting at position 7 x `a`.

**The network.** Each window's 10 values go into a network with 10 inputs, 10 hidden neurons and 1 output. A neuron adds up its inputs, each multiplied by a weight (a number that sets how much that input matters). Hidden neurons pass the total through tanh, which squashes it into the range -1 to 1. The output neuron does not. The same weights are used for all 1,000 windows, so the program gets 1,000 outputs, one per window.

**The goal.** For each window, the program works out two weighted averages of the outputs around it:

- a short-range average, which mostly looks at nearby windows (half-life 32 windows)
- a long-range average, which looks across almost the whole row (half-life 3,200 windows)

U measures how far the outputs are from the short-range average, and V measures how far they are from the long-range average. The program tries to make F = log(V / U) as large as possible. A large F means the output barely changes between neighbouring windows (small U) but differs a lot across the row (large V). Disparity on a real surface usually behaves like that, so the hope is that the network discovers disparity on its own.

**Training.** Each round, the program works out how F changes with each output, passes that back through the network (backpropagation), and changes each weight by a small step (learning rate 0.001).

| File | What is in it |
| --- | --- |
| `main.cpp` | runs `NUM_EPOCHS` (100) training rounds and writes `outputs.csv` |
| `disparityNet.h` | `DisparityNet`: reads the CSV files, builds each window's input, runs one training round, writes CSV |
| `model.h` | `Model`: sets up the network shape, which `DisparityNet` builds on |
| `Network.h`, `Network.cpp` | the network, the averaging kernels, U, V, F and their slopes |
| `NN.h`, `NN.cpp` | the neuron: weights, tanh, weight updates |
| `image1.csv`, `image2.csv` | the left and right image rows |
| `image_old/` | an earlier, larger pair of image rows |
| `outputs.csv` | outputs from an earlier run |

## Ideas to try

1. Fix the bug. In `Network.h`, `CalcU` and `CalcV` store their sums in `long`, a whole-number type, so any value below 1 becomes 0. Change those `long` variables to `double`, and do the same in `getDUDX`, `getDVDX` and `getErrorDerivative` in `Network.cpp`. Then print U, V and F in `CalcF`. In a test run with this change, U and V became real numbers, but F settled near -3.76 and did not rise. Can you find out why?
2. In `main.cpp`, change `NUM_EPOCHS` and plot F against the round number.
3. In `Network.h`, change the half-lives `h_U` and `h_V`. How do they affect U and V?
4. Plot `outputs.csv` in a spreadsheet or Python. Does the output change smoothly along the row?
5. In `NN.cpp`, the learning rate is the number `0.001` in `updateInputWeights`. Try larger and smaller values once idea 1 is done.
6. In `disparityNet.h`, `getInputVector` takes 5 pixels from each image every 7 positions. Try wider windows. Remember to change the first layer size in the constructor to match.

## Technical details

```text
Network      -> feedForward, backPropagate, UpdateWeights, NormalizeWeights
Model        -> SetTopology, InitializeTopology
DisparityNet -> readCSV, getInputVector, Train, write_csv
```

- Everything under the network is written out by hand: `NN` holds the neuron and layer types, `Network` does the feed-forward and backward passes and owns the weight update, and `Model` wraps topology setup so a task can subclass it. `DisparityNet` is that subclass.
- `UpdateWeights` and `NormalizeWeights` are declared in `Network.h` but have no body. `Model::UpdateWeights` calls `UpdateWeights`, and the program still links because nothing calls `Model::UpdateWeights`.
- Weights start as random numbers between -1 and 1. At construction the full weight list is divided by its sum (`normalizeVector`). There is no further normalisation during training.
- There are no bias neurons. The output neuron is linear.
- The averages are convolutions with a normalised kernel exp(-lambda x^2), with lambda = ln 2 / h, a window of 4h on each side capped at 499, and indices that wrap around the 1,000 outputs.
- The error passed back is the sum over all windows of (1/V) dV/dy - (1/U) dU/dy. It is applied to the output neuron's gradient directly.
- The image values have mean 0 and standard deviation 1, so they look standardised. Each CSV starts with a `"` character, so the first value is read as 0.
- [Iterative-Disparity-Network](https://github.com/godofecht/Iterative-Disparity-Network) is the version that replaces gradient descent with conjugate gradient and adds an R reference implementation to check against.

## Project status

This is an unfinished 2021 experiment. Known problems:

- `CalcU`, `CalcV`, `getDUDX`, `getDVDX` and `getErrorDerivative` use `long`, so U is always 0. That makes F = log(V / 0), the error terms become infinite or undefined, and training does not work. The program prints U each round, which is why every line is `0`.
- The earlier build command `g++ -std=c++17 main.cpp NN.cpp -o disparityNet` fails with "duplicate symbol" errors, because `NN.cpp` is already included through `Network.cpp`. Use the command in Run it.
- `learningrule.h` is empty apart from `#include` lines.
- There is no screenshot or plot of the results, and no correct disparity values to compare against.

## Licence

You can use, copy, change and share this code for any purpose, as long as you keep the copyright notice. See [LICENSE](LICENSE) (MIT).

## Credits

- Written by Abhishek Shivakumar.
