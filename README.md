# Disparity-Network

A stereo disparity network for depth from binocular image pairs.

## What this is

Your two eyes see the world from slightly different places. A near object shifts more between the left and right views than a far one. That shift is called disparity, and your brain uses it to judge depth.

This project is a small neural network (a set of connected artificial neurons that turns inputs into outputs), written in C++ with no libraries, that tries to learn disparity from one row of a left image and the same row of a right image. Nobody gives it the right answers. It is trained to find an output that changes smoothly from one place to the next along the row, while still varying across the row as a whole. The idea is that disparity behaves this way on real surfaces.

When it runs, it trains for 100 rounds, prints the score it is trying to raise after each round, and writes its 1,000 outputs to `outputs.csv`. The score goes up every round. Whether the outputs match the true disparity has not been checked (see [Project status](#project-status)).

## What you can learn from it

- Biology and Physics: binocular vision, and how disparity relates to depth.
- Computer Science: learning without labels (unsupervised learning), where the network gets a goal to aim for instead of right answers.
- Computer Science: how a neural network is built from neurons, weights and layers, in plain C++.
- Maths: weighted moving averages with an exponential kernel, and variance.
- Computer Science: how a type such as `long` can silently throw away the fractional part of a number, and the bugs that causes. This project had that bug until 2026; the fix is in its Git history.
- Maths: checking a derivative by nudging a number up and down and measuring the change (a finite difference).

## Run it

### On your own computer

**You will need:** a C++17 compiler: Xcode Command Line Tools on macOS, `g++` from `build-essential` on Ubuntu, or [MinGW-w64](https://www.mingw-w64.org/) or Visual Studio on Windows. Tested on macOS for this guide.

1. Download the code with [Git](https://git-scm.com/downloads):

   ```bash
   git clone https://github.com/godofecht/Disparity-Network.git
   cd Disparity-Network
   ```

2. Compile it. The three `.cpp` files are compiled together:

   ```bash
   g++ -std=c++17 main.cpp Network.cpp NN.cpp -o disparityNet
   ```

3. Run it from the same folder, so it can find `image1.csv` and `image2.csv`:

   ```bash
   ./disparityNet
   ```

It finishes in about 15 seconds. It prints 101 lines. Each line is a round number and the score F for that round, starting with round 0 before any training. F should rise on every line, for example from about 0.09 to about 0.16. It then overwrites `outputs.csv` with a column headed `Values` and 1,000 numbers. The numbers change from run to run because the starting weights are random.

4. Optional: check that the training maths is right. This compares the slope that training uses for each weight with a slope measured directly, by nudging the weight up and down and recomputing F:

   ```bash
   g++ -std=c++17 gradient_check.cpp Network.cpp NN.cpp -o gradientCheck
   ./gradientCheck
   ```

   The last line should report a worst relative error below 0.0001 (about 0.0000001 in testing).

## How it works

**The input.** `image1.csv` and `image2.csv` each hold one row of 7,000 pixel values, from the left and right images. The program cuts the row into 1,000 windows. Window number `a` takes 5 pixels from the left row and the 5 pixels at the same positions in the right row, starting at position 7 x `a`.

**The network.** Each window's 10 values go into a network with 10 inputs, 10 hidden neurons and 1 output. A neuron adds up its inputs, each multiplied by a weight (a number that sets how much that input matters). Hidden neurons pass the total through tanh, which squashes it into the range -1 to 1. The output neuron does not. The same weights are used for all 1,000 windows, so the program gets 1,000 outputs, one per window.

**The goal.** For each window, the program works out two weighted averages of the outputs around it:

- a short-range average, which mostly looks at nearby windows (half-life 32 windows)
- a long-range average, which looks across almost the whole row (half-life 3,200 windows)

U measures how far the outputs are from the short-range average, and V measures how far they are from the long-range average. The program tries to make F = log(V / U) as large as possible. A large F means the output barely changes between neighbouring windows (small U) but differs a lot across the row (large V). Disparity on a real surface usually behaves like that, so the hope is that the network discovers disparity on its own.

**Training.** Each round, the program works out how F changes with each output. For each window in turn it runs the network again and passes that window's slope back through the network (backpropagation). That gives the slope of F for every weight, summed over all 1,000 windows. Each weight then moves a small step uphill. The step size (learning rate) starts at 0.01. If a step lowers F, the program undoes it and halves the learning rate. If a step raises F, it keeps it and makes the learning rate 10% larger.

| File | What is in it |
| --- | --- |
| `main.cpp` | runs `NUM_EPOCHS` (100) training rounds and writes `outputs.csv` |
| `disparityNet.h` | `DisparityNet`: reads the CSV files, builds each window's input, runs one training round, writes CSV |
| `model.h` | `Model`: sets up the network shape, which `DisparityNet` builds on |
| `Network.h`, `Network.cpp` | the network, the averaging kernels, U, V, F and their slopes |
| `NN.h`, `NN.cpp` | the neuron: weights, tanh, weight updates |
| `gradient_check.cpp` | checks the backpropagated slopes against measured ones |
| `image1.csv`, `image2.csv` | the left and right image rows |
| `image_old/` | an earlier, larger pair of image rows |
| `outputs.csv` | outputs from one run of the current code |

## Ideas to try

1. Break the maths on purpose. In `Network.cpp`, in `getDVDX`, change `- (a == k ? 1.0 : 0.0)` to `+ (a == k ? 1.0 : 0.0)`. Rebuild `gradientCheck` and run it. What does it report, and what happens to F when you train?
2. In `main.cpp`, change `NUM_EPOCHS` and plot F against the round number.
3. In `Network.h`, change the half-lives `h_U` and `h_V`. How do they affect U and V?
4. Plot `outputs.csv` in a spreadsheet or Python. Does the output change smoothly along the row?
5. In `disparityNet.h`, `Train` adapts the learning rate. Replace it with a fixed learning rate and watch F. Why does it stop rising every round?
6. In `disparityNet.h`, `getInputVector` takes 5 pixels from each image every 7 positions. Try wider windows. Remember to change the first layer size in the constructor to match.

## Technical details

```text
Network      -> feedForward, backPropagate, applyWeightChanges, UpdateWeights, NormalizeWeights
Model        -> SetTopology, InitializeTopology
DisparityNet -> readCSV, getInputVector, CalcObjective, AccumulateGradient, Train, write_csv
```

- Everything under the network is written out by hand: `NN` holds the neuron and layer types, `Network` does the feed-forward and backward passes and owns the weight update, and `Model` wraps topology setup so a task can subclass it. `DisparityNet` is that subclass.
- `UpdateWeights` and `NormalizeWeights` are declared in `Network.h` but have no body. `Model::UpdateWeights` calls `UpdateWeights`, and the program still links because nothing calls `Model::UpdateWeights`.
- Weights start as random numbers between -1 and 1. At construction the full weight list is scaled to length 1. There is no further normalisation during training.
- There are no bias neurons. The output neuron is linear.
- The averages are convolutions with a normalised kernel exp(-lambda x^2), with lambda = ln 2 / h, a window of 4h on each side capped at 499, and offsets that wrap around the 1,000 outputs (`maskWeight`).
- The slope passed back for window a is dF/dy_a = (1/V) dV/dy_a - (1/U) dU/dy_a, where dV/dy_a = sum over k of (ybar_k - y_k)(m(a - k) - [a = k]), and likewise for U. Weight changes are summed over all windows and applied once per round (batch gradient ascent).
- The image values have mean 0 and standard deviation 1, so they look standardised. Each CSV row is wrapped in `"` characters, which `readCSV` removes.
- [Iterative-Disparity-Network](https://github.com/godofecht/Iterative-Disparity-Network) is the version that replaces gradient descent with conjugate gradient and adds an R reference implementation to check against.

## Project status

This is a 2021 experiment. The build and the training rule were fixed in 2026, and training now raises F on every round. In three test runs of 100 rounds, F went from about 0.08 to about 0.17. A run of 1,000 rounds reached 0.19 and was still rising slowly.

Known limits:

- There are no correct disparity values to compare against, so it is not known whether the outputs follow the true disparity. A rough check against a disparity estimated directly from the two rows found little agreement after 100 rounds.
- `learningrule.h` is empty apart from `#include` lines, and `UpdateWeights` and `NormalizeWeights` have no body.
- There is no screenshot or plot of the results.

## Licence

You can use, copy, change and share this code for any purpose, as long as you keep the copyright notice. See [LICENSE](LICENSE) (MIT).

## Credits

- Written by Abhishek Shivakumar.
