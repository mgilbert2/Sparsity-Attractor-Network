# Sparsity Attractor Network

A model of AB/AC paired-associate learning. The goal is a three-layer network with a Hopfield network as the hidden layer. For now the hidden layer is an ordinary layer, so the network is a plain autoencoder.

## The network

The input and output layers are split into three parts:

- A, the cue item
- B/C, the associated item (B on the first list, C on the second)
- L, the list context (AB or AC)

The network learns to reproduce its input on its output. To test recall, it is given A and L with the B/C part blank, and it has to fill in B or C on the output.

## Running

- `just` lists the commands
- `just run` trains and prints recall
- `just plot` does the same and saves a plot to `figures/`

`just run` first trains on all the AB pairs for 100 epochs, then on all the AC pairs for 100 epochs. After each epoch it tests every pair on both lists. The AB and AC columns show the fraction of pairs recalled correctly. With 8 pairs, 2 correct is 2/8 = 0.25.

A typical run learns AB, then forgets it while learning AC. This is catastrophic interference.
