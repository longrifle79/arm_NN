/*
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║           MNIST Neural Network Trainer — Fully Annotated Edition           ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 *
 * This file is designed to be READ as much as it is run. Every function, every
 * line, and every math concept is explained in plain English, with references to
 * pages in the books you have.
 *
 * ARCHITECTURE:
 *   Input (784) → Hidden Layer 1 (128) → Hidden Layer 2 (64) → Output (10)
 *
 * BOOK REFERENCES USED THROUGHOUT:
 *   [Rashid]  = "Make Your Own Neural Network" by Tariq Rashid
 *   [Nield]   = "Essential Math for Data Science" by Thomas Nield
 *   [Stewart] = "Calculus: Concepts and Contexts" by James Stewart
 *   [Taylor]  = "Make Your Own Neural Network (Visual Introduction)" by Michael Taylor
 *
 * BUILD:
 *   g++ -O2 -std=c++17 -o mnist_train mnist_train_annotated.cpp
 *
 * DATA FILES (place in same folder as the executable):
 *   train-images.idx3-ubyte
 *   train-labels.idx1-ubyte
 *   t10k-images.idx3-ubyte
 *   t10k-labels.idx1-ubyte
 *   Download from: http://yann.lecun.com/exdb/mnist/
 */

// ─── STANDARD LIBRARY HEADERS ────────────────────────────────────────────────
// These are the "toolboxes" C++ gives us for free. Each one unlocks a set of
// functions we'll use throughout the program.

#include <iostream>   // cout and cerr: lets us print to the screen/console
#include <fstream>    // ifstream: lets us open and read files from disk (the MNIST files)
#include <vector>     // vector<T>: a resizable array — our main data structure everywhere
#include <cmath>      // exp(), sqrt(), log(), pow(): math functions used in activations/loss
#include <algorithm>  // max_element(), shuffle(): find the biggest element, or shuffle a list
#include <numeric>    // iota(): fills a vector with sequential numbers (0, 1, 2, 3...)
#include <random>     // mt19937, normal_distribution: random number generation for weight init
#include <chrono>     // steady_clock: used to time how long each epoch takes
#include <iomanip>    // setw(), setprecision(), fixed: format numbers when printing output
#include <cassert>    // assert(): a safety check — stops the program with a message if false

using namespace std; // Lets us write "vector" instead of "std::vector" everywhere


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 1 — DATA CONTAINER
// ═══════════════════════════════════════════════════════════════════════════════

// ─── MNISTData struct ─────────────────────────────────────────────────────────
//
// A "struct" in C++ is just a named group of related variables bundled together.
// Think of it as a labelled box that holds two things at once:
//   1. The images (pixel data)
//   2. The labels (the correct answer — which digit 0-9 each image shows)
//
// WHY TWO SEPARATE VECTORS?
//   Each image is itself a list of 784 numbers (28 pixels wide × 28 pixels tall,
//   flattened into one row). So `images` is a list of lists.
//   Each label is just a single integer (0 through 9).
//   By keeping them in parallel vectors, images[i] always matches labels[i].
//
//   [Rashid, p. 144-146] — The MNIST dataset and how images are stored as pixel arrays
//   [Nield, p. 342-343]  — How input data flows into a neural network
//
struct MNISTData
{
    vector<vector<float>> images;  // Outer vector = all samples; inner vector = 784 pixel values
    vector<int>           labels;  // One integer per image: the correct digit (0-9)
};


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 2 — HYPERPARAMETERS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Hyperparameters are NOT learned by the network. YOU choose them before training
// starts. They control the shape of the network and how it learns.
//
// The word "constexpr" means these values are fixed at compile time and cannot
// change while the program runs. They're the "settings" for the experiment.
//
// [Rashid, p. 33-35, 99]  — Explanation of learning rate as a moderating factor
// [Nield, p. 254-258]      — Gradient descent and how step size (learning rate) affects convergence
// [Taylor, Ch. 4]          — Hyperparameter tuning in practice

constexpr int   INPUT_SIZE    = 784;    // 28*28 pixels per image, flattened to 1D
                                        // Each pixel is one input neuron

constexpr int   HIDDEN1_SIZE  = 128;    // First hidden layer: 128 neurons
                                        // More neurons = more capacity to learn complex patterns,
                                        // but slower and more prone to overfitting

constexpr int   HIDDEN2_SIZE  = 64;     // Second hidden layer: 64 neurons (narrowing down features)

constexpr int   OUTPUT_SIZE   = 10;     // One output neuron per digit class (0-9)
                                        // The neuron with the highest output = our prediction

constexpr int   EPOCHS        = 10;     // One "epoch" = one full pass through all 60,000 training images
                                        // More epochs = more learning, but risks "memorising" training data

constexpr int   BATCH_SIZE    = 64;     // Mini-batch size: instead of updating weights after EVERY image
                                        // (slow and noisy) or after ALL images (too slow per update),
                                        // we process 64 images and average their gradients.
                                        // [Rashid, p. 99] — Mini-batches and averaging errors

constexpr float LEARNING_RATE = 0.01f; // How big a step we take when adjusting weights.
                                        // Too large = overshoot the minimum, too small = learn too slowly.
                                        // [Rashid, pp. 33-35, 87-88] — The hill-descent analogy for LR
                                        // [Nield, p. 254-258]         — Gradient descent step size

constexpr float MOMENTUM      = 0.9f;  // Momentum makes the optimizer "remember" its previous direction.
                                        // Instead of making purely local decisions, momentum lets
                                        // the weight update carry some velocity from the last step.
                                        // Think of it as a rolling ball that keeps some of its speed.
                                        // Value of 0.9 means 90% of previous velocity is kept each step.

constexpr float L2_LAMBDA     = 1e-4f; // L2 Regularisation (weight decay): adds a small penalty for
                                        // large weight values. This prevents the network from fitting
                                        // the training data TOO perfectly (overfitting).
                                        // The penalty added to the loss is: λ × sum(w²)
                                        // [Nield, p. 251] — Regularisation to reduce overfitting


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 3 — BYTE REVERSAL UTILITY
// ═══════════════════════════════════════════════════════════════════════════════

// ─── reverseBytes() ───────────────────────────────────────────────────────────
//
// WHAT IT DOES:
//   Takes a 32-bit unsigned integer and reverses the order of its 4 bytes.
//
// WHY WE NEED IT:
//   The MNIST binary files were saved on a computer that stored multi-byte numbers
//   in "big-endian" format (most significant byte first). Modern Intel/AMD x86
//   processors use "little-endian" (least significant byte first). If we just read
//   the bytes directly without reversing, the number 60000 (training images count)
//   would be read as garbage.
//
//   Example: the number 60000 in hex is 0x0000EA60
//   Big-endian on disk:    00 00 EA 60
//   Little-endian in CPU:  60 EA 00 00  (would read as 1,625,686,016 — wrong!)
//   After reverseBytes():  00 00 EA 60  = 60000 — correct!
//
// HOW THE BIT MANIPULATION WORKS:
//   val & 0xFF000000 isolates the highest byte, then >> 24 moves it to the lowest position
//   val & 0x00FF0000 isolates the 2nd byte,   then >> 8  moves it one position down
//   val & 0x0000FF00 isolates the 3rd byte,   then << 8  moves it one position up
//   val & 0x000000FF isolates the lowest byte, then << 24 moves it to the highest position
//   The | (bitwise OR) stitches all four pieces back together.
//
// CALLED BY: loadImages() — used right after reading each 4-byte header field

uint32_t reverseBytes(uint32_t val)
{
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >>  8) |
           ((val & 0x0000FF00) <<  8) |
           ((val & 0x000000FF) << 24);
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 4 — DATA LOADING
// ═══════════════════════════════════════════════════════════════════════════════

// ─── loadImages() ─────────────────────────────────────────────────────────────
//
// WHAT IT DOES:
//   Opens two binary files (the image file and the label file from the MNIST
//   dataset), reads every pixel from every image, normalizes the values,
//   and returns them in an MNISTData struct.
//
// PARAMETERS:
//   imgPath — file path to the image file (e.g. "train-images.idx3-ubyte")
//   lblPath — file path to the label file (e.g. "train-labels.idx1-ubyte")
//
// RETURNS: An MNISTData struct with both images and labels populated.
//
// CALLED BY: main() — once for training data, once for test data
//
// MATH CONCEPT — NORMALIZATION:
//   Raw pixel values are integers from 0 (black) to 255 (white).
//   Neural networks perform much better when inputs are in the range [0.0, 1.0]
//   because large input values cause large activations which can destabilize
//   gradient descent. Dividing by 255.0 scales every pixel into [0.0, 1.0].
//   This is called min-max normalization.
//   [Rashid, pp. 102, 134] — "Preparing Data" chapter: why we normalise inputs
//   [Nield, p. 343]         — Input scaling before feeding a neural network
//
// MNIST FILE FORMAT:
//   Images file header (bytes 0-15):
//     [0-3]  Magic number: 2051  (identifies this as an MNIST image file)
//     [4-7]  Number of images:   60000 (training) or 10000 (test)
//     [8-11] Rows per image:     28
//     [12-15] Cols per image:    28
//   After the header: numImages × 784 bytes, one byte per pixel.
//
//   Labels file header (bytes 0-7):
//     [0-3]  Magic number: 2049  (identifies this as an MNIST label file)
//     [4-7]  Number of labels:   same as number of images
//   After the header: numLabels × 1 byte, value = 0 to 9

MNISTData loadImages(const string& imgPath, const string& lblPath)
{
    // ── Open the image file in binary mode ──
    // ios::binary tells C++ NOT to interpret any bytes specially
    // (on Windows, without this, certain byte values like 0x1A are treated as end-of-file)
    ifstream imgFile(imgPath, ios::binary);

    // If the file couldn't be opened (wrong path, file doesn't exist, no permission)
    // throw an exception that will be caught in main() and printed as an error message
    if (!imgFile)
    {
        throw runtime_error("Cannot open: " + imgPath);
    }

    // ── Read the 16-byte header ──
    uint32_t magic, numImages, rows, cols;

    // reinterpret_cast<char*>(&magic) tells the file reader to treat the
    // address of our integer variable as a sequence of raw bytes
    // The second argument (4) means "read exactly 4 bytes"
    imgFile.read(reinterpret_cast<char*>(&magic),     4);
    imgFile.read(reinterpret_cast<char*>(&numImages), 4);
    imgFile.read(reinterpret_cast<char*>(&rows),      4);
    imgFile.read(reinterpret_cast<char*>(&cols),      4);

    // Convert the header values from big-endian (disk format) to little-endian (CPU format)
    magic     = reverseBytes(magic);
    numImages = reverseBytes(numImages);
    rows      = reverseBytes(rows);
    cols      = reverseBytes(cols);

    // Verify we're reading a real MNIST image file (magic == 2051) and 28×28 images
    // If this assert fires, the file is corrupted or the wrong file was opened
    assert(magic == 2051 && rows == 28 && cols == 28);

    // ── Allocate storage for all images ──
    // We create a 2D vector: numImages rows, each row has INPUT_SIZE (784) float values.
    // All values are initialised to 0.0f by default.
    vector<vector<float>> images(numImages, vector<float>(INPUT_SIZE));

    // ── Read pixel data for every image ──
    for (uint32_t i = 0; i < numImages; ++i)           // Loop over each image
    {
        for (int j = 0; j < INPUT_SIZE; ++j)           // Loop over each pixel (0 to 783)
        {
            uint8_t pixel;                             // One raw byte: 0 (black) to 255 (white)
            imgFile.read(reinterpret_cast<char*>(&pixel), 1);

            // NORMALIZATION: divide by 255 to scale pixel into [0.0, 1.0]
            // This is critical for stable gradient descent.
            // [Rashid, p. 134] — MNIST input normalisation
            images[i][j] = pixel / 255.0f;
        }
    }

    // ── Open and read the label file ──
    ifstream lblFile(lblPath, ios::binary);
    if (!lblFile)
    {
        throw runtime_error("Cannot open: " + lblPath);
    }

    uint32_t lMagic, numLabels;
    lblFile.read(reinterpret_cast<char*>(&lMagic),    4);
    lblFile.read(reinterpret_cast<char*>(&numLabels), 4);

    lMagic    = reverseBytes(lMagic);
    numLabels = reverseBytes(numLabels);

    // Magic 2049 = MNIST label file; number of labels must equal number of images
    assert(lMagic == 2049 && numLabels == numImages);

    // ── Read all labels ──
    vector<int> labels(numLabels);
    for (uint32_t i = 0; i < numLabels; ++i)
    {
        uint8_t lbl;
        lblFile.read(reinterpret_cast<char*>(&lbl), 1);
        labels[i] = static_cast<int>(lbl);             // Cast the raw byte (0-9) to int
    }

    // Return both images and labels bundled together in one MNISTData struct
    return { images, labels };
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 5 — ACTIVATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════
//
// An activation function is applied to the output of every neuron after the
// weighted sum is computed. Without activation functions, a neural network with
// multiple layers would behave exactly the same as one layer, because stacking
// linear functions just gives you another linear function.
//
// Activation functions introduce NON-LINEARITY, which lets the network learn
// curved, complex decision boundaries.
//
// [Rashid, pp. 43-45]       — Why neurons need an activation threshold / function
// [Nield, pp. 348-354]       — ReLU, sigmoid, softmax activation functions
// [Taylor, Ch. 3]            — Visual walkthrough of activation functions


// ─── relu() ───────────────────────────────────────────────────────────────────
//
// WHAT IT DOES:
//   ReLU = Rectified Linear Unit. For positive inputs, it passes the value through
//   unchanged. For negative inputs, it returns 0.
//
// THE MATH:
//   relu(x) = max(0, x)
//
//   This is the most commonly used activation function for hidden layers in modern
//   deep learning. It is simple, computationally cheap, and avoids the "vanishing
//   gradient" problem that sigmoid activations suffer from with many layers.
//
// CALLED BY: Layer::forward() — applied to every hidden neuron's output
//
// [Nield, pp. 348-350] — ReLU definition, graph, and why it's preferred
// [Taylor, Ch. 3]       — Comparison of ReLU vs sigmoid

inline float relu(float x)
{
    return x > 0.0f ? x : 0.0f;   // Ternary: "if x > 0, return x, else return 0"
}


// ─── reluGrad() ───────────────────────────────────────────────────────────────
//
// WHAT IT DOES:
//   Computes the DERIVATIVE of the ReLU function with respect to its input.
//   This is needed during backpropagation to compute the gradient of the loss
//   with respect to the weights in each layer.
//
// THE MATH (DERIVATIVE OF RELU):
//   d/dx relu(x) = { 1 if x > 0
//                  { 0 if x ≤ 0
//
//   This is a STEP FUNCTION — it's either 0 or 1. It tells the gradient how much
//   the loss should "flow through" this neuron. If the neuron was "off" (x ≤ 0),
//   no gradient flows back through it. This is called the "dead neuron" problem
//   but in practice it rarely causes major issues.
//
// CALLED BY: Layer::backprop() — to compute delta at each hidden layer
//
// [Nield, p. 349-350]     — ReLU derivative
// [Stewart, Section 3.1]  — Derivative of a piecewise function

inline float reluGrad(float x)
{
    return x > 0.0f ? 1.0f : 0.0f; // Gradient is 1 for positive pre-activations, 0 otherwise
}


// ─── softmax() ────────────────────────────────────────────────────────────────
//
// WHAT IT DOES:
//   Takes a vector of raw output scores (called "logits") from the final layer
//   and converts them into a probability distribution: every value is in [0, 1]
//   and they all sum to exactly 1.0.
//
//   The result: each output neuron now represents the probability that the input
//   image belongs to that digit class (0-9).
//
// THE MATH:
//   For a vector z = [z₀, z₁, ..., z₉]:
//     softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
//
//   Where exp() is e^x (Euler's number ≈ 2.718 raised to the power x).
//   The exponential function makes large values even larger relative to small ones,
//   which sharpens the "confidence" of the prediction.
//
// WHY WE SUBTRACT THE MAX (the `maxVal` trick):
//   If the raw scores are large (like z = [300, 400, 200]), then exp(400) overflows
//   to infinity in floating point. Subtracting the maximum value from all scores
//   doesn't change the final probabilities (the numerator and denominator both get
//   multiplied by the same constant, which cancels out), but it ensures the largest
//   exp() we compute is exp(0) = 1, preventing overflow.
//
//   Proof that subtracting max doesn't change the result:
//     softmax(zᵢ - max(z)) = exp(zᵢ - M) / Σⱼ exp(zⱼ - M)
//                           = exp(zᵢ)·exp(-M) / [Σⱼ exp(zⱼ)·exp(-M)]
//                           = exp(zᵢ) / Σⱼ exp(zⱼ)   ← same as original!
//
// CALLED BY: Network::forward() — applied to the output layer's activations
//
// [Nield, pp. 354-356]     — Softmax definition and use with multi-class output
// [Rashid, p. 144]         — How output layer probabilities are interpreted
// [Taylor, Ch. 4]          — Softmax in practice for digit classification

void softmax(vector<float>& v)
{
    // Find the largest element to subtract (numerical stability trick)
    float maxVal = *max_element(v.begin(), v.end());

    float sum = 0.0f;

    // Two passes: first compute exp(x - max), then divide by the sum
    for (int i = 0; i < (int)v.size(); ++i)
    {
        v[i] = exp(v[i] - maxVal);   // Replace each element with e^(z - max)
        sum += v[i];                  // Accumulate the denominator
    }

    for (int i = 0; i < (int)v.size(); ++i)
    {
        v[i] /= sum;                  // Divide each by the total sum → gives probability
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 6 — THE LAYER STRUCT
// ═══════════════════════════════════════════════════════════════════════════════
//
// A "Layer" represents one fully-connected layer of neurons. It stores all the
// weights and biases between this layer and the previous layer, and implements
// the forward pass (making predictions) and backpropagation (computing gradients).
//
// We create three Layer objects in the Network struct:
//   l1: connects INPUT (784 neurons) → HIDDEN1 (128 neurons)
//   l2: connects HIDDEN1 (128)       → HIDDEN2 (64 neurons)
//   l3: connects HIDDEN2 (64)        → OUTPUT  (10 neurons)
//
// MATHEMATICAL MODEL OF ONE LAYER:
//   Given input vector x (size = inSize):
//     z[j] = b[j] + Σᵢ W[j,i] × x[i]     ← weighted sum + bias (dot product)
//     a[j] = activation(z[j])              ← apply activation function
//
//   z is called the "pre-activation" and a is the "post-activation" or "output".
//
// [Rashid, pp. 51-65]   — Signals flowing through layers, matrix multiplication as forward pass
// [Nield, pp. 356-358]  — Forward propagation through a layer
// [Taylor, Ch. 3]       — Weights, biases, and the linear combination

struct Layer
{
    int inSize, outSize;        // Dimensions: how many neurons feed in, how many this layer produces

    // ── Learnable Parameters ──────────────────────────────────────────────────
    // W (weights):  A matrix stored as a flat 1D vector.
    //               W[j * inSize + i] = the weight from input neuron i to output neuron j.
    //               Size = outSize × inSize
    //
    // b (biases):   One bias per output neuron. The bias shifts the activation function
    //               horizontally — it lets a neuron "fire" even when all inputs are zero.
    //               Initialised to 0.
    //               Size = outSize
    //
    // [Rashid, pp. 57-63]  — Why we need weights AND biases
    // [Nield, pp. 342-346] — Weights and biases as the parameters we are learning
    vector<float> W;   // Weight matrix (flat), size = outSize * inSize
    vector<float> b;   // Bias vector,          size = outSize

    // ── Gradient Accumulators ─────────────────────────────────────────────────
    // dW and db store the ACCUMULATED gradient of the loss with respect to each
    // weight/bias during a mini-batch. After the batch, we use them to update W and b,
    // then reset them to zero for the next batch.
    //
    // THE MATH:
    //   dW[j,i] = ∂Loss/∂W[j,i]    (how much the loss changes if we tweak weight j,i)
    //   db[j]   = ∂Loss/∂b[j]      (how much the loss changes if we tweak bias j)
    //
    // [Rashid, pp. 83-105]   — The weight update formula and why we need these gradients
    // [Nield, pp. 363-373]   — Backpropagation computing partial derivatives
    // [Stewart, Section 3.5] — Partial derivatives and the chain rule
    vector<float> dW;  // Accumulated weight gradients (same size as W)
    vector<float> db;  // Accumulated bias gradients   (same size as b)

    // ── Momentum Velocity Buffers ─────────────────────────────────────────────
    // Momentum is like giving the weight update a "running speed".
    // Instead of updating W by just -lr * gradient (vanilla SGD), we update a
    // velocity variable vW first, then apply it to W:
    //   vW = momentum * vW - lr * gradient
    //   W  = W + vW
    //
    // This smooths out noisy gradients and helps the optimizer escape shallow minima.
    //
    // [Nield, p. 254-258] — Momentum in gradient descent
    vector<float> vW;  // Momentum buffer for weights (same size as W)
    vector<float> vb;  // Momentum buffer for biases  (same size as b)

    // ── Cached Activations ───────────────────────────────────────────────────
    // We MUST save the values computed during the forward pass, because
    // backpropagation needs them later to compute gradients.
    //
    // z = pre-activation  (the raw weighted sum before applying ReLU)
    //     Needed in backprop to compute reluGrad(z)
    //
    // a = post-activation (the output after applying ReLU or linear)
    //     Needed in backprop to compute weight gradients: dW[j,i] += delta[j] * a[i]
    //
    // [Rashid, pp. 74-78] — Why we need to save activations for backpropagation
    vector<float> z;   // Pre-activation values  (size = outSize)
    vector<float> a;   // Post-activation values (size = outSize)


    // ─── Layer Constructor ────────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Called once when we create the Network. Allocates memory for all the
    //   vectors listed above, then initialises the weights using "He initialisation".
    //
    // PARAMETERS:
    //   in  — number of input neurons (previous layer's size)
    //   out — number of output neurons (this layer's size)
    //   rng — a Mersenne Twister random number generator (shared by all layers)
    //
    // HE INITIALISATION (why not just set all weights to zero?):
    //   If all weights start at zero, every neuron in a layer computes the same
    //   output during the forward pass, producing the same gradient in backprop.
    //   All neurons are "symmetric" and update identically — the network never learns
    //   different features. This is called the "symmetry problem."
    //
    //   Random initialisation breaks this symmetry. But HOW random matters:
    //   weights that are too large cause large activations that saturate the network;
    //   weights that are too small cause vanishing gradients.
    //
    //   He initialisation (Kaiming He, 2015) draws weights from a normal distribution
    //   with mean = 0 and standard deviation = sqrt(2 / fan_in), where fan_in is the
    //   number of input neurons. The factor of 2 is specifically tuned for ReLU:
    //   because ReLU zeros out roughly half its inputs, the factor of 2 compensates
    //   to keep the variance of activations stable across layers.
    //
    //   [Nield, p. 363] — Weight initialisation strategies
    //
    // CALLED BY: Network constructor (when creating l1, l2, l3)

    Layer(int in, int out, mt19937& rng)
        : inSize(in), outSize(out)     // Member initialiser list: set inSize and outSize first
    {
        // Resize all vectors to their correct sizes
        W.resize(out * in);            // Weight matrix: out×in entries
        b.resize(out, 0.0f);           // Bias: one per output neuron, initialised to 0

        dW.resize(out * in, 0.0f);     // Gradient accumulators start at 0
        db.resize(out, 0.0f);

        vW.resize(out * in, 0.0f);     // Momentum buffers start at 0 (no initial velocity)
        vb.resize(out, 0.0f);

        z.resize(out);                 // Pre-activation buffer (size = outputs)
        a.resize(out);                 // Post-activation buffer (size = outputs)

        // ── He Initialisation ──
        // Compute the standard deviation: sqrt(2.0 / number_of_inputs)
        // This formula comes from the 2015 paper "Delving Deep into Rectifiers" by He et al.
        float stddev = sqrt(2.0f / in);

        // Create a normal (Gaussian) distribution with mean=0, std=stddev
        // The random number generator rng is seeded before calling this,
        // so results are reproducible across runs.
        normal_distribution<float> dist(0.0f, stddev);

        // Fill every weight with a random sample from the distribution
        for (int i = 0; i < (int)W.size(); ++i)
        {
            W[i] = dist(rng);
        }
        // Biases are left at 0 — that's standard practice; the weights handle symmetry breaking
    }


    // ─── Layer::forward() ────────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Performs the forward pass through this layer. Given the output of the
    //   previous layer (called `input`), it computes the weighted sum + bias
    //   for each output neuron, then applies the activation function.
    //
    // THE MATH (for each output neuron j):
    //   z[j] = b[j] + Σᵢ (W[j,i] × input[i])     ← dot product of one row of W with input
    //   a[j] = relu(z[j])     if use_relu == true
    //   a[j] = z[j]           if use_relu == false  (output layer; softmax applied externally)
    //
    //   This is the core "matrix-vector multiply" operation. The weights W form a matrix,
    //   and input is a column vector. The dot product of each row of W with input gives
    //   one output value.
    //
    // [Rashid, pp. 57-63]   — "Matrix Multiplication is Useful" — this is exactly that!
    // [Nield, pp. 169-180]  — Chapter 4: dot products and matrix-vector multiplication
    // [Nield, pp. 356-358]  — Forward propagation implementation
    //
    // PARAMETERS:
    //   input     — post-activation output from the previous layer (or raw pixel data for l1)
    //   use_relu  — true for hidden layers (apply ReLU), false for output layer (linear, then softmax)
    //
    // CALLED BY: Network::forward()

    void forward(const vector<float>& input, bool use_relu)
    {
        for (int j = 0; j < outSize; ++j)              // Loop over each output neuron
        {
            float sum = b[j];                          // Start with the bias for neuron j

            // W is stored in row-major order: W[j * inSize + 0], W[j * inSize + 1], ...
            // `row` is a pointer to the start of row j in the weight matrix
            const float* row = &W[j * inSize];

            // Compute the dot product of row j of W with the input vector
            // This is: sum += W[j,0]*input[0] + W[j,1]*input[1] + ... + W[j,inSize-1]*input[inSize-1]
            // [Nield, p. 170] — Dot product definition and computation
            for (int i = 0; i < inSize; ++i)
            {
                sum += row[i] * input[i];
            }

            z[j] = sum;                                // Save pre-activation (needed for backprop)
            a[j] = use_relu ? relu(sum) : sum;         // Apply ReLU or leave linear for output layer
        }
    }


    // ─── Layer::accumulateGrads() ────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Adds the contribution of ONE sample in the mini-batch to the gradient
    //   accumulators dW and db. After all BATCH_SIZE samples have been processed,
    //   update() will average these and apply them to the weights.
    //
    // THE MATH (backpropagation gradient for weights and biases):
    //   For output neuron j and input neuron i:
    //     dW[j,i] += delta[j] × prevA[i]    ← "outer product" of delta and previous activation
    //     db[j]   += delta[j]                ← bias gradient is just delta directly
    //
    //   Here, `delta` is the "error signal" propagated backwards from the next layer.
    //   It tells us: if we increased z[j] by a tiny amount, how much would the loss change?
    //   Multiplying delta[j] by prevA[i] gives the gradient of the loss w.r.t. W[j,i].
    //
    //   INTUITION: The more strongly input i activated (large prevA[i]), and the more
    //   that neuron j was "responsible" for error (large delta[j]), the more we need to
    //   adjust the weight connecting i → j.
    //
    // [Rashid, pp. 78-82]   — Deriving weight update rules from the error signal
    // [Nield, pp. 363-373]  — Backpropagation and computing weight gradients
    // [Stewart, Section 3.5]— Chain rule applied to nested functions (as used here)
    //
    // PARAMETERS:
    //   delta — error signal for this layer's output neurons (size = outSize)
    //   prevA — post-activation values from the PREVIOUS layer (size = inSize)
    //
    // CALLED BY: Network::backward()

    void accumulateGrads(const vector<float>& delta, const vector<float>& prevA)
    {
        for (int j = 0; j < outSize; ++j)
        {
            db[j] += delta[j];                         // Bias gradient: just add delta directly

            float* row = &dW[j * inSize];              // Pointer to row j of the gradient matrix

            for (int i = 0; i < inSize; ++i)
            {
                row[i] += delta[j] * prevA[i];         // Weight gradient: delta × previous activation
            }
        }
    }


    // ─── Layer::backprop() ───────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Computes the "delta" (error signal) for the PREVIOUS layer, given the
    //   delta for the CURRENT layer. This is the core of backpropagation.
    //
    // THE MATH (CHAIN RULE):
    //   We want ∂Loss/∂z_prev[i] for the previous layer.
    //   By the chain rule:
    //     ∂Loss/∂z_prev[i] = (Σⱼ W[j,i] × delta[j]) × relu'(prevZ[i])
    //
    //   The first part (Σⱼ W[j,i] × delta[j]) is a "transpose matrix-vector multiply":
    //   instead of multiplying W by the forward input, we multiply Wᵀ by delta.
    //   This propagates the error BACKWARD through the weight matrix.
    //
    //   The second part (relu'(prevZ[i])) applies the derivative of the activation
    //   function. If prevZ[i] ≤ 0 (the neuron was "off"), no gradient flows through it.
    //
    //   This is sometimes called the "delta rule" or "backprop through a layer."
    //
    // [Rashid, pp. 74-82]   — "Backpropagating Errors" and the matrix form
    // [Nield, pp. 363-373]  — The chain rule in backpropagation
    // [Stewart, Section 3.5]— Chain rule for composite functions
    //
    // PARAMETERS:
    //   delta            — error signal from THIS layer (size = outSize)
    //   prevZ            — pre-activation values from the PREVIOUS layer (size = inSize)
    //   apply_relu_grad  — true for hidden layers (multiply by relu derivative), false for input layer
    //
    // RETURNS:
    //   prevDelta — the error signal to be passed back to the layer before this one
    //
    // CALLED BY: Network::backward()

    vector<float> backprop(const vector<float>& delta,
                           const vector<float>& prevZ,
                           bool apply_relu_grad) const
    {
        vector<float> prevDelta(inSize, 0.0f);   // Initialise error signal for previous layer

        for (int i = 0; i < inSize; ++i)         // Loop over neurons in the previous layer
        {
            float sum = 0.0f;
            for (int j = 0; j < outSize; ++j)
            {
                // Sum error contributions from all neurons in THIS layer that are
                // connected to neuron i in the previous layer.
                // W[j * inSize + i] = weight connecting prev neuron i → this neuron j
                sum += W[j * inSize + i] * delta[j];   // Transposed weight multiply
            }

            // Multiply by the ReLU derivative to "gate" the gradient:
            // If the previous neuron was inactive (prevZ[i] ≤ 0), no gradient flows
            prevDelta[i] = apply_relu_grad ? sum * reluGrad(prevZ[i]) : sum;
        }

        return prevDelta;
    }


    // ─── Layer::update() ─────────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   After a full mini-batch has been processed, this function uses the
    //   accumulated gradients (dW, db) to update the weights (W, b) using
    //   SGD with momentum and L2 regularisation. Then it resets dW and db to 0.
    //
    // THE FULL UPDATE MATH (SGD + Momentum + L2):
    //
    //   Step 1: Compute the effective gradient (averaged over batch + L2 penalty):
    //     grad_W[k] = (dW[k] / batch_size) + λ × W[k]
    //                  ↑ average gradient      ↑ L2 penalty pushes weight toward 0
    //
    //   Step 2: Update the velocity using momentum:
    //     vW[k] = momentum × vW[k]  -  lr × grad_W[k]
    //              ↑ keep old velocity    ↑ nudge in gradient direction
    //
    //   Step 3: Apply velocity to weights:
    //     W[k] = W[k] + vW[k]
    //
    //   The same three steps apply to biases (without L2 penalty — biases are rarely regularised).
    //
    // WHY DIVIDE BY BATCH_SIZE?
    //   We accumulated gradients from BATCH_SIZE samples. We divide to get the
    //   AVERAGE gradient, which is a more stable estimate of the true gradient.
    //
    // WHY ADD λ × W[k] TO THE GRADIENT?
    //   This is L2 regularisation (also called "weight decay"). It penalises large
    //   weights by always pulling them toward zero, discouraging the network from
    //   overfitting to the training data.
    //
    // [Rashid, pp. 87-105]  — Gradient descent weight update formula
    // [Nield, pp. 254-258]  — Gradient descent and learning rate
    // [Nield, p. 251]       — L2 regularisation
    //
    // PARAMETERS:
    //   lr         — current learning rate (may be decayed over epochs)
    //   momentum   — fraction of old velocity to keep (e.g. 0.9)
    //   l2_lambda  — regularisation strength (e.g. 0.0001)
    //   batchSz    — number of samples in the mini-batch (used to average gradients)
    //
    // CALLED BY: Network::update(), which is called at the end of every mini-batch

    void update(float lr, float momentum, float l2_lambda, int batchSz)
    {
        float scale = 1.0f / batchSz;                  // 1/64 for BATCH_SIZE=64

        // ── Update all weights ──
        for (int k = 0; k < outSize * inSize; ++k)
        {
            // Compute effective gradient: average batch gradient + L2 regularisation term
            float grad = dW[k] * scale + l2_lambda * W[k];

            // Momentum update: blend old velocity with new gradient direction
            vW[k] = momentum * vW[k] - lr * grad;

            W[k] += vW[k];         // Apply the velocity to the weight

            dW[k] = 0.0f;          // Reset accumulated gradient to zero for next batch
        }

        // ── Update all biases ──
        for (int j = 0; j < outSize; ++j)
        {
            float grad = db[j] * scale;               // Average gradient only — no L2 on biases

            vb[j] = momentum * vb[j] - lr * grad;

            b[j] += vb[j];

            db[j] = 0.0f;
        }
    }
};


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 7 — THE NETWORK STRUCT
// ═══════════════════════════════════════════════════════════════════════════════
//
// The Network struct ties together three Layer objects into a complete feedforward
// neural network. It provides three high-level operations:
//   - forward()  — compute a prediction for one image
//   - backward() — compute gradients using backpropagation
//   - update()   — apply gradients to all weights (calls each layer's update)
//   - predict()  — returns the winning digit class (0-9)
//
// DATA FLOW (FORWARD):
//   raw pixels (784) → l1 → relu → l2 → relu → l3 → softmax → probabilities (10)
//
// DATA FLOW (BACKWARD):
//   cross-entropy loss → delta3 → l3 → delta2 → l2 → delta1 → l1
//
// [Rashid, pp. 51-57, 74-82] — Full network signal flow, forward and backward
// [Nield, pp. 356-375]        — Network architecture and backpropagation
// [Taylor, Ch. 4-5]           — End-to-end network operations

struct Network
{
    Layer l1, l2, l3;   // l1 = 784→128, l2 = 128→64, l3 = 64→10

    // ─── Network Constructor ─────────────────────────────────────────────────
    //
    // Creates all three layers with their correct sizes and the shared RNG.
    // The initialiser list `: l1(...), l2(...), l3(...)` constructs each Layer
    // before the constructor body runs.

    Network(mt19937& rng)
        : l1(INPUT_SIZE,   HIDDEN1_SIZE, rng),
          l2(HIDDEN1_SIZE, HIDDEN2_SIZE, rng),
          l3(HIDDEN2_SIZE, OUTPUT_SIZE,  rng) {}


    // ─── Network::forward() ──────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Feeds one image (a vector of 784 pixel values) through all three layers
    //   and returns 10 probabilities — one for each digit class.
    //
    // THE MATH (complete forward pass):
    //   a1 = relu( W1 · x  + b1 )     ← layer 1: 784 → 128
    //   a2 = relu( W2 · a1 + b2 )     ← layer 2: 128 → 64
    //   a3 =       W3 · a2 + b3       ← layer 3: 64 → 10 (linear, no activation yet)
    //   p  = softmax(a3)               ← convert scores to probabilities
    //
    //   The final output p[k] = probability that the image is digit k.
    //
    // RETURN VALUE:
    //   A vector of 10 floats, each in [0,1], summing to 1.0.
    //   Example: [0.01, 0.02, 0.90, 0.01, ...] → network thinks it's a "2"
    //
    // [Rashid, pp. 52-57]  — "Following Signals Through a Neural Network"
    // [Nield, pp. 356-360] — Forward propagation step by step
    // CALLED BY: backward() (to get probabilities for loss computation) and predict()

    vector<float> forward(const vector<float>& x)
    {
        l1.forward(x,    true);    // Input pixels → hidden layer 1, apply ReLU
        l2.forward(l1.a, true);    // Hidden layer 1 → hidden layer 2, apply ReLU
        l3.forward(l2.a, false);   // Hidden layer 2 → output layer, NO activation yet

        // Copy the raw output scores (logits) from l3
        vector<float> probs = l3.a;

        // Apply softmax to turn raw scores into probabilities
        softmax(probs);

        return probs;              // 10 probabilities for digits 0-9
    }


    // ─── Network::backward() ─────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Given the forward pass output (probabilities) and the true label, computes
    //   the gradients of the loss with respect to every weight and bias in the network
    //   using backpropagation. Accumulates these into each layer's dW/db buffers.
    //   Also returns the scalar loss for this one sample.
    //
    // THE MATH — CROSS-ENTROPY LOSS:
    //   The loss for one sample is:
    //     L = -log(p[y])
    //   where p[y] is the predicted probability for the TRUE class y.
    //
    //   Example: if true label is 3, and the network says p[3] = 0.9, loss = -log(0.9) ≈ 0.105
    //            if p[3] = 0.1 (bad prediction), loss = -log(0.1) ≈ 2.303 (much higher)
    //
    //   This loss function heavily penalises confident wrong predictions (small p[y]).
    //   Cross-entropy is the standard loss for classification tasks.
    //   [Nield, p. 363-364] — Loss functions for classification
    //
    // THE MATH — OUTPUT LAYER GRADIENT (Softmax + Cross-Entropy combined):
    //   This is a beautiful result: when you combine softmax with cross-entropy loss,
    //   the gradient simplifies to:
    //     delta3[j] = p[j] - one_hot[j]
    //
    //   where one_hot[j] = 1 if j == label, else 0.
    //   This means: the gradient is just how wrong the probability is.
    //   If p[3] = 0.9 for the true class 3: delta3[3] = 0.9 - 1 = -0.1 (small, good prediction)
    //   If p[3] = 0.1:                       delta3[3] = 0.1 - 1 = -0.9 (large, bad prediction)
    //
    //   [Rashid, pp. 93-99]  — Deriving the gradient of the output layer
    //   [Nield, pp. 363-365] — Cross-entropy gradient derivation
    //   [Stewart, Section 3.5] — Chain rule (this derivation uses it twice!)
    //
    // BACKPROP CHAIN:
    //   delta3 → accumulateGrads(l3) → backprop() → delta2 → accumulateGrads(l2) → ... → l1
    //
    // PARAMETERS:
    //   x      — the input image pixels (784 floats)
    //   label  — the true class (0-9)
    //   probs  — the softmax output from forward() (10 floats)
    //
    // RETURNS: Cross-entropy loss for this sample (a single float)
    //
    // CALLED BY: main training loop

    float backward(const vector<float>& x, int label, const vector<float>& probs)
    {
        // ── Step 1: Compute output layer delta (softmax + cross-entropy gradient) ──
        vector<float> delta3(OUTPUT_SIZE);

        for (int j = 0; j < OUTPUT_SIZE; ++j)
        {
            // delta = predicted probability - true probability (1 if correct class, else 0)
            // This is the gradient of cross-entropy loss w.r.t. the pre-softmax scores
            delta3[j] = probs[j] - (j == label ? 1.0f : 0.0f);
        }

        // ── Step 2: Accumulate gradients for layer 3 ──
        // l2.a is the post-activation output of hidden layer 2 (the "input" to l3)
        l3.accumulateGrads(delta3, l2.a);

        // ── Step 3: Backpropagate delta through layer 3 to get layer 2's delta ──
        // l2.z is the pre-activation of hidden layer 2 (needed for reluGrad)
        vector<float> delta2 = l3.backprop(delta3, l2.z, true);

        // ── Step 4: Accumulate gradients for layer 2 ──
        l2.accumulateGrads(delta2, l1.a);

        // ── Step 5: Backpropagate delta through layer 2 to get layer 1's delta ──
        vector<float> delta1 = l2.backprop(delta2, l1.z, true);

        // ── Step 6: Accumulate gradients for layer 1 ──
        // The input to l1 is the raw pixel vector x (no previous layer activation)
        l1.accumulateGrads(delta1, x);

        // ── Return the cross-entropy loss for this one sample ──
        // max(..., 1e-9f) prevents log(0) = -infinity if the network is totally wrong
        return -log(max(probs[label], 1e-9f));
    }


    // ─── Network::update() ───────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Calls update() on all three layers to apply the accumulated gradients
    //   and reset them for the next mini-batch.
    //
    // CALLED BY: main training loop — once per mini-batch, after backward() has been
    //            called BATCH_SIZE times.

    void update(float lr, float momentum, float l2_lambda, int batchSz)
    {
        l1.update(lr, momentum, l2_lambda, batchSz);
        l2.update(lr, momentum, l2_lambda, batchSz);
        l3.update(lr, momentum, l2_lambda, batchSz);
    }


    // ─── Network::predict() ──────────────────────────────────────────────────
    //
    // WHAT IT DOES:
    //   Runs a single forward pass and returns the digit class with the highest
    //   predicted probability (the argmax of the softmax output).
    //
    // THE MATH:
    //   prediction = argmax(softmax(W3 · relu(W2 · relu(W1 · x + b1) + b2) + b3))
    //
    //   argmax = "the index of the largest element"
    //   For a handwritten "3", we'd expect p[3] to be the highest.
    //
    // [Rashid, p. 56-57] — Output layer and selecting the highest-confidence answer
    // [Nield, p. 360]     — Using argmax for multi-class prediction
    //
    // CALLED BY: main() — to compute accuracy on training and test sets

    int predict(const vector<float>& x)
    {
        vector<float> probs = forward(x);

        // max_element() returns an iterator to the largest element
        // Subtracting probs.begin() gives the INDEX of that element (0-9)
        return static_cast<int>(max_element(probs.begin(), probs.end()) - probs.begin());
    }
};


// ═══════════════════════════════════════════════════════════════════════════════
//  SECTION 8 — MAIN TRAINING LOOP
// ═══════════════════════════════════════════════════════════════════════════════

int main()
{
    cout << "=== MNIST Trainer (Allman Style + Full Braces + No auto + No range-based for + No size_t) ===\n\n";

    // ── 1. Load the MNIST dataset ────────────────────────────────────────────
    // loadImages() opens the binary files and returns an MNISTData struct.
    // The try/catch block handles the case where files aren't found.
    // [Rashid, pp. 144-152] — The MNIST dataset: what it is and how it's structured

    MNISTData train, test;

    try
    {
        // Training set: 60,000 images
        train = loadImages("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        // Test set: 10,000 images (NEVER used during training — only for evaluation)
        test  = loadImages("t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte");
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\nMake sure the 4 .ubyte files are in this folder.\n";
        return 1;
    }

    cout << "Training samples: " << train.images.size() << "\n";
    cout << "Test samples:     " << test.images.size()  << "\n\n";

    // ── 2. Set up the random number generator ───────────────────────────────
    // mt19937 is a Mersenne Twister, a high-quality pseudo-random number generator.
    // Seeding with 42 means every run will produce the same random weights,
    // making experiments reproducible. Change the seed to get different starting points.
    // [Rashid, pp. 88-89] — Why different starting weights matter (avoiding local minima)

    mt19937 rng(42);

    // ── 3. Build the network ─────────────────────────────────────────────────
    // This calls the Network constructor, which creates l1, l2, l3.
    // Each Layer's constructor runs He initialisation on its weights.

    Network net(rng);

    // ── 4. Create a list of shuffled indices ─────────────────────────────────
    // Instead of shuffling the actual images (expensive), we shuffle a list of
    // index numbers (0, 1, 2, ..., 59999) and use that to index into the data.
    // iota() fills the vector with sequential values starting from 0.

    vector<int> indices(train.images.size());
    iota(indices.begin(), indices.end(), 0);    // indices = [0, 1, 2, ..., 59999]

    cout << "Starting training...\n\n";

    // ── 5. EPOCH LOOP ────────────────────────────────────────────────────────
    //
    // One epoch = one complete pass through the entire training dataset.
    // Multiple epochs allow the network to refine its weights repeatedly.
    // Shuffling at the start of each epoch prevents the network from "memorising"
    // the order of examples, which would hurt generalisation.
    //
    // [Rashid, pp. 99-100] — The purpose of multiple training epochs

    for (int epoch = 1; epoch <= EPOCHS; ++epoch)
    {
        // Time the epoch so we can show how fast training is progressing
        chrono::steady_clock::time_point t0 = chrono::steady_clock::now();

        // Shuffle the index order so images appear in a different sequence each epoch
        // [Rashid, p. 102] — Shuffling training data each epoch
        shuffle(indices.begin(), indices.end(), rng);

        float epochLoss = 0.0f;                              // Running total loss for this epoch
        int numBatches = train.images.size() / BATCH_SIZE;   // How many full mini-batches fit

        // ── LEARNING RATE DECAY ───────────────────────────────────────────────
        // We reduce the learning rate as training progresses ("step decay"):
        //   epoch 1-5:  lr = 0.01  × (0.5)^0 = 0.01
        //   epoch 6-10: lr = 0.01  × (0.5)^1 = 0.005
        //   epoch 11-15: lr = 0.01 × (0.5)^2 = 0.0025
        //
        // WHY? Early in training, big steps quickly improve accuracy.
        // Later, smaller steps allow the optimizer to fine-tune without overshooting.
        // [Rashid, pp. 34-35]  — Moderating step size (learning rate)
        // [Nield, pp. 255-257] — Learning rate decay schedules

        float lr = LEARNING_RATE * pow(0.5f, (epoch - 1) / 5);

        // ── MINI-BATCH LOOP ───────────────────────────────────────────────────
        //
        // Mini-batch SGD: instead of one weight update per sample (noisy)
        // or one per full epoch (smooth but slow), we update every BATCH_SIZE samples.
        //
        // [Rashid, p. 99]      — Mini-batches as a compromise
        // [Nield, pp. 373-374] — Stochastic/mini-batch gradient descent

        for (int b = 0; b < numBatches; ++b)
        {
            float batchLoss = 0.0f;

            // ── Process one mini-batch ─────────────────────────────────────
            for (int k = 0; k < BATCH_SIZE; ++k)
            {
                // Look up which training sample to use (shuffled order)
                int idx = indices[b * BATCH_SIZE + k];

                // ── FORWARD PASS ─────────────────────────────────────────
                // Feed the image through the network → get 10 probabilities
                // [Rashid, pp. 52-57] — Forward signal propagation
                // [Nield, pp. 356-360] — Forward propagation
                vector<float> probs = net.forward(train.images[idx]);

                // ── BACKWARD PASS (Backpropagation) ──────────────────────
                // Compute how wrong we were, and accumulate gradients
                // Returns the cross-entropy loss for this one sample
                // [Rashid, pp. 74-82, 93-99] — Backpropagation theory and derivation
                // [Nield, pp. 363-373]        — Backpropagation implementation
                batchLoss += net.backward(train.images[idx], train.labels[idx], probs);
            }

            // ── WEIGHT UPDATE ─────────────────────────────────────────────
            // After processing all BATCH_SIZE samples, apply the averaged gradients
            // to all weights and biases using SGD + momentum + L2.
            // [Rashid, pp. 99-105] — Weight update worked example
            // [Nield, pp. 254-258] — Gradient descent optimisation
            net.update(lr, MOMENTUM, L2_LAMBDA, BATCH_SIZE);

            epochLoss += batchLoss / BATCH_SIZE;   // Add average batch loss to epoch total
        }

        // Stop timing the epoch
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();

        // ── COMPUTE TEST ACCURACY ─────────────────────────────────────────
        // Run every test image through the network and count how many are correct.
        // This is done AFTER weight updates (not used for training).
        // Test accuracy tells us how well the network GENERALISES to unseen data.
        // [Rashid, pp. 152-155] — Testing the trained network on MNIST

        int correct = 0;
        for (int i = 0; i < (int)test.images.size(); ++i)
        {
            if (net.predict(test.images[i]) == test.labels[i])
            {
                ++correct;
            }
        }

        // ── PRINT EPOCH SUMMARY ───────────────────────────────────────────
        cout << "Epoch " << setw(2) << epoch << "/" << EPOCHS
             << "  loss: " << fixed << setprecision(4) << epochLoss / numBatches
             << "  test acc: " << setprecision(2) << (100.0f * correct / test.images.size()) << "%"
             << "  lr: " << setprecision(5) << lr
             << "  (" << setprecision(1) << elapsed << "s)\n";
    }

    cout << "\nTraining finished!\n";
    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 *  FURTHER READING — WHERE EACH CONCEPT IS EXPLAINED IN DEPTH
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  CONCEPT                    RASHID               NIELD              STEWART
 *  ─────────────────────────────────────────────────────────────────────────
 *  What is a neural network?  pp. 3-12             pp. 342-345        —
 *  Activation functions       pp. 43-45            pp. 348-354        —
 *  ReLU specifically          pp. 43-45            pp. 348-350        —
 *  Softmax output             p. 144               pp. 354-356        —
 *  Forward pass (matrix)      pp. 57-65            pp. 356-360        —
 *  Dot product / matrix mult  pp. 57-63            pp. 169-180 (Ch4)  —
 *  Gradient descent           pp. 84-92            pp. 254-258 (Ch5)  —
 *  Learning rate              pp. 33-35, 99        pp. 255-257        —
 *  Backpropagation            pp. 74-82            pp. 363-373 (Ch7)  —
 *  Cross-entropy loss         pp. 93-99            pp. 363-364        —
 *  Weight update formula      pp. 99-105           pp. 363-373        —
 *  Chain rule (math)          pp. 93-99            pp. 59-63 (Ch1)    Section 3.5
 *  Derivatives (math)         Appendix A           pp. 49-53 (Ch1)    Chapter 3
 *  Weight initialisation      p. 28 (random init)  p. 363             —
 *  Normalising inputs         pp. 102, 134         p. 343             —
 *  The MNIST dataset          pp. 144-155          p. 278 (Ch7 App)   —
 *  Mini-batch training        p. 99                pp. 373-374        —
 *  Overfitting / L2 reg.      —                    p. 251             —
 *  Momentum                   —                    pp. 254-258        —
 *
 *  ALSO RECOMMENDED:
 *  - 3Blue1Brown "Neural Networks" YouTube series (visual, excellent)
 *    mentioned in [Nield, p. 386] and [Rashid, p. 7]
 * ════════════════════════════════════════════════════════════════════════════
 */