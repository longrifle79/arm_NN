/*
 * MNIST Neural Network Trainer in C++
 *
 * Architecture: 784 -> 128 -> 64 -> 10
 * Activation:   ReLU (hidden), Softmax (output)
 * Loss:         Cross-entropy
 * Optimizer:    Mini-batch SGD with momentum
 *
 * Build:
 *   g++ -O2 -std=c++17 -o mnist_train mnist_train.cpp
 *
 * Download MNIST binary files from: http://yann.lecun.com/exdb/mnist/
 * Expected files in ./data/:
 *   train-images-idx3-ubyte
 *   train-labels-idx1-ubyte
 *   t10k-images-idx3-ubyte
 *   t10k-labels-idx1-ubyte
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

// ─── Hyperparameters ─────────────────────────────────────────────────────────
constexpr int    INPUT_SIZE    = 784;
constexpr int    HIDDEN1_SIZE  = 128;
constexpr int    HIDDEN2_SIZE  = 64;
constexpr int    OUTPUT_SIZE   = 10;
constexpr int    EPOCHS        = 20;
constexpr int    BATCH_SIZE    = 64;
constexpr float  LEARNING_RATE = 0.01f;
constexpr float  MOMENTUM      = 0.9f;
constexpr float  L2_LAMBDA     = 1e-4f;   // weight decay

// ─── Utility ──────────────────────────────────────────────────────────────────

// Reverse bytes for big-endian MNIST files
uint32_t reverseBytes(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >>  8) |
           ((val & 0x0000FF00) <<  8) |
           ((val & 0x000000FF) << 24);
}

// ─── MNIST Loader ─────────────────────────────────────────────────────────────

struct MNISTData {
    std::vector<std::vector<float>> images;  // [n_samples][784], values in [0,1]
    std::vector<int>                labels;  // [n_samples], class 0-9
};

MNISTData loadImages(const std::string& imgPath, const std::string& lblPath) {
    // --- images ---
    std::ifstream imgFile(imgPath, std::ios::binary);
    if (!imgFile) throw std::runtime_error("Cannot open: " + imgPath);

    uint32_t magic, numImages, rows, cols;
    imgFile.read(reinterpret_cast<char*>(&magic),     4);
    imgFile.read(reinterpret_cast<char*>(&numImages), 4);
    imgFile.read(reinterpret_cast<char*>(&rows),      4);
    imgFile.read(reinterpret_cast<char*>(&cols),      4);

    magic     = reverseBytes(magic);
    numImages = reverseBytes(numImages);
    rows      = reverseBytes(rows);
    cols      = reverseBytes(cols);

    assert(magic == 2051 && "Invalid image file magic number");
    assert(rows == 28 && cols == 28);

    std::vector<std::vector<float>> images(numImages, std::vector<float>(INPUT_SIZE));
    for (uint32_t i = 0; i < numImages; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            uint8_t pixel;
            imgFile.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f;
        }
    }

    // --- labels ---
    std::ifstream lblFile(lblPath, std::ios::binary);
    if (!lblFile) throw std::runtime_error("Cannot open: " + lblPath);

    uint32_t lMagic, numLabels;
    lblFile.read(reinterpret_cast<char*>(&lMagic),    4);
    lblFile.read(reinterpret_cast<char*>(&numLabels), 4);
    lMagic    = reverseBytes(lMagic);
    numLabels = reverseBytes(numLabels);

    assert(lMagic == 2049 && "Invalid label file magic number");
    assert(numLabels == numImages);

    std::vector<int> labels(numLabels);
    for (uint32_t i = 0; i < numLabels; ++i) {
        uint8_t lbl;
        lblFile.read(reinterpret_cast<char*>(&lbl), 1);
        labels[i] = static_cast<int>(lbl);
    }

    return { images, labels };
}

// ─── Activation Functions ─────────────────────────────────────────────────────

inline float relu(float x)      { return x > 0.0f ? x : 0.0f; }
inline float reluGrad(float x)  { return x > 0.0f ? 1.0f : 0.0f; }

void softmax(std::vector<float>& v) {
    float maxVal = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (auto& x : v) { x = std::exp(x - maxVal); sum += x; }
    for (auto& x : v) x /= sum;
}

// ─── Layer ────────────────────────────────────────────────────────────────────

struct Layer {
    int    inSize, outSize;
    // Parameters
    std::vector<float> W;   // [outSize * inSize]
    std::vector<float> b;   // [outSize]
    // Gradients
    std::vector<float> dW;
    std::vector<float> db;
    // Momentum buffers
    std::vector<float> vW;
    std::vector<float> vb;
    // Cached activations (for backprop)
    std::vector<float> z;   // pre-activation
    std::vector<float> a;   // post-activation

    Layer(int in, int out, std::mt19937& rng) : inSize(in), outSize(out) {
        W.resize(out * in);
        b.resize(out, 0.0f);
        dW.resize(out * in, 0.0f);
        db.resize(out, 0.0f);
        vW.resize(out * in, 0.0f);
        vb.resize(out, 0.0f);
        z.resize(out);
        a.resize(out);

        // He initialisation (good for ReLU)
        float stddev = std::sqrt(2.0f / in);
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& w : W) w = dist(rng);
    }

    // Forward: output written to `a`; use_relu=false for final layer
    void forward(const std::vector<float>& input, bool use_relu) {
        for (int j = 0; j < outSize; ++j) {
            float sum = b[j];
            const float* row = &W[j * inSize];
            for (int i = 0; i < inSize; ++i) sum += row[i] * input[i];
            z[j] = sum;
            a[j] = use_relu ? relu(sum) : sum;  // softmax applied outside
        }
    }

    // Accumulate weight/bias gradients from delta and previous activation
    void accumulateGrads(const std::vector<float>& delta,
                         const std::vector<float>& prevA) {
        for (int j = 0; j < outSize; ++j) {
            db[j] += delta[j];
            float* row = &dW[j * inSize];
            for (int i = 0; i < inSize; ++i) row[i] += delta[j] * prevA[i];
        }
    }

    // Propagate delta to previous layer
    std::vector<float> backprop(const std::vector<float>& delta,
                                const std::vector<float>& prevZ,
                                bool apply_relu_grad) const {
        std::vector<float> prevDelta(inSize, 0.0f);
        for (int i = 0; i < inSize; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < outSize; ++j) sum += W[j * inSize + i] * delta[j];
            prevDelta[i] = apply_relu_grad ? sum * reluGrad(prevZ[i]) : sum;
        }
        return prevDelta;
    }

    // SGD + momentum update, then zero gradients
    void update(float lr, float momentum, float l2, int batchSz) {
        float scale = 1.0f / batchSz;
        for (int k = 0; k < outSize * inSize; ++k) {
            float grad = dW[k] * scale + l2 * W[k];
            vW[k] = momentum * vW[k] - lr * grad;
            W[k] += vW[k];
            dW[k]  = 0.0f;
        }
        for (int j = 0; j < outSize; ++j) {
            float grad = db[j] * scale;
            vb[j] = momentum * vb[j] - lr * grad;
            b[j] += vb[j];
            db[j]  = 0.0f;
        }
    }
};

// ─── Network ──────────────────────────────────────────────────────────────────

struct Network {
    Layer l1, l2, l3;   // 784->128, 128->64, 64->10

    Network(std::mt19937& rng)
        : l1(INPUT_SIZE,   HIDDEN1_SIZE, rng),
          l2(HIDDEN1_SIZE, HIDDEN2_SIZE, rng),
          l3(HIDDEN2_SIZE, OUTPUT_SIZE,  rng) {}

    // Returns softmax probabilities
    std::vector<float> forward(const std::vector<float>& x) {
        l1.forward(x,    true);   // ReLU
        l2.forward(l1.a, true);   // ReLU
        l3.forward(l2.a, false);  // linear (softmax below)
        std::vector<float> probs = l3.a;
        softmax(probs);
        return probs;
    }

    // Returns cross-entropy loss for one sample, accumulates gradients
    float backward(const std::vector<float>& x,
                   int label,
                   const std::vector<float>& probs) {
        // Output delta: dL/dz = p - one_hot(y)
        std::vector<float> delta3(OUTPUT_SIZE);
        for (int j = 0; j < OUTPUT_SIZE; ++j)
            delta3[j] = probs[j] - (j == label ? 1.0f : 0.0f);

        l3.accumulateGrads(delta3, l2.a);
        auto delta2 = l3.backprop(delta3, l2.z, true);

        l2.accumulateGrads(delta2, l1.a);
        auto delta1 = l2.backprop(delta2, l1.z, true);

        l1.accumulateGrads(delta1, x);

        return -std::log(std::max(probs[label], 1e-9f));
    }

    void update(float lr, float momentum, float l2, int batchSz) {
        l1.update(lr, momentum, l2, batchSz);
        l2.update(lr, momentum, l2, batchSz);
        l3.update(lr, momentum, l2, batchSz);
    }

    int predict(const std::vector<float>& x) {
        auto probs = forward(x);
        return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
    }
};

// ─── Training ─────────────────────────────────────────────────────────────────

float evaluate(Network& net, const MNISTData& data) {
    int correct = 0;
    for (size_t i = 0; i < data.images.size(); ++i)
        if (net.predict(data.images[i]) == data.labels[i]) ++correct;
    return 100.0f * correct / data.images.size();
}

int main() {
    const std::string dataDir = "./data/";

    // ── Load data ──
    std::cout << "Loading MNIST data...\n";
    MNISTData train, test;
    try {
        train = loadImages(dataDir + "train-images-idx3-ubyte",
                           dataDir + "train-labels-idx1-ubyte");
        test  = loadImages(dataDir + "t10k-images-idx3-ubyte",
                           dataDir + "t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"
                  << "Make sure MNIST binary files are in: " << dataDir << "\n";
        return 1;
    }
    std::cout << "Training samples: " << train.images.size() << "\n"
              << "Test samples:     " << test.images.size()  << "\n\n";

    // ── Build network ──
    std::mt19937 rng(42);
    Network net(rng);

    // ── Index shuffle ──
    std::vector<int> indices(train.images.size());
    std::iota(indices.begin(), indices.end(), 0);

    int numBatches = static_cast<int>(train.images.size()) / BATCH_SIZE;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Arch: " << INPUT_SIZE << " → " << HIDDEN1_SIZE
              << " → " << HIDDEN2_SIZE << " → " << OUTPUT_SIZE << "\n";
    std::cout << "LR: " << LEARNING_RATE << "  Momentum: " << MOMENTUM
              << "  L2: " << L2_LAMBDA << "  BatchSize: " << BATCH_SIZE << "\n\n";

    // ── Training loop ──
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();

        std::shuffle(indices.begin(), indices.end(), rng);

        // Learning rate decay every 5 epochs
        float lr = LEARNING_RATE * std::pow(0.5f, (epoch - 1) / 5);

        float epochLoss = 0.0f;

        for (int b = 0; b < numBatches; ++b) {
            float batchLoss = 0.0f;
            for (int k = 0; k < BATCH_SIZE; ++k) {
                int idx = indices[b * BATCH_SIZE + k];
                auto probs = net.forward(train.images[idx]);
                batchLoss += net.backward(train.images[idx], train.labels[idx], probs);
            }
            net.update(lr, MOMENTUM, L2_LAMBDA, BATCH_SIZE);
            epochLoss += batchLoss / BATCH_SIZE;
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        float trainAcc = evaluate(net, train);
        float testAcc  = evaluate(net, test);

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << EPOCHS
                  << "  loss: " << std::setw(7) << epochLoss / numBatches
                  << "  train: " << std::setw(6) << trainAcc << "%"
                  << "  test: "  << std::setw(6) << testAcc  << "%"
                  << "  lr: "    << std::setw(8) << lr
                  << "  (" << std::setprecision(1) << elapsed << "s)\n"
                  << std::setprecision(2);
    }

    std::cout << "\nDone.\n";
    return 0;
}
