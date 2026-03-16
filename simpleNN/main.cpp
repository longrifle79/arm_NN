/*
 * MNIST Neural Network Trainer in C++
 * Architecture: 784 -> 128 -> 64 -> 10
 * Allman brace style as requested
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

using namespace std;

// ─── Hyperparameters ─────────────────────────────────────────────────────────
constexpr int INPUT_SIZE   = 784;
constexpr int HIDDEN1_SIZE = 128;
constexpr int HIDDEN2_SIZE = 64;
constexpr int OUTPUT_SIZE  = 10;
constexpr int EPOCHS       = 10;
constexpr int BATCH_SIZE   = 64;
constexpr float LEARNING_RATE = 0.01f;
constexpr float MOMENTUM   = 0.9f;
constexpr float L2_LAMBDA  = 1e-4f;

// ─── Byte reversal for MNIST ─────────────────────────────────────────────────
uint32_t reverseBytes(uint32_t val)
{
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >>  8) |
           ((val & 0x0000FF00) <<  8) |
           ((val & 0x000000FF) << 24);
}

// ─── MNIST Loader ────────────────────────────────────────────────────────────
struct MNISTData
{
    vector<vector<float>> images;
    vector<int> labels;
};

MNISTData loadImages(const string& imgPath, const string& lblPath)
{
    // Images
    ifstream imgFile(imgPath, ios::binary);
    if (!imgFile) throw runtime_error("Cannot open " + imgPath);

    uint32_t magic, numImages, rows, cols;
    imgFile.read(reinterpret_cast<char*>(&magic), 4);
    imgFile.read(reinterpret_cast<char*>(&numImages), 4);
    imgFile.read(reinterpret_cast<char*>(&rows), 4);
    imgFile.read(reinterpret_cast<char*>(&cols), 4);

    magic = reverseBytes(magic);
    numImages = reverseBytes(numImages);
    rows = reverseBytes(rows);
    cols = reverseBytes(cols);

    assert(magic == 2051 && rows == 28 && cols == 28);

    vector<vector<float>> images(numImages, vector<float>(INPUT_SIZE));
    for (uint32_t i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            uint8_t pixel;
            imgFile.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f;
        }
    }

    // Labels
    ifstream lblFile(lblPath, ios::binary);
    if (!lblFile) throw runtime_error("Cannot open " + lblPath);

    uint32_t lMagic, numLabels;
    lblFile.read(reinterpret_cast<char*>(&lMagic), 4);
    lblFile.read(reinterpret_cast<char*>(&numLabels), 4);

    lMagic = reverseBytes(lMagic);
    numLabels = reverseBytes(numLabels);

    assert(lMagic == 2049 && numLabels == numImages);

    vector<int> labels(numLabels);
    for (uint32_t i = 0; i < numLabels; ++i)
    {
        uint8_t lbl;
        lblFile.read(reinterpret_cast<char*>(&lbl), 1);
        labels[i] = static_cast<int>(lbl);
    }

    return {images, labels};
}

// ─── Activation Functions ────────────────────────────────────────────────────
inline float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

inline float reluGrad(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

void softmax(vector<float>& v)
{
    float maxVal = *max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (auto& x : v)
    {
        x = exp(x - maxVal);
        sum += x;
    }
    for (auto& x : v) x /= sum;
}

// ─── Layer ───────────────────────────────────────────────────────────────────
struct Layer
{
    int inSize, outSize;
    vector<float> W, b, dW, db, vW, vb, z, a;

    Layer(int in, int out, mt19937& rng)
        : inSize(in), outSize(out)
    {
        W.resize(out*in); b.resize(out, 0.0f);
        dW.resize(out*in, 0.0f); db.resize(out, 0.0f);
        vW.resize(out*in, 0.0f); vb.resize(out, 0.0f);
        z.resize(out); a.resize(out);

        float stddev = sqrt(2.0f / in);
        normal_distribution<float> dist(0.0f, stddev);
        for (auto& w : W) w = dist(rng);
    }

    void forward(const vector<float>& input, bool use_relu)
    {
        for (int j = 0; j < outSize; ++j)
        {
            float sum = b[j];
            const float* row = &W[j * inSize];
            for (int i = 0; i < inSize; ++i) sum += row[i] * input[i];
            z[j] = sum;
            a[j] = use_relu ? relu(sum) : sum;
        }
    }

    void accumulateGrads(const vector<float>& delta, const vector<float>& prevA)
    {
        for (int j = 0; j < outSize; ++j)
        {
            db[j] += delta[j];
            float* row = &dW[j * inSize];
            for (int i = 0; i < inSize; ++i) row[i] += delta[j] * prevA[i];
        }
    }

    vector<float> backprop(const vector<float>& delta, const vector<float>& prevZ, bool apply_relu_grad) const
    {
        vector<float> prevDelta(inSize, 0.0f);
        for (int i = 0; i < inSize; ++i)
        {
            float sum = 0.0f;
            for (int j = 0; j < outSize; ++j) sum += W[j * inSize + i] * delta[j];
            prevDelta[i] = apply_relu_grad ? sum * reluGrad(prevZ[i]) : sum;
        }
        return prevDelta;
    }

    void update(float lr, float momentum, float l2_lambda, int batchSz)
    {
        float scale = 1.0f / batchSz;
        for (int k = 0; k < outSize * inSize; ++k)
        {
            float grad = dW[k] * scale + l2_lambda * W[k];
            vW[k] = momentum * vW[k] - lr * grad;
            W[k] += vW[k];
            dW[k] = 0.0f;
        }
        for (int j = 0; j < outSize; ++j)
        {
            float grad = db[j] * scale;
            vb[j] = momentum * vb[j] - lr * grad;
            b[j] += vb[j];
            db[j] = 0.0f;
        }
    }
};

// ─── Network ─────────────────────────────────────────────────────────────────
struct Network
{
    Layer l1, l2, l3;

    Network(mt19937& rng)
        : l1(INPUT_SIZE, HIDDEN1_SIZE, rng),
          l2(HIDDEN1_SIZE, HIDDEN2_SIZE, rng),
          l3(HIDDEN2_SIZE, OUTPUT_SIZE, rng) {}

    vector<float> forward(const vector<float>& x)
    {
        l1.forward(x, true);
        l2.forward(l1.a, true);
        l3.forward(l2.a, false);
        vector<float> probs = l3.a;
        softmax(probs);
        return probs;
    }

    float backward(const vector<float>& x, int label, const vector<float>& probs)
    {
        vector<float> delta3(OUTPUT_SIZE);
        for (int j = 0; j < OUTPUT_SIZE; ++j)
            delta3[j] = probs[j] - (j == label ? 1.0f : 0.0f);

        l3.accumulateGrads(delta3, l2.a);
        auto delta2 = l3.backprop(delta3, l2.z, true);
        l2.accumulateGrads(delta2, l1.a);
        auto delta1 = l2.backprop(delta2, l1.z, true);
        l1.accumulateGrads(delta1, x);

        return -log(max(probs[label], 1e-9f));
    }

    void update(float lr, float momentum, float l2_lambda, int batchSz)
    {
        l1.update(lr, momentum, l2_lambda, batchSz);
        l2.update(lr, momentum, l2_lambda, batchSz);
        l3.update(lr, momentum, l2_lambda, batchSz);
    }

    int predict(const vector<float>& x)
    {
        auto probs = forward(x);
        return static_cast<int>(max_element(probs.begin(), probs.end()) - probs.begin());
    }
};

// ─── Main ────────────────────────────────────────────────────────────────────
int main()
{
    cout << "=== MNIST Trainer (Allman Style) ===\n\n";

    MNISTData train, test;
    try
    {
        train = loadImages("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        test  = loadImages("t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte");
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\nMake sure the 4 .ubyte files are in this folder.\n";
        return 1;
    }

    cout << "Training samples: " << train.images.size() << "\n";
    cout << "Test samples: " << test.images.size() << "\n\n";

    mt19937 rng(42);
    Network net(rng);

    vector<int> indices(train.images.size());
    iota(indices.begin(), indices.end(), 0);

    cout << "Starting training...\n\n";

    for (int epoch = 1; epoch <= EPOCHS; ++epoch)
    {
        auto t0 = chrono::steady_clock::now();
        shuffle(indices.begin(), indices.end(), rng);

        float epochLoss = 0.0f;
        int numBatches = train.images.size() / BATCH_SIZE;

        for (int b = 0; b < numBatches; ++b)
        {
            float batchLoss = 0.0f;
            for (int k = 0; k < BATCH_SIZE; ++k)
            {
                int idx = indices[b * BATCH_SIZE + k];
                auto probs = net.forward(train.images[idx]);
                batchLoss += net.backward(train.images[idx], train.labels[idx], probs);
            }
            float lr = LEARNING_RATE * pow(0.5f, (epoch-1)/5);
            net.update(lr, MOMENTUM, L2_LAMBDA, BATCH_SIZE);
            epochLoss += batchLoss / BATCH_SIZE;
        }

        auto t1 = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();

        int correct = 0;
        for (size_t i = 0; i < test.images.size(); ++i)
            if (net.predict(test.images[i]) == test.labels[i]) ++correct;

        cout << "Epoch " << setw(2) << epoch << "/" << EPOCHS
             << "  loss: " << fixed << setprecision(4) << epochLoss/numBatches
             << "  test acc: " << setprecision(2) << (100.0f * correct / test.images.size()) << "%"
             << "  (" << setprecision(1) << elapsed << "s)\n";
    }

    cout << "\nTraining finished!\n";
    return 0;
}