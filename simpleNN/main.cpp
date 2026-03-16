#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <direct.h>
#include <numeric>

using namespace std;

string get_current_dir() {
    char buff[1024];
    _getcwd(buff, 1024);
    return string(buff);
}

// ====================== MNIST LOADER ======================
uint32_t read_big_endian(ifstream& f) {
    uint32_t val = 0;
    f.read(reinterpret_cast<char*>(&val), 4);
    return ((val & 0x000000FF) << 24) |
           ((val & 0x0000FF00) <<  8) |
           ((val & 0x00FF0000) >>  8) |
           ((val & 0xFF000000) >> 24);
}

void load_mnist(const string& img_file, const string& lbl_file,
                vector<vector<float>>& images, vector<uint8_t>& labels) {
    ifstream img(img_file, ios::binary);
    ifstream lbl(lbl_file, ios::binary);
    if (!img.is_open() || !lbl.is_open()) { cout << "ERROR: Files missing!\n"; exit(1); }

    read_big_endian(img); read_big_endian(lbl);
    uint32_t n = read_big_endian(img);
    read_big_endian(img); read_big_endian(img);

    images.resize(n, vector<float>(784));
    labels.resize(n);

    for (uint32_t i = 0; i < n; ++i) {
        for (int j = 0; j < 784; ++j) {
            uint8_t p = 0; img.read(reinterpret_cast<char*>(&p), 1);
            images[i][j] = p / 255.0f;
        }
        lbl.read(reinterpret_cast<char*>(&labels[i]), 1);
    }
    cout << "Loaded " << n << " images successfully.\n";
}

// ====================== NEURAL NETWORK (Mini-Batch + Xavier) ======================
struct MLP {
    vector<float> w1, b1, w2, b2;
    vector<float> hidden, output;
    mt19937 gen;

    MLP() : gen(42) {
        float limit1 = sqrt(6.0f / (784 + 64));
        float limit2 = sqrt(6.0f / (64 + 10));
        uniform_real_distribution<float> dist1(-limit1, limit1);
        uniform_real_distribution<float> dist2(-limit2, limit2);

        w1.resize(784*64); b1.resize(64);
        w2.resize(64*10);  b2.resize(10);
        hidden.resize(64); output.resize(10);

        for (auto& v : w1) v = dist1(gen);
        for (auto& v : b1) v = 0.0f;
        for (auto& v : w2) v = dist2(gen);
        for (auto& v : b2) v = 0.0f;
    }

    void forward(const vector<float>& input) {
        fill(hidden.begin(), hidden.end(), 0.0f);
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 784; ++j) hidden[i] += w1[i*784 + j] * input[j];
            hidden[i] += b1[i];
            hidden[i] = max(0.0f, hidden[i]);
        }
        fill(output.begin(), output.end(), 0.0f);
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 64; ++j) output[i] += w2[i*64 + j] * hidden[j];
            output[i] += b2[i];
        }
        float maxv = *max_element(output.begin(), output.end());
        float sum = 0.0f;
        for (auto& v : output) { v = expf(v - maxv); sum += v; }
        for (auto& v : output) v /= sum;
    }

    int predict(const vector<float>& input) {
        forward(input);
        return max_element(output.begin(), output.end()) - output.begin();
    }

    void train(const vector<vector<float>>& images, const vector<uint8_t>& labels,
               int epochs = 5, float lr = 0.1f, int batch_size = 64) {
        vector<size_t> indices(images.size());
        iota(indices.begin(), indices.end(), 0);
        vector<float> o_grad(10), h_grad(64);

        for (int e = 0; e < epochs; ++e) {
            shuffle(indices.begin(), indices.end(), gen);
            int correct = 0;

            for (size_t start = 0; start < images.size(); start += batch_size) {
                int end = min(start + batch_size, images.size());
                int batch_n = end - start;

                vector<float> dw1(w1.size(), 0.0f), db1(b1.size(), 0.0f);
                vector<float> dw2(w2.size(), 0.0f), db2(b2.size(), 0.0f);

                for (size_t b = start; b < end; ++b) {
                    size_t i = indices[b];
                    forward(images[i]);
                    int true_lbl = labels[i];

                    for (int j = 0; j < 10; ++j)
                        o_grad[j] = output[j] - (j == true_lbl ? 1.0f : 0.0f);

                    fill(h_grad.begin(), h_grad.end(), 0.0f);
                    for (int j = 0; j < 64; ++j) {
                        for (int k = 0; k < 10; ++k)
                            h_grad[j] += w2[k*64 + j] * o_grad[k];
                        h_grad[j] *= (hidden[j] > 0.0f ? 1.0f : 0.0f);
                    }

                    for (int j = 0; j < 10; ++j) {
                        db2[j] += o_grad[j];
                        for (int k = 0; k < 64; ++k)
                            dw2[j*64 + k] += o_grad[j] * hidden[k];
                    }
                    for (int j = 0; j < 64; ++j) {
                        db1[j] += h_grad[j];
                        for (int k = 0; k < 784; ++k)
                            dw1[j*784 + k] += h_grad[j] * images[i][k];
                    }

                    if (max_element(output.begin(), output.end()) - output.begin() == true_lbl)
                        ++correct;
                }

                // Apply averaged gradients
                float scale = lr / batch_n;
                for (size_t i = 0; i < w1.size(); ++i) w1[i] -= scale * dw1[i];
                for (size_t i = 0; i < b1.size(); ++i) b1[i] -= scale * db1[i];
                for (size_t i = 0; i < w2.size(); ++i) w2[i] -= scale * dw2[i];
                for (size_t i = 0; i < b2.size(); ++i) b2[i] -= scale * db2[i];
            }
            cout << "Epoch " << e+1 << "/" << epochs
                 << "  Train Accuracy: " << fixed << setprecision(2)
                 << (correct * 100.0f / images.size()) << "%\n";
        }
    }
};

// ====================== MAIN ======================
int main() {
    cout << "=== MNIST Character Detector (Working Mini-Batch Version) ===\n\n";

    vector<vector<float>> train_img, test_img;
    vector<uint8_t> train_lbl, test_lbl;

    load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", train_img, train_lbl);
    load_mnist("t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte",  test_img,  test_lbl);

    cout << "\nTraining (5 epochs - ~2 minutes)...\n";
    MLP net;
    net.train(train_img, train_lbl, 5, 0.1f, 64);

    int correct = 0;
    for (size_t i = 0; i < test_img.size(); ++i) {
        if (net.predict(test_img[i]) == test_lbl[i]) ++correct;
    }
    cout << "\nFinal test accuracy: " << (correct * 100.0f / test_img.size()) << "%\n\n";

    cout << "=== CHARACTER DETECTION MODE ===\nEnter index (0-9999) or -1 to quit:\n";
    while (true) {
        int idx; cout << "Index: "; cin >> idx;
        if (idx == -1) break;
        if (idx < 0 || idx >= (int)test_img.size()) { cout << "Invalid!\n"; continue; }

        int pred = net.predict(test_img[idx]);
        cout << "\nPredicted: " << pred << " (True: " << (int)test_lbl[idx] << ")\n\n";

        cout << "Digit image:\n";
        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c)
                cout << (test_img[idx][r*28 + c] > 0.3f ? "##" : "  ");
            cout << "\n";
        }
        cout << "────────────────────────────────────────\n\n";
    }
    return 0;
}