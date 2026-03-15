#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <direct.h>     // For Windows current directory

using namespace std;

// ====================== HELPER: SHOW CURRENT FOLDER ======================
string get_current_dir() {
    char buff[1024];
    _getcwd(buff, 1024);
    return string(buff);
}

// ====================== MNIST LOADER (with debug) ======================
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
    
    cout << "\n=== DEBUG INFO ===\n";
    cout << "Current working directory: " << get_current_dir() << "\n\n";
    cout << "Looking for these exact files:\n";
    cout << "1. train-images-idx3-ubyte\n";
    cout << "2. train-labels-idx1-ubyte\n";
    cout << "3. t10k-images-idx3-ubyte\n";
    cout << "4. t10k-labels-idx1-ubyte\n\n";

    ifstream img(img_file, ios::binary);
    ifstream lbl(lbl_file, ios::binary);

    if (!img.is_open() || !lbl.is_open()) {
        cout << "ERROR: One or more MNIST files are missing or still compressed!\n";
        cout << "→ Make sure the 4 files above are in the folder shown above (no .gz)\n";
        cout << "→ If you have .gz files, right-click each one → Extract All\n";
        exit(1);
    }

    read_big_endian(img); read_big_endian(lbl); // magic
    uint32_t n = read_big_endian(img);
    read_big_endian(img); read_big_endian(img); // rows/cols

    images.resize(n, vector<float>(784));
    labels.resize(n);

    for (uint32_t i = 0; i < n; ++i) {
        for (int j = 0; j < 784; ++j) {
            uint8_t pixel = 0;
            img.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f;
        }
        lbl.read(reinterpret_cast<char*>(&labels[i]), 1);
    }
    cout << "Successfully loaded " << n << " images!\n";
}

// ====================== NEURAL NETWORK ======================
struct MLP {
    // (same as before - kept exactly the same for simplicity)
    vector<float> w1, b1, w2, b2;
    vector<float> hidden, output;

    MLP() {
        mt19937 gen(42);
        uniform_real_distribution<float> dist(-0.1f, 0.1f);
        w1.resize(784 * 64); b1.resize(64);
        w2.resize(64 * 10);  b2.resize(10);
        hidden.resize(64);
        output.resize(10);
        for (auto& v : w1) v = dist(gen);
        for (auto& v : b1) v = dist(gen);
        for (auto& v : w2) v = dist(gen);
        for (auto& v : b2) v = dist(gen);
    }

    void forward(const vector<float>& input) {
        fill(hidden.begin(), hidden.end(), 0.0f);
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 784; ++j)
                hidden[i] += w1[i*784 + j] * input[j];
            hidden[i] += b1[i];
            hidden[i] = max(0.0f, hidden[i]);
        }
        fill(output.begin(), output.end(), 0.0f);
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 64; ++j)
                output[i] += w2[i*64 + j] * hidden[j];
            output[i] += b2[i];
        }
        float maxv = *max_element(output.begin(), output.end());
        float sum = 0.0f;
        for (auto& v : output) { v = expf(v - maxv); sum += v; }
        for (auto& v : output) v /= sum;
    }

    void train(const vector<vector<float>>& images, const vector<uint8_t>& labels,
               int epochs = 5, float lr = 0.01f) {
        vector<float> o_grad(10), h_grad(64);
        for (int e = 0; e < epochs; ++e) {
            float loss = 0.0f; int correct = 0;
            for (size_t i = 0; i < images.size(); ++i) {
                forward(images[i]);
                int true_lbl = labels[i];
                for (int j = 0; j < 10; ++j)
                    o_grad[j] = output[j] - (j == true_lbl ? 1.0f : 0.0f);
                loss -= log(output[true_lbl] + 1e-8f);
                fill(h_grad.begin(), h_grad.end(), 0.0f);
                for (int j = 0; j < 64; ++j) {
                    for (int k = 0; k < 10; ++k)
                        h_grad[j] += w2[k*64 + j] * o_grad[k];
                    h_grad[j] *= (hidden[j] > 0 ? 1.0f : 0.0f);
                }
                for (int j = 0; j < 10; ++j) {
                    b2[j] -= lr * o_grad[j];
                    for (int k = 0; k < 64; ++k)
                        w2[j*64 + k] -= lr * o_grad[j] * hidden[k];
                }
                for (int j = 0; j < 64; ++j) {
                    b1[j] -= lr * h_grad[j];
                    for (int k = 0; k < 784; ++k)
                        w1[j*784 + k] -= lr * h_grad[j] * images[i][k];
                }
                if (max_element(output.begin(), output.end()) - output.begin() == true_lbl) ++correct;
            }
            cout << "Epoch " << e+1 << "/" << epochs
                 << "  Accuracy: " << fixed << setprecision(2) << (correct * 100.0f / images.size()) << "%\n";
        }
    }

    int predict(const vector<float>& input) {
        forward(input);
        return max_element(output.begin(), output.end()) - output.begin();
    }
};

// ====================== MAIN ======================
int main() {
    cout << "=== MNIST Character Detector (Laptop Version) ===\n\n";

    vector<vector<float>> train_img, test_img;
    vector<uint8_t> train_lbl, test_lbl;

    cout << "Loading training data...\n";
    load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", train_img, train_lbl);
    cout << "Loading test data...\n";
    load_mnist("t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte",  test_img,  test_lbl);

    cout << "\nTraining...\n";
    MLP net;
    net.train(train_img, train_lbl, 5, 0.01f);

    // Test accuracy
    int correct = 0;
    for (size_t i = 0; i < test_img.size(); ++i) {
        if (net.predict(test_img[i]) == test_lbl[i]) ++correct;
    }
    cout << "\nFinal test accuracy: " << (correct * 100.0f / test_img.size()) << "%\n\n";

    // Interactive mode
    cout << "=== CHARACTER DETECTION MODE ===\nEnter test image index (0-9999) or -1 to quit.\n";
    while (true) {
        int idx; cout << "Enter index: "; cin >> idx;
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