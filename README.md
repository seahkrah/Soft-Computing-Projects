# Soft Computing Mini-Projects: Neural Networks for Cyberattack Detection

Two small, beginner-friendly neural network projects written in Python. Both use simple cyberattack scenarios to show how classic soft computing models work, 
and both produce printed results and plots when run.

| # | Project | Model | Task |
|---|---------|-------|------|
| 1 | `perceptron_attack_classifier.py` | Perceptron | **Classifies** network traffic as NORMAL or ATTACK |
| 2 | `hopfield_attack_pattern_recognition.py` | Hopfield network + Hebbian rule | **Recognizes** attack types from noisy patterns |

---

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

Install with:

```bash
pip install numpy matplotlib
```

## How to run

```bash
python perceptron_attack_classifier.py
python hopfield_attack_pattern_recognition.py
```

Each script prints its results in the terminal, saves a PNG plot in the same folder, and opens the plot in a window.

| Script | Saved plot |
|--------|-----------|
| Perceptron | `perceptron_output.png` |
| Hopfield | `hopfield_output.png` |

---

## Project 1: Perceptron Cyberattack Classifier

**File:** `perceptron_attack_classifier.py`

### Idea
A single artificial neuron learns a straight line that separates normal traffic from attack traffic.

### Features (inputs)
| Feature | Meaning |
|---------|---------|
| `x1` | Requests per second (scaled 0 to 1) |
| `x2` | Failed logins per minute (scaled 0 to 1) |

### Classes (output)
- `0` = NORMAL (low requests, low failed logins)
- `1` = ATTACK, in three flavours:
  - DDoS flood: high requests, few failed logins
  - Brute-force: few requests, many failed logins
  - Combined: both high

The data is synthetic (randomly generated clusters), split 75% training and 25% testing.

### How it works
1. Compute `w1*x1 + w2*x2 + b`.
2. If the result is 0 or more, predict ATTACK. Otherwise predict NORMAL (step activation).
3. For each training sample, apply the **perceptron learning rule**:
   `w = w + lr * (target - prediction) * x`
4. Repeat for several epochs until there are no more mistakes.

### Sample output
```
 Epochs until convergence : 2
 Training accuracy        : 100.0%
 Test accuracy            : 97.4%

 New traffic samples:
   req/s=0.20, failed_logins=0.10  ->  NORMAL
   req/s=0.90, failed_logins=0.85  ->  ATTACK
   req/s=0.30, failed_logins=0.90  ->  ATTACK
   req/s=0.85, failed_logins=0.20  ->  ATTACK
```

### Plots
- **Left:** the decision boundary with training points, test points, and new samples (stars).
- **Right:** number of misclassifications per epoch (the learning curve).

---

## Project 2: Hopfield Attack Pattern Recognition

**File:** `hopfield_attack_pattern_recognition.py`

### Idea
A Hopfield network is an associative memory. It stores a few patterns, and when given a damaged or noisy version of one, it "falls back" to the closest stored pattern.

### Stored attack signatures (7x7 grids = 49 neurons)
| Attack | Pattern | Why this shape |
|--------|---------|----------------|
| DDoS | Horizontal bars | A flood of repeated requests |
| Phishing | Vertical stripes | Many look-alike pages and links |
| SQL Injection | X shape | Malicious input crossing the query |

These shapes are symbolic. They were chosen to be clearly different from each other, because Hopfield networks work best when stored patterns do not overlap much.

### How it works
1. **Encode:** `#` becomes `+1` and `.` becomes `-1`.
2. **Hebbian learning:** build the weight matrix in one step:
   `W = (1/N) * sum(p * p^T)`, with the diagonal set to 0.
3. **Add noise:** randomly flip some pixels of a stored pattern.
4. **Recall:** update neurons one at a time using `s_i = sign(sum_j W_ij * s_j)` until the state stops changing.
5. **Identify:** compare the final state to each stored pattern and pick the best match.

### Sample output
```
 Input: noisy DDoS           -> Recognized: DDoS           (match 100.0%)  [CORRECT]
 Input: noisy Phishing       -> Recognized: Phishing       (match 100.0%)  [CORRECT]
 Input: noisy SQL Injection  -> Recognized: SQL Injection  (match 100.0%)  [CORRECT]
```

It also prints accuracy against noise level. In the latest run, recognition stayed at 100% up to 14 flipped pixels (about 29% noise) and dropped to 70% at 20 flipped pixels. Exact numbers vary slightly between runs because the noise is random.

### Plots
- **Top row:** the three stored attack patterns.
- **Middle rows:** the noisy input and the recalled output for each attack.
- **Bottom:** a robustness curve of recognition accuracy versus amount of noise.

---

## Things to try

**Perceptron**
- Change `lr` (learning rate) or `epochs` and watch the learning curve change.
- Move the cluster centres in `make_data()` closer together to see accuracy drop.
- Add a third feature such as packet size and extend the weights.

**Hopfield**
- Increase `NOISE_BITS` to make recall harder.
- Add a fourth attack pattern and see how recall degrades. This shows the network's limited memory capacity, roughly 0.14 x N patterns.
- Draw patterns that look similar and watch the network confuse them.

## Limitations

- The data and patterns are **synthetic**, for learning purposes only. They are not a real intrusion detection system.
- A Perceptron can only learn **linearly separable** classes. Data that needs a curved boundary requires a multi-layer network.
- A Hopfield network has limited capacity and can settle on a wrong or "spurious" pattern when the stored patterns are too similar or the noise is too high.

## Files in this folder

```
perceptron_attack_classifier.py
hopfield_attack_pattern_recognition.py
perceptron_output.png      (created when you run project 1)
hopfield_output.png        (created when you run project 2)
README.md
```
