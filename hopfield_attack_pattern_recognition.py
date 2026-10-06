"""
Project 2: Hopfield Network + Hebbian Learning for Attack Pattern Recognition
------------------------------------------------------------------------------
A Hopfield network memorizes three attack "signatures" (7x7 binary patterns)
using the Hebbian rule. When given a noisy / corrupted version of a signature,
the network recalls the closest stored attack pattern.

    D = DDoS      P = Phishing      S = SQL Injection

Hebbian rule :  W = (1/N) * sum_over_patterns( p p^T ),  diagonal = 0
Update rule  :  s_i <- sign( sum_j W_ij s_j )

Run:  python hopfield_attack_pattern_recognition.py
Needs: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

# ------------------------------------------------------------------
# 1. Attack signatures (7x7 grids, '#' = +1, '.' = -1)
# ------------------------------------------------------------------
RAW = {
    "DDoS": [            # horizontal bars = a flood of repeated requests
        "#######",
        ".......",
        "#######",
        ".......",
        "#######",
        ".......",
        "#######",
    ],
    "Phishing": [        # vertical stripes = many look-alike pages / links
        "#.#.#.#",
        "#.#.#.#",
        "#.#.#.#",
        "#.#.#.#",
        "#.#.#.#",
        "#.#.#.#",
        "#.#.#.#",
    ],
    "SQL Injection": [   # X shape = malicious input crossing the query
        "#.....#",
        ".#...#.",
        "..#.#..",
        "...#...",
        "..#.#..",
        ".#...#.",
        "#.....#",
    ],
}
names = list(RAW.keys())

def to_vec(rows):
    return np.array([1 if c == "#" else -1 for r in rows for c in r])

patterns = np.array([to_vec(RAW[n]) for n in names])   # shape (3, 49)
N = patterns.shape[1]

# ------------------------------------------------------------------
# 2. Hebbian learning: build the weight matrix in one step
# ------------------------------------------------------------------
W = np.zeros((N, N))
for p in patterns:
    W += np.outer(p, p)
W /= N
np.fill_diagonal(W, 0)                                  # no self-connections

# ------------------------------------------------------------------
# 3. Recall (asynchronous updates until the state stops changing)
# ------------------------------------------------------------------
def recall(state, max_sweeps=20):
    s = state.copy()
    for _ in range(max_sweeps):
        changed = False
        for i in np.random.permutation(N):
            new = 1 if W[i] @ s >= 0 else -1
            if new != s[i]:
                s[i], changed = new, True
        if not changed:
            break
    return s

def add_noise(p, n_flip):
    noisy = p.copy()
    flip = np.random.choice(N, n_flip, replace=False)
    noisy[flip] *= -1
    return noisy

def identify(state):
    overlap = patterns @ state / N                       # 1.0 = perfect match
    k = int(np.argmax(overlap))
    return names[k], overlap[k]

# ------------------------------------------------------------------
# 4. Experiment: corrupt each attack signature and recall it
# ------------------------------------------------------------------
print("=" * 55)
print(" HOPFIELD ATTACK PATTERN RECOGNITION (Hebbian rule)")
print("=" * 55)
print(f" Neurons: {N}   Stored attack patterns: {len(names)}\n")

NOISE_BITS = 8                                           # ~16% of pixels flipped
rows = []
for name, p in zip(names, patterns):
    noisy = add_noise(p, NOISE_BITS)
    recalled = recall(noisy)
    guess, score = identify(recalled)
    ok = "CORRECT" if guess == name else "WRONG"
    print(f" Input: noisy {name:<14} -> Recognized: {guess:<14} "
          f"(match {score*100:5.1f}%)  [{ok}]")
    rows.append((name, noisy, recalled, guess))

# Robustness test: accuracy vs. amount of noise
print("\n Recognition accuracy vs. noise level (50 trials each):")
levels = list(range(0, 21, 2))
accs = []
for k in levels:
    hits = 0
    for _ in range(50):
        t = np.random.randint(len(names))
        out = recall(add_noise(patterns[t], k))
        hits += identify(out)[0] == names[t]
    accs.append(hits / 50 * 100)
    print(f"   {k:2d} flipped pixels ({k/N*100:4.1f}%) -> {accs[-1]:5.1f}% correct")

# ------------------------------------------------------------------
# 5. Visualization
# ------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.4])

for c, (name, p) in enumerate(zip(names, patterns)):
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(p.reshape(7, 7), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"Stored: {name}", fontsize=10)

for c, (name, noisy, recalled, guess) in enumerate(rows):
    ax = fig.add_subplot(gs[1, c])
    ax.imshow(noisy.reshape(7, 7), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"Noisy input ({NOISE_BITS} bits flipped)", fontsize=9)
    ax = fig.add_subplot(gs[2, c])
    ax.imshow(recalled.reshape(7, 7), cmap="gray_r"); ax.axis("off")
    ax.set_title(f"Recalled -> {guess}", fontsize=9,
                 color="green" if guess == name else "red")

ax = fig.add_subplot(gs[3, :])
ax.plot(levels, accs, marker="o")
ax.set_xlabel("Number of flipped pixels (noise)")
ax.set_ylabel("Correct recognition (%)")
ax.set_title("Robustness: how much noise can the network tolerate?")
ax.set_ylim(0, 105); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hopfield_output.png", dpi=120)
print("\n Plot saved as hopfield_output.png")
plt.show()
