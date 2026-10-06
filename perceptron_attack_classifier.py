"""
Project 1: Perceptron Classification of Cyberattacks
-----------------------------------------------------
A single-neuron Perceptron learns to separate NORMAL network traffic
from ATTACK traffic (DDoS flood, brute-force login, or both) using two features:

    x1 = requests per second  (scaled 0-1)
    x2 = failed logins per minute (scaled 0-1)

Run:  python perceptron_attack_classifier.py
Needs: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# ------------------------------------------------------------------
# 1. Create a small synthetic dataset
# ------------------------------------------------------------------
def make_data(n_each=60):
    # Normal traffic: few requests, few failed logins
    normal = np.random.normal(loc=[0.25, 0.20], scale=0.08, size=(n_each, 2))
    # Attack traffic comes in three flavours (all labelled "attack"):
    ddos    = np.random.normal(loc=[0.85, 0.20], scale=0.07, size=(n_each // 2, 2))  # flood, few logins
    brute   = np.random.normal(loc=[0.25, 0.85], scale=0.07, size=(n_each // 2, 2))  # brute-force logins
    combo   = np.random.normal(loc=[0.80, 0.80], scale=0.07, size=(n_each // 2, 2))  # both
    attack = np.vstack([ddos, brute, combo])
    X = np.vstack([normal, attack])
    y = np.array([0] * n_each + [1] * len(attack))   # 0 = normal, 1 = attack
    idx = np.random.permutation(len(y))
    return np.clip(X[idx], 0, 1), y[idx]

X, y = make_data()
split = int(0.75 * len(y))
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

# ------------------------------------------------------------------
# 2. The Perceptron
# ------------------------------------------------------------------
class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        # step activation: output 1 if w.x + b >= 0 else 0
        return (X @ self.w + self.b >= 0).astype(int)

    def fit(self, X, y, epochs=20):
        errors_per_epoch = []
        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.predict(xi.reshape(1, -1))[0]
                update = self.lr * (target - pred)      # perceptron learning rule
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            errors_per_epoch.append(errors)
            if errors == 0:                              # converged
                break
        return errors_per_epoch

model = Perceptron(n_features=2, lr=0.1)
errors = model.fit(X_train, y_train, epochs=30)

# ------------------------------------------------------------------
# 3. Results (printed output)
# ------------------------------------------------------------------
train_acc = np.mean(model.predict(X_train) == y_train) * 100
test_acc  = np.mean(model.predict(X_test)  == y_test)  * 100
print("=" * 50)
print(" PERCEPTRON CYBERATTACK CLASSIFIER")
print("=" * 50)
print(f" Epochs until convergence : {len(errors)}")
print(f" Learned weights          : {np.round(model.w, 3)}")
print(f" Learned bias             : {model.b:.3f}")
print(f" Training accuracy        : {train_acc:.1f}%")
print(f" Test accuracy            : {test_acc:.1f}%")

# Classify a few brand-new traffic samples
samples = np.array([[0.20, 0.10],   # quiet traffic
                    [0.90, 0.85],   # flood + many failed logins
                    [0.30, 0.90],   # brute-force style
                    [0.85, 0.20]])  # high volume only
labels = {0: "NORMAL", 1: "ATTACK"}
print("\n New traffic samples:")
for s, p in zip(samples, model.predict(samples)):
    print(f"   req/s={s[0]:.2f}, failed_logins={s[1]:.2f}  ->  {labels[p]}")

# ------------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# (a) Decision boundary
xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax[0].contourf(xx, yy, zz, alpha=0.2, cmap="coolwarm")
ax[0].scatter(*X_train[y_train == 0].T, c="tab:blue", label="Normal (train)", edgecolor="k")
ax[0].scatter(*X_train[y_train == 1].T, c="tab:red",  label="Attack (train)", edgecolor="k")
ax[0].scatter(*X_test[y_test == 0].T, c="tab:blue", marker="s", label="Normal (test)", edgecolor="k")
ax[0].scatter(*X_test[y_test == 1].T, c="tab:red",  marker="s", label="Attack (test)", edgecolor="k")
ax[0].scatter(*samples.T, c="gold", marker="*", s=250, edgecolor="k", label="New samples")
ax[0].set_xlabel("Requests per second (scaled)")
ax[0].set_ylabel("Failed logins per minute (scaled)")
ax[0].set_title(f"Perceptron decision boundary (test acc {test_acc:.0f}%)")
ax[0].legend(fontsize=8, loc="lower right")

# (b) Learning curve
ax[1].plot(range(1, len(errors) + 1), errors, marker="o")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Misclassifications")
ax[1].set_title("Training: errors per epoch")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("perceptron_output.png", dpi=120)
print("\n Plot saved as perceptron_output.png")
plt.show()
