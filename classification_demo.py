"""
GIRLsmarts4tech workshop demo (DRAFT): "Can a computer tell cats from dogs?"

Three short experiments for Grades 6-9, each tied to one idea:
  1. How a model learns from examples (train, predict, evaluate)
  2. Overfitting: memorizing vs. actually learning
  3. Dataset bias: what happens when training examples leave something out

All data is made up (synthetic) so students can focus on the ideas.
Needs: numpy, pandas, scikit-learn.  Run:  python classification_demo.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(42)


# ---------------------------------------------------------------- data
def make_pets(n_cats=300, n_dogs=300):
    """Pretend measurements. Cats are usually lighter with shorter ears,
    but dogs come in many sizes, so the two groups overlap on purpose."""
    cats = pd.DataFrame({
        "weight_kg": rng.normal(4.5, 1.0, n_cats).clip(2, 8),
        "ear_length_cm": rng.normal(6.0, 1.0, n_cats).clip(3, 9),
        "pet": "cat",
    })
    # dogs: a mix of small, medium and large
    sizes = rng.choice(["small", "medium", "large"], n_dogs, p=[0.3, 0.4, 0.3])
    w = np.where(sizes == "small", rng.normal(7, 1.5, n_dogs),
        np.where(sizes == "medium", rng.normal(18, 4, n_dogs),
                 rng.normal(35, 6, n_dogs)))
    e = np.where(sizes == "small", rng.normal(7, 1.2, n_dogs),
        np.where(sizes == "medium", rng.normal(9, 1.5, n_dogs),
                 rng.normal(11, 1.5, n_dogs)))
    dogs = pd.DataFrame({"weight_kg": w.clip(3, 60),
                         "ear_length_cm": e.clip(4, 16),
                         "pet": "dog", "size": sizes})
    return pd.concat([cats, dogs], ignore_index=True)


pets = make_pets()
FEATURES = ["weight_kg", "ear_length_cm"]

# ------------------------------------------------- 1. learn from examples
print("=" * 60)
print("EXPERIMENT 1: Learning from examples")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    pets[FEATURES], pets["pet"], test_size=0.25, stratify=pets["pet"], random_state=1
)
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy on pets it has NEVER seen: {accuracy_score(y_test, pred):.0%}")

# Look at the mistakes: a great discussion starter for students
errors = X_test[pred != y_test].copy()
errors["truth"] = y_test[pred != y_test]
errors["model_said"] = pred[pred != y_test]
print(f"\nThe model got {len(errors)} of {len(y_test)} wrong. A few mistakes:")
print(errors.head(5).round(1).to_string())
print("Why might these be hard? (Hint: look at the weights.)")

# ------------------------------------------------------- 2. overfitting
print("\n" + "=" * 60)
print("EXPERIMENT 2: Memorizing is not learning (overfitting)")
print("=" * 60)

# Use a small training set so memorizing is easy, and flip some labels
# to mimic messy real-world data.
small = pets.sample(80, random_state=3)
noisy_y = small["pet"].copy()
flip = rng.choice(small.index, size=8, replace=False)
noisy_y.loc[flip] = noisy_y.loc[flip].map({"cat": "dog", "dog": "cat"})

rest = pets.drop(small.index)
print(f"{'tree depth':>10} | {'training score':>14} | {'new-data score':>14}")
for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(small[FEATURES], noisy_y)
    tr = accuracy_score(noisy_y, tree.predict(small[FEATURES]))
    te = accuracy_score(rest["pet"], tree.predict(rest[FEATURES]))
    label = "no limit" if depth is None else str(depth)
    print(f"{label:>10} | {tr:>14.0%} | {te:>14.0%}")
print("Watch the training score climb to 100% while the new-data score stops improving and drops.")
print("The model is memorizing the messy examples instead of the pattern.")

# --------------------------------------------------------- 3. dataset bias
print("\n" + "=" * 60)
print("EXPERIMENT 3: Who is missing from the training data? (bias)")
print("=" * 60)

# Reserve the same unseen pets for both models to make the comparison fair.
train_pets, test_pets = train_test_split(
    pets, test_size=0.25, stratify=pets["pet"], random_state=7
)

# The restricted model never learns from small dogs; the comparison model does.
restricted_train = train_pets[
    (train_pets["pet"] == "cat") | (train_pets["size"] != "small")
]

restricted_model = KNeighborsClassifier(n_neighbors=5).fit(
    restricted_train[FEATURES], restricted_train["pet"]
)
representative_model = KNeighborsClassifier(n_neighbors=5).fit(
    train_pets[FEATURES], train_pets["pet"]
)

test_dogs = test_pets[test_pets["pet"] == "dog"]
for size in ["small", "medium", "large"]:
    group = test_dogs[test_dogs["size"] == size]
    without_small = accuracy_score(
        group["pet"], restricted_model.predict(group[FEATURES])
    )
    with_small = accuracy_score(
        group["pet"], representative_model.predict(group[FEATURES])
    )
    print(f"{size:>6} dogs identified correctly ({len(group)} unseen examples) | "
          f"trained WITHOUT small dogs: {without_small:>4.0%} | "
          f"trained WITH small dogs: {with_small:>4.0%}")
print("Discussion: Who is missing from the training data, and who is affected?")
print("This is a simplified demonstration of representation gaps, not a fairness audit.")
