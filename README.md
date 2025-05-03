# Linear Regression From Scratch

This Jupyter notebook implements **linear regression** end-to-end—without using `scikit-learn`’s estimator—so you can see exactly how feature scaling, cost computation, and gradient descent work under the hood.

---

1. **Imports & Data Loading**  
   - Load the **Diabetes** dataset (`m≈442, n=10`) or **California Housing** dataset (`m≈20 000, n=8`) from `sklearn.datasets`.

2. **Helper Functions**  
   - `normalize_features(X)`  
     Standardizes each column of `X` to zero mean, unit variance.  
   - `compute_cost(X, y, theta)`  
     Computes the Mean Squared Error cost:  
     \[ J(\theta) = \tfrac{1}{2m}\sum_i (Xθ - y)^2 \]  
   - `gradient_descent(X, y, theta, alpha, num_iters)`  
     Performs **batch** gradient descent to minimize `J(theta)`.

3. **Demo: Synthetic Data**  
   - Generate a 1-D toy example:  
     \[ y = 4 + 3x + \epsilon \]  
   - Scale features, add an intercept term, run gradient descent, and print learned parameters & final cost.

4. **Demo: Real Dataset**  
   - Swap in `load_diabetes()` or `fetch_california_housing()`.  
   - Normalize all features, add bias column, train, and inspect convergence.
