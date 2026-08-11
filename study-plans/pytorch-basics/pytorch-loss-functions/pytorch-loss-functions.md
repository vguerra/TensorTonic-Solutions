# <span style="font-size: 20px;">Loss Functions</span>

<span style="font-size: 14px;">A loss function turns the difference between a model's prediction and the correct answer into one number. A smaller value means the prediction is closer to what the selected loss considers correct. This problem implements three losses with different purposes: mean squared error for numeric predictions, cross-entropy for choosing among classes, and Huber loss for numeric predictions that may contain unusually large errors.</span>

---

## <span style="font-size: 16px;">What the Function Must Produce</span>

<span style="font-size: 14px;">Each method first computes a loss for every element or sample, then returns the mean. The final result is a Python float rather than a tensor. The methods share one function interface, but they do not share the same interpretation of predictions and targets.</span>

* <span style="font-size: 14px;">**MSE:** predictions and targets are matching numeric values. Every pair contributes one squared error.</span>
* <span style="font-size: 14px;">**Cross-entropy:** each prediction row contains one raw score for every class. Each target is the integer index of the correct class.</span>
* <span style="font-size: 14px;">**Huber:** predictions and targets are matching numeric values. Small errors use a squared penalty, while large errors use a linear penalty.</span>

<span style="font-size: 14px;">This distinction matters because a classification logit is not a numeric prediction to compare directly with a class index. The class index selects one entry from a row of logits, while MSE and Huber compare prediction and target values element by element.</span>

---

## <span style="font-size: 16px;">Mean Squared Error</span>

<span style="font-size: 14px;">Mean squared error, usually abbreviated as **MSE**, measures the average squared distance between predictions and targets. For prediction $\hat{y}_i$, target $y_i$, and $n$ values, it is</span>

$$
L_{\mathrm{MSE}}=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

<span style="font-size: 14px;">The subtraction produces an error for each value. Squaring makes every contribution non-negative, so errors cannot cancel each other. It also makes a large error grow much faster than a small one. An error of $2$ contributes $4$, while an error of $4$ contributes $16$.</span>

<span style="font-size: 14px;">MSE is useful when large misses should receive a strong penalty. That same behavior makes it sensitive to outliers. One extreme prediction can dominate the mean even when the other predictions are close to their targets.</span>

### <span style="font-size: 14px;">MSE example</span>

<span style="font-size: 14px;">For predictions $[1,2,3]$ and targets $[1.5,2.5,3.5]$, every error is $-0.5$. Squaring produces $[0.25,0.25,0.25]$, and their mean is</span>

$$
\frac{0.25+0.25+0.25}{3}=0.25
$$

<span style="font-size: 14px;">The sign of the original error does not affect MSE. Predictions that are equally far above or below their targets contribute the same squared loss.</span>

---

## <span style="font-size: 16px;">Cross-Entropy from Logits</span>

<span style="font-size: 14px;">Cross-entropy is used when the model must choose one class from several possibilities. Each sample provides a row of **logits**, which are unrestricted model scores. Logits may be positive, negative, or larger than one. They are not probabilities and should not be expected to sum to one.</span>

<span style="font-size: 14px;">For sample $i$, class logits $z_{i,j}$, and correct class index $t_i$, the loss is</span>

$$
L_i=\log\left(\sum_j e^{z_{i,j}}\right)-z_{i,t_i}
$$

<span style="font-size: 14px;">The first term considers every class score. The second subtracts the score of the correct class. When the correct class score is much larger than the alternatives, the loss approaches zero. When an incorrect class has a much larger score, the loss becomes large.</span>

<span style="font-size: 14px;">The target is used as an index. If a row has three class logits and its target is $0$, the first logit is the correct-class logit. A target of $2$ selects the third logit.</span>

<span style="font-size: 14px;">After calculating one loss per sample, take their mean:</span>

$$
L_{\mathrm{CE}}=\frac{1}{N}\sum_{i=1}^{N}L_i
$$

### <span style="font-size: 14px;">Cross-entropy example</span>

<span style="font-size: 14px;">Consider logits $[2,1,0.1]$ with target class $0$. The correct class has logit $2$. The direct expression is</span>

$$
\log(e^2+e^1+e^{0.1})-2\approx0.417
$$

<span style="font-size: 14px;">The loss is fairly small because the correct class has the largest logit, but it is not zero because the other classes still receive some probability.</span>

---

## <span style="font-size: 16px;">Why Cross-Entropy Needs Log-Sum-Exp</span>

<span style="font-size: 14px;">Computing exponentials directly can overflow. A logit of $1000$ is a valid model score, but $e^{1000}$ is too large for ordinary floating-point storage. The calculation can be stabilized by subtracting the largest logit in each row before exponentiation.</span>

<span style="font-size: 14px;">Let $m_i$ be the largest logit in row $i$. Then</span>

$$
\log\left(\sum_j e^{z_{i,j}}\right)
=m_i+\log\left(\sum_j e^{z_{i,j}-m_i}\right)
$$

<span style="font-size: 14px;">Subtracting the same value from every logit does not change the softmax distribution. It only makes the largest shifted logit equal to zero, so every exponential lies between zero and one. The stable per-sample loss becomes</span>

$$
L_i=m_i+\log\left(\sum_j e^{z_{i,j}-m_i}\right)-z_{i,t_i}
$$

<span style="font-size: 14px;">The maximum is computed separately for every sample row. A single maximum over the entire batch would mix unrelated samples and can leave some rows poorly scaled.</span>

<span style="font-size: 14px;">The problem asks for the log-sum-exp calculation to be implemented explicitly. Applying softmax first and taking its logarithm later is mathematically related, but it recreates the numerical problem this exercise is intended to avoid.</span>

---

## <span style="font-size: 16px;">Huber Loss</span>

<span style="font-size: 14px;">Huber loss is a compromise between squared error and absolute error. It treats small errors quadratically, which gives a smooth curve near the correct value, and treats large errors linearly, which prevents outliers from dominating as strongly as they do under MSE.</span>

<span style="font-size: 14px;">For absolute error $a=|\hat{y}-y|$ and positive threshold $\delta$, the element-wise loss is</span>

$$
L_{\delta}(a)=
\begin{cases}
\frac{1}{2}a^2, & a\leq\delta \\
\delta\left(a-\frac{1}{2}\delta\right), & a>\delta
\end{cases}
$$

<span style="font-size: 14px;">The threshold $\delta$ decides where the behavior changes. Errors at or below the threshold use the quadratic branch. Errors above it use the linear branch. Both formulas give $\frac{1}{2}\delta^2$ at the boundary, so the loss does not jump when the branch changes.</span>

<span style="font-size: 14px;">A smaller threshold moves more errors into the linear region and increases robustness to outliers. A larger threshold makes Huber behave like a squared loss over a wider range.</span>

### <span style="font-size: 14px;">Huber example</span>

<span style="font-size: 14px;">For predictions $[1,5]$, targets $[2,2]$, and $\delta=1$, the absolute errors are $[1,3]$.</span>

* <span style="font-size: 14px;">The first error is exactly at the threshold, so its loss is $\frac{1}{2}(1)^2=0.5$.</span>
* <span style="font-size: 14px;">The second error is above the threshold, so its loss is $1(3-0.5)=2.5$.</span>

<span style="font-size: 14px;">The returned mean is</span>

$$
\frac{0.5+2.5}{2}=1.5
$$

<span style="font-size: 14px;">For comparison, MSE on the same errors would be $(1^2+3^2)/2=5$. Huber reduces the influence of the larger error without ignoring it.</span>

---

## <span style="font-size: 16px;">From Element-Wise Losses to One Float</span>

<span style="font-size: 14px;">All three methods return a mean rather than a list of losses. Reduction happens only after the correct element-wise or per-sample values have been computed.</span>

* <span style="font-size: 14px;">MSE averages squared errors across all prediction-target pairs.</span>
* <span style="font-size: 14px;">Cross-entropy calculates one loss per row of class logits, then averages across rows.</span>
* <span style="font-size: 14px;">Huber selects the appropriate branch for every absolute error, then averages the resulting values.</span>

<span style="font-size: 14px;">PyTorch reduction operations return scalar tensors. The required return type is a Python float, so the scalar tensor must be converted after the mean is calculated. Converting earlier would discard the tensor operations needed to compute the loss.</span>

<span style="font-size: 14px;">Numeric targets for MSE and Huber use floating-point values. Cross-entropy targets represent positions in the class dimension, so they use integer indices. Keeping these meanings separate avoids silent conversions and invalid indexing.</span>

---

## <span style="font-size: 16px;">Implementation Order</span>

* <span style="font-size: 14px;">Convert predictions to a floating-point tensor.</span>
* <span style="font-size: 14px;">For MSE, convert targets to floating point, square element-wise differences, and take the mean.</span>
* <span style="font-size: 14px;">For cross-entropy, convert targets to integer class indices, compute a stable log-sum-exp for every row, subtract the selected correct-class logits, and take the batch mean.</span>
* <span style="font-size: 14px;">For Huber, convert targets to floating point, calculate absolute errors, choose the quadratic or linear expression for each error, and take the mean.</span>
* <span style="font-size: 14px;">Convert the final scalar tensor to a Python float.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Treating logits as probabilities.** Cross-entropy receives raw scores. Applying an unnecessary probability conversion and then repeating the exponential calculation changes the intended computation and can reduce numerical stability.</span>
* <span style="font-size: 14px;">**Using floating-point class targets.** Cross-entropy targets in this problem are integer class indices used to select one logit from each row.</span>
* <span style="font-size: 14px;">**Computing an unstable exponential sum.** Exponentiating large logits directly can produce infinity. Subtract the maximum of each row before exponentiation.</span>
* <span style="font-size: 14px;">**Applying one Huber branch to the whole tensor.** Branch selection is element-wise because one batch can contain errors on both sides of $\delta$.</span>
* <span style="font-size: 14px;">**Forgetting the mean or the return conversion.** The required result is one mean loss returned as a Python float.</span>

---