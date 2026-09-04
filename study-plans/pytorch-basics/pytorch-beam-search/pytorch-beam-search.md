## <span style="font-size: 18px;">Beam Search Decoding</span>

### <span style="font-size: 16px;">Overview</span>

<span style="font-size: 14px;">Beam search is a breadth-limited search algorithm widely used in sequence generation tasks. It balances the trade-off between exhaustive search (which is computationally infeasible for large vocabularies) and greedy search (which may miss globally optimal sequences).</span>

### <span style="font-size: 16px;">Key Concepts</span>

* <span style="font-size: 14px;">**Beam**: a partial sequence of tokens paired with its cumulative log-probability score</span>
* <span style="font-size: 14px;">**Beam width** ($k$): the number of top-scoring candidates retained at each step</span>
* <span style="font-size: 14px;">**Log-probability scoring**: sequences are scored by summing the log-probabilities of individual token predictions, which is equivalent to maximizing the joint probability</span>

$$
\text{score}(\mathbf{y}) = \sum_{t=1}^{T} \log P(y_t \mid y_1, \ldots, y_{t-1})
$$

### <span style="font-size: 16px;">Algorithm Steps</span>

* <span style="font-size: 14px;">Initialize with a single beam containing the start token and score 0</span>
* <span style="font-size: 14px;">At each time step:</span>
    * <span style="font-size: 14px;">For each active beam, compute log-probabilities over the full vocabulary</span>
    * <span style="font-size: 14px;">Generate all possible one-token extensions, updating cumulative scores</span>
    * <span style="font-size: 14px;">Retain only the top $k$ candidates</span>
    * <span style="font-size: 14px;">Move any beam ending with the end token to a completed list</span>
* <span style="font-size: 14px;">Terminate when all beams are complete or max length is reached</span>
* <span style="font-size: 14px;">Return the highest-scoring sequence from both completed and active beams</span>

### <span style="font-size: 16px;">Greedy vs. Beam Search</span>

* <span style="font-size: 14px;">Greedy decoding ($k=1$) selects the single most probable token at each step</span>
* <span style="font-size: 14px;">Greedy can get trapped in locally optimal but globally suboptimal paths</span>
* <span style="font-size: 14px;">Beam search with $k > 1$ explores multiple hypotheses in parallel, often finding better overall sequences</span>
* <span style="font-size: 14px;">Increasing $k$ improves quality but increases computation linearly</span>

### <span style="font-size: 16px;">Scoring and Comparison</span>

* <span style="font-size: 14px;">Log-probabilities are used instead of raw probabilities to avoid numerical underflow from multiplying many small numbers</span>
* <span style="font-size: 14px;">Since $\log$ is monotonic, maximizing the sum of log-probabilities is equivalent to maximizing the product of probabilities</span>

$$
\arg\max_{\mathbf{y}} \prod_{t} P(y_t \mid y_{<t}) = \arg\max_{\mathbf{y}} \sum_{t} \log P(y_t \mid y_{<t})
$$

### <span style="font-size: 16px;">Handling End Tokens</span>

* <span style="font-size: 14px;">When a beam generates the end-of-sequence token, it is considered complete</span>
* <span style="font-size: 14px;">Completed beams are stored separately and not expanded further</span>
* <span style="font-size: 14px;">This allows shorter sequences to compete fairly with longer ones</span>
* <span style="font-size: 14px;">The end token itself is excluded from the final output</span>

### <span style="font-size: 16px;">Practical Considerations</span>

* <span style="font-size: 14px;">Typical beam widths in practice range from 2 to 10</span>
* <span style="font-size: 14px;">Very large beam widths show diminishing returns and may even degrade output quality in neural language models</span>
* <span style="font-size: 14px;">Length normalization can be applied to prevent bias toward shorter sequences, though the basic algorithm does not include it</span>
* <span style="font-size: 14px;">Beam search is deterministic given the same scoring function, unlike sampling-based methods</span>

### <span style="font-size: 16px;">Complexity</span>

* <span style="font-size: 14px;">Time complexity per step: $O(k \cdot V)$ where $V$ is vocabulary size</span>
* <span style="font-size: 14px;">Total time: $O(T \cdot k \cdot V)$ where $T$ is the maximum sequence length</span>
* <span style="font-size: 14px;">Space: $O(k \cdot T)$ to store the active beams</span>