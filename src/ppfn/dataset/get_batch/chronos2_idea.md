# Chronos based ideas: Multitask Priors for Hyperparameter Optimization

Similarily to the Multivariatizer in Chronos-2, we can define
multitask priors for Hyperparameter Optimization (HPO) benchmarks based on
Bayesian Neural Networks (BNNs). The goal is to create a set of related tasks
that share some underlying structure, allowing for transfer learning and improved
performance across tasks.

# A. The "Output-Mixing" Prior (The Chronos-2 approach)
This is the most direct port of the Chronos-2
multivariatizer. 
* Sample $N$ independent BNNs $g_1(x), \dots, g_n(x)$ using different architectures (to mimic KernelSynth
flexibility). 
* Sample a random Mixing Matrix $M$.
* The $K$ tasks are defined
as: $\begin{bmatrix} f_1(x) \\ \vdots \\ f_k(x) \end{bmatrix} = M \begin{bmatrix} g_1(x) \\ \vdots \\ g_n(x) \end{bmatrix}$
* The "Flexibility" trick: If you use non-linear mixing (e.g., passing the result through a Copula or a Tanh), you can
model tasks where the correlation is only high in the "optima" regions (the peaks), which is crucial for HPO.

# B. The "Latent Embedding" Prior (Most Flexible)
This treats the BNN as a universal function that takes both the
Hyperparameters $(x)$ and a Task Latent $(z)$ as input.
* $f(x, z) = BNN(x, z; \theta_{global})$. 
* During data generation,
you sample a set of $z_k$ vectors that are clustered or follow a manifold.
* Why this is the "best": This allows the
relationship between tasks to be input-dependent. For example, Task A and Task B might be very similar when the Learning
Rate is low, but behave completely differently when it's high.