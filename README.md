# gaussian-process-implicit-surface-tutorial

## Gaussian Process Regression

Q: What it is?
A: 
**Weight Matrix View**
> how to express a function in a linear combinition of other functions
* Objective: Learning w:
$$
f(\mathbf{x})=\mathbf{x}^{\top} \mathbf{w}, \quad y=f(\mathbf{x})+\varepsilon
$$
i.e. choosing w to maximize the probability in training set:
$$
\begin{aligned}
p(\mathbf{y} \mid X, \mathbf{w}) &=\prod_{i=1}^{n} p\left(y_{i} \mid \mathbf{x}_{i}, \mathbf{w}\right)=\prod_{i=1}^{n} \frac{1}{\sqrt{2 \pi} \sigma_{n}} \exp \left(-\frac{\left(y_{i}-\mathbf{x}_{i}^{\top} \mathbf{w}\right)^{2}}{2 \sigma_{n}^{2}}\right) \\
&=\frac{1}{\left(2 \pi \sigma_{n}^{2}\right)^{n / 2}} \exp \left(-\frac{1}{2 \sigma_{n}^{2}}\left|\mathbf{y}-X^{\top} \mathbf{w}\right|^{2}\right)=\mathcal{N}\left(X^{\top} \mathbf{w}, \sigma_{n}^{2} I\right)
\end{aligned}
$$
* How to get w:
View w as a Gaussian Distribution, we update the posterior distribution of w by Bayes's rule (MAP):
$$
\mathbf{w} \sim \mathcal{N}\left(\mathbf{0}, \Sigma_{p}\right) \\
p(\mathbf{w} \mid \mathbf{y}, X)=\frac{p(\mathbf{y} \mid X, \mathbf{w}) p(\mathbf{w})}{p(\mathbf{y} \mid X)} \sim \mathcal{N}\left(\overline{\mathbf{w}}=\frac{1}{\sigma_{n}^{2}} A^{-1} X \mathbf{y}, A^{-1}\right), A=\sigma_{n}^{-2} X X^{\top}+\Sigma_{p}^{-1}
$$
* We can predict without getting w:
$$
\begin{aligned}
p\left(f_{*} \mid \mathbf{x}_{*}, X, \mathbf{y}\right) &=\int p\left(f_{*} \mid \mathbf{x}_{*}, \mathbf{w}\right) p(\mathbf{w} \mid X, \mathbf{y}) d \mathbf{w} \\
&=\mathcal{N}\left(\frac{1}{\sigma_{n}^{2}} \mathbf{x}_{*}^{\top} A^{-1} X \mathbf{y}, \mathbf{x}_{*}^{\top} A^{-1} \mathbf{x}_{*}\right)
\end{aligned}
$$

* Generalize to non-linear case: using base function to replace x:

$$
f(\mathbf{x})=\phi(\mathbf{x})^{\top} \mathbf{w} \\
\begin{aligned}
\mathbb{E}[f(\mathbf{x})] &=\boldsymbol{\phi}(\mathbf{x})^{\top} \mathbb{E}[\mathbf{w}]=0 \\
\mathbb{E}\left[f(\mathbf{x}) f\left(\mathbf{x}^{\prime}\right)\right] &=\boldsymbol{\phi}(\mathbf{x})^{\top} \mathbb{E}\left[\mathbf{w} \mathbf{w}^{\top}\right] \boldsymbol{\phi}\left(\mathbf{x}^{\prime}\right)=\boldsymbol{\phi}(\mathbf{x})^{\top} \Sigma_{p} \phi\left(\mathbf{x}^{\prime}\right)
\end{aligned} \\
\begin{aligned} 
f_{*} \mid \mathbf{x}_{*}, X, \mathbf{y} \sim \mathcal{N}(& \phi_{*}^{\top} \Sigma_{p} \Phi\left(K+\sigma_{n}^{2} I\right)^{-1} \mathbf{y} \\
&\left.\phi_{*}^{\top} \Sigma_{p} \phi_{*}-\phi_{*}^{\top} \Sigma_{p} \Phi\left(K+\sigma_{n}^{2} I\right)^{-1} \Phi^{\top} \Sigma_{p} \phi_{*}\right),
\end{aligned} , K = \Phi^{\top} \Sigma_{p} \Phi
$$

inspired by inner dot product, we define the kernel function as:
$$
k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\phi(\mathbf{x})^{\top} \Sigma_{p} \phi\left(\mathbf{x}^{\prime}\right)
$$

**Function View**

> how to express a function as a gaussian process

* Objective: learning mean and kernel:

$$
f(\mathbf{x}) \sim \mathcal{G} \mathcal{P}\left(m(\mathbf{x}), k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)
$$

kernel: the distribution over functions

$$
\left[\begin{array}{l}
\mathbf{f} \\
\mathbf{f}_{*}
\end{array}\right] \sim \mathcal{N}\left(\mathbf{0},\left[\begin{array}{ll}
K(X, X) & K\left(X, X_{*}\right) \\
K\left(X_{*}, X\right) & K\left(X_{*}, X_{*}\right)
\end{array}\right]\right) . \\
\begin{aligned}
\mathbf{f}_{*} \mid X_{*}, X, \mathbf{f} \sim \mathcal{N}(& K\left(X_{*}, X\right) K(X, X)^{-1} \mathbf{f} \\
&\left.K\left(X_{*}, X_{*}\right)-K\left(X_{*}, X\right) K(X, X)^{-1} K\left(X, X_{*}\right)\right) .
\end{aligned}
$$

* update mean and covariance

input: observed data point (x, y)
output: new function distribution 

* deal with noisy input:

$$
\operatorname{cov}(\mathbf{y})=K(X, X)+\sigma_{n}^{2} I
$$

* relationship with weight-space view:

The kernel function can be expanded into many base functions, which are  functions in the weight-space. 

* pipelines

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4v86udgt3j20nf07v75d.jpg)




Q: What are the hyper-parameters here?

A: 
parameters of kernel function: what is the base functions (or prior function set) looks like?
parameters of mean function: what is the mean function looks like?

Please note the the bayesian rules only used to calculate the posterior distribution of the query data. The hyper-parameters are optimized by maximizing the marginal likelihood.

Q: How to tune the hyper-parameters?

A: 
given initial value, optimize the loss to maximize the expected loss.

Q: How to evaluate results?

A: 
1. MSE -> 0
2. SMSE -> 0 (MSE/variance)
3. standardized log loss (SLL) $-\log p\left(y_{*} \mid \mathcal{D}, \mathbf{x}_{*}\right)=\frac{1}{2} \log \left(2 \pi \sigma_{*}^{2}\right)+\frac{\left(y_{*}-\bar{f}\left(\mathbf{x}_{*}\right)\right)^{2}}{2 \sigma_{*}^{2}}$ -> -inf 

Q: What is the pipeline to use it?

A: 
1. optimize kernel hyper-parameters by maximizing the marginal likelihood
2. esitimate the posterior distribution

Q: How to understand Gaussian Process Regression in equivalent kernel? (Why GPR can smooth out the functions?)

A: [Not sure]
Basic idea: the predicted value is the linear combination of the kernel functions. The update process is also updating the kernel functions.
**Matrix view**
Rewrite the predicted value function:
$$
\overline{\mathbf{f}}=K\left(K+\sigma_{n}^{2} I\right)^{-1} \mathbf{y} = \sum_{i=1}^{n} \frac{\gamma_{i} \lambda_{i}}{\lambda_{i}+\sigma_{n}^{2}} \mathbf{u}_{i} = \mathbf{h}\left(\mathbf{x}_{*}\right)^{\top} \mathbf{y}, \mathbf{h}\left(\mathbf{x}_{*}\right)=\left(K+\sigma_{n}^{2} I\right)^{-1} \mathbf{k}\left(\mathbf{x}_{*}\right)
$$
**Kernel view**
where u is the eigenvectors of K, gamma is the decomposition factor of the dataset y. h(x) is the weighted function, more data, more oscillatory. (like posterior kernel)


Q: How to deal with not zero mean function?

A: 
Trival Solution: using fixed mean function:
$$
f(\mathbf{x}) \sim \mathcal{G P}\left(m(\mathbf{x}), k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right) \\ 
\overline{\mathbf{f}}_{*}=\mathbf{m}\left(X_{*}\right)+K\left(X_{*}, X\right) K_{y}^{-1}(\mathbf{y}-\mathbf{m}(X))
$$
Solution: specify mean function bases:
$$
g(\mathbf{x})=f(\mathbf{x})+\mathbf{h}(\mathbf{x})^{\top} \boldsymbol{\beta}, \text { where } f(\mathbf{x}) \sim \mathcal{G P}\left(0, k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right) \\
g(\mathbf{x}) \sim \mathcal{G} \mathcal{P}\left(\mathbf{h}(\mathbf{x})^{\top} \mathbf{b}, k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)+\mathbf{h}(\mathbf{x})^{\top} B \mathbf{h}\left(\mathbf{x}^{\prime}\right)\right) \\
\begin{aligned}
\overline{\mathbf{g}}\left(X_{*}\right) &=H_{*}^{\top} \overline{\boldsymbol{\beta}}+K_{*}^{\top} K_{y}^{-1}\left(\mathbf{y}-H^{\top} \overline{\boldsymbol{\beta}}\right)=\overline{\mathbf{f}}\left(X_{*}\right)+R^{\top} \overline{\boldsymbol{\beta}} \\
\operatorname{cov}\left(\mathbf{g}_{*}\right) &=\operatorname{cov}\left(\mathbf{f}_{*}\right)+R^{\top}\left(B^{-1}+H K_{y}^{-1} H^{\top}\right)^{-1} R
\end{aligned} \\
\begin{aligned}
\log p(\mathbf{y} \mid X, \mathbf{b}, B)=&-\frac{1}{2}\left(H^{\top} \mathbf{b}-\mathbf{y}\right)^{\top}\left(K_{y}+H^{\top} B H\right)^{-1}\left(H^{\top} \mathbf{b}-\mathbf{y}\right) \\
&-\frac{1}{2} \log \left|K_{y}+H^{\top} B H\right|-\frac{n}{2} \log 2 \pi
\end{aligned}
$$
where beta is the combination coefficient, which can be represented by a Gaussian distribution.
