# vectors
Vectors in the wild

## Our method
How do you represent a Transformer base language model as a vector? We want to derive vector representations of fine-tuned model and see if we can find a way to cluster them. We use the transformer architecture. As a matter of fact, in a Transformer Encoder we have :
- An **embedding layer** : It takes as input a sequence of tokens $x = [x_1, \ldots x_n]$ and turn it into a matrix $X \in \mathbb{R}^{n\times d_{model}}$.
- A **self-attention module** : The self-attention module usually works with multi-head attention. It is characterized by 4 matrices :
  - $W_q \in \mathbb{R}^{d_{model}\times d_{model}}$ : It is the matrix of keys. In reality, we can represent it as a matrix in $\mathbb{R}^{h\times d_{model} \times \frac{d_{model}}{h}}$ in order to make it clear that we use multihead attention where $h$ is the number of heads. ($d_k = \frac{d_{model}}{h})$
  - $W_k$ : It is the matrix of keys.
  - $W_v$ : It is the matrix of values
  - $W_o$ : It is the output matrix in multi-head attention. For each head we have $softmax \left ( \frac{XW_q (XW_k)^T}{\sqrt{d_k}} \right )W_v \in \mathbb{R}^{n\times d_k}$. We concatenate each of this output (rows-wise, to obtain a matrix in $\mathbb{R}^{n \times d_{model}}$) and we multiply the result by $W_o$.
- A **feedforward network** : It is characterized by 2 matrices :
  - $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ where either $d_{ff} = 4d_{model}$ or $d_{ff} = \frac{8}{3}d_{model}$ in practice.
  - $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
  - We do not consider the biases (They are not always used in practice either).

An Encoder-based model have $L$ layers. We can derive a vector representation by considering a *per-layer representation* and average it on the number of layers. The *per-layer representation* is done by computing a low-rank decomposition (can be done with singular value decomposition) of each of the matrix mentionned with a chosen rank `r`. A matrix $W$ of size $n\times m$ is then converted into a $n \times r$ matrix, which is equivalement to a $nr\times 1$ vector.
