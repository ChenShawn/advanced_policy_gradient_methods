# advanced_policy_gradient_methods

## How this code is different from the openai baselines

- Many features used in the implementation has been included in [TensorFlow](https://github.com/tensorflow/tensorflow), though, the openai baselines choose to re-implement them in [their common libraries](https://github.com/openai/baselines/tree/master/baselines/common) (Possibly due to the high stability requirement). The following are a few examples:
  - `tf.distributions.Distribution`： When reparameterization-trick is used to randomize a network.
  - `tf.orthogonal_initializer`: Different from computer vision tasks, in most cases, the reinforcement learning deep learning models use orthogonal initialization instead of Xavier initialization. A minor difference is that, the orthogonalization in TensorFlow is implemented using QR decomposition, while OpenAI uses SVD.