# LLaMA in R
This repo was set up to learn how to interact with the LLaMA implementation in R following a walk-through from Posit AI blog post [_LLaMA in R with Keras and TensorFlow_](https://blogs.rstudio.com/ai/posts/2023-05-25-llama-tensorflow-keras/), by Tomasz Kalinowski.

# Recommended Installation Pattern
- Ensure your machine has recently released versions of R and Python
- Download and install several R packages including `reticulate` (to execute Python code / scripts in the backend), `keras`, and `tensorflow`
- Run `keras::install_keras()`
  - This function installs Tensorflow and all Keras dependencies
  - This is a thin wrapper around `tensorflow::install_tensorflow()`, with the only difference being that this includes by default additional extra packages that `keras` expects, and the default version of `tensorflow` installed by `keras::install_keras()` may at times be different from the default installed `tensorflow::install_tensorflow()`
  - As of this writing (`2024-01-17`), the default version of `tensorflow` installed by `keras::install_keras()` is "2.13"
