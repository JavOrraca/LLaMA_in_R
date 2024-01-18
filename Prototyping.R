# 0. Environment Setup ----------------------------------------------------

### Install dev package versions
remotes::install_github("rstudio/tensorflow")
install.packages(c("reticulate", "keras"))

# reticulate::install_python("3.10:latest")
# reticulate::virtualenv_create("./.venv", version = "3.10:latest")
reticulate::virtualenv_create(envname = "./.venv", 
                              version = "3.10.13", 
                              force = TRUE)

#### Install TensorFlow v2.15 and Keras
tensorflow::install_tensorflow(envname = "./.venv", 
                               version = "2.15", 
                               cuda = FALSE)

keras::install_keras(envname = "./.venv", 
                     version = "2.15")


# 1. TensorFlow Setup -----------------------------------------------------

library(purrr)
library(envir)
library(tensorflow)
library(tfautograph)
library(keras)

use_virtualenv("./.venv")
options(tensorflow.extract.warn_tensors_passed_asis = FALSE)

attach_eval({
  import_from(glue, glue)
  import_from(jsonlite, read_json)
  import_from(withr, with_dir, with_options)
  import_from(keras$layers, Dense)
  np <- reticulate::import("numpy", convert = FALSE)
  
  seq_len0 <- function(x) seq.int(from = 0L, length.out = x)
})
