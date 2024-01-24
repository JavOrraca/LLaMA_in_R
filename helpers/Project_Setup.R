# Environment Setup -------------------------------------------------------

### Restore libraries in renv.lock
renv::restore()
# remotes::install_github("rstudio/tensorflow")
# install.packages(c("reticulate", "keras"))
logger::log_info("Packages found in renv lockfile ('~/renv.lock') restored")

### If you want to install the latest Python 3.10 release,
### uncomment lines 10-11 and comment out lines 13-15
# reticulate::install_python("3.10:latest")
# reticulate::virtualenv_create("./.venv", version = "3.10:latest")
reticulate::virtualenv_create(envname = "./.venv", 
                              version = "3.10.13", 
                              force = TRUE)
logger::log_info("Python virtual env created in '~/.venv/'")

### Install TensorFlow v2.15 and Keras for the same TF build
tensorflow::install_tensorflow(
  envname = "./.venv", 
  version = "2.15")
logger::log_info("TensorFlow 2.15 was installed in '~/.venv/'")

keras::install_keras(envname = "./.venv", version = "2.15")
logger::log_info("Keras installed for the TensorFlow 2.15 build")

### Installing TensorFlow-Text on Apple silicon chips (M1/M2/M3):
## Medium article: https://solomonmg.medium.com/so-you-want-to-install-tensorflow-text-in-python-on-that-new-m1-m2-laptop-5e7d37be591e
## GitHub for tf-text Python wheel: https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases
## Note: Download and install from filepath the tensorflow-text Python wheel given your version of Python and TensorFlow
reticulate::py_install("/Users/javierorraca/Downloads/tensorflow_text-2.15.0-cp310-cp310-macosx_11_0_arm64.whl", 
                       envname = "./.venv",
                       pip = TRUE)
