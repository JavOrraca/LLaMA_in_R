# LLaMA in R
This repo was set up to learn how to interact with the LLaMA implementation in R following a walk-through from Posit AI blog post [_LLaMA in R with Keras and TensorFlow_](https://blogs.rstudio.com/ai/posts/2023-05-25-llama-tensorflow-keras/), by Tomasz Kalinowski.

# Setup
- Ensure your machine has access to recently released versions of R and Python
- This tutorial makes use of R's native pipe operator `|>` introduced in R 4.1.0
- This project makes use of R's `renv` framework for project-package dependency management

# R Session Info
```{r}
R version 4.3.2 (2023-10-31)
Platform: aarch64-apple-darwin20 (64-bit)
Running under: macOS Sonoma 14.2.1

Matrix products: default
BLAS:   /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib 
LAPACK: /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.11.0

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

time zone: America/Los_Angeles
tzcode source: internal

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
[1] keras_2.13.0      tfautograph_0.3.2 tensorflow_2.14.0 envir_0.2.2      
[5] purrr_1.0.2      

loaded via a namespace (and not attached):
 [1] backports_1.4.1   R6_2.5.1          base64enc_0.1-3   Matrix_1.6-5     
 [5] lattice_0.22-5    reticulate_1.34.0 magrittr_2.0.3    remotes_2.4.2.1  
 [9] generics_0.1.3    png_0.1-8         lifecycle_1.0.4   cli_3.6.2        
[13] grid_4.3.2        vctrs_0.6.5       zeallot_0.1.0     tfruns_1.5.1     
[17] compiler_4.3.2    rstudioapi_0.15.0 tools_4.3.2       whisker_0.4.1    
[21] Rcpp_1.0.12       rlang_1.1.3       jsonlite_1.8.8   
```
