# - Code from RStudio blog was slightly modified to process LLaMA-2-7B
# - Some of this code and copy was taken verbatim from the blog post above
# - Hats off to Tomasz Kalinowski for writing such an in-depth overview 🙏
# - Attribution:
#   - Kalinowski (2023, May 25). Posit AI Blog: LLaMA in R with Keras and TensorFlow. 
#     Retrieved from https://blogs.rstudio.com/tensorflow/posts/2023-05-25-llama-tensorflow-keras/


# Environment & LLaMA-2-7B Setup ------------------------------------------

# source("helpers/Project_Setup.R")
# source("helpers/LLaMA_Setup.R")


# Load Packages & Weights -------------------------------------------------

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

# Convert pre-trained torch weights from the checkpoint format to something 
# that’s more framework agnostic:
# reticulate::py_install("torch", pip = TRUE)
torch <- reticulate::import("torch", convert = FALSE)

with_dir(here::here("llama-2-7b"), {
  pretrained_weights <- torch$load("consolidated.00.pth")
  for (name in names(pretrained_weights)) {
    filename <- sprintf("%s.npy", name)
    array <- pretrained_weights[[name]]$to(dtype = torch$float32)$numpy()
    np$save(filename, array)
    message(glue(
      "wrote: '{basename(filename)}' with shape: {array$shape}"))
  }
})

weights_path <- function(filename) normalizePath(
  file.path("llama-2-7b/", glue::glue(filename, .envir = parent.frame())), 
  mustWork = TRUE
)

params <- read_json("llama-2-7b/params.json")
str(params)


# Tokenizer ---------------------------------------------------------------

tf_text <- reticulate::import("tensorflow_text")
tokenizer_path <- here::here("tokenizer.model")
tokenizer <- tf_text$SentencepieceTokenizer(
  tf$io$gfile$GFile(tokenizer_path, "rb")$read(),
  add_bos = TRUE, add_eos = FALSE,
)

prompt <- "The best way to attract bees"
tokenizer$tokenize(prompt)

prompt |> tokenizer$tokenize() |> tokenizer$detokenize()

show_tokens <- function(what) {
  if(is.character(what))
    token_ids <- what |> tokenizer$tokenize() |> as.integer()
  else
    token_ids <- as.integer(what)
  tokens <- token_ids |>
    map_chr(function(id) {
      id |>
        as_tensor(shape = c(1)) |>
        tokenizer$detokenize() |>
        as.character()
    })
  
  names(tokens) <- token_ids
  tokens
}

show_tokens(prompt)

# Test some additional use cases
show_tokens("ing")
show_tokens("working")
show_tokens("flexing")
show_tokens("wonking")

# How many tokenizers are there?
as.integer(tokenizer$vocab_size())

# Frequently encountered tokens are assigned lower IDs
show_tokens(seq(50, len = 10))
show_tokens(seq(100, len = 10))
show_tokens(seq(1000, len = 10))
show_tokens(seq(10000, len = 10))
show_tokens(seq(20000, len = 10))
show_tokens(seq(to = as.integer(tokenizer$vocab_size()) - 1, len = 10))


# Embeddings --------------------------------------------------------------

tok_embeddings <- keras$layers$Embedding(
  input_dim = tokenizer$vocab_size(),
  output_dim = params$dim,
  embeddings_initializer =
    \(...) np$load(weights_path("tok_embeddings.weight.npy"))
)

tok_embeddings(3L) |> str()

prompt |> # "The best way to attract bees"
  tokenizer$tokenize() |>
  tok_embeddings() |>
  str()


# TransformerBlock --------------------------------------------------------

weights_path("params.json")  |> read_json() |> _$n_layers

# A TransformerBlock is basically an Attention layer followed by a few (fancy) 
# dense layers, with some simple composition patterns (tricks) that help with 
# training. Attention is the heart of the model: it’s the most interesting, and 
# also the most involved.
# 
# TransformerBlock class has two methods: 
#   1. initialize: called when we first create the block 
#       - In initialize, we create 4 layers:
#         * an Attention layer 
#         * a FeedForward layer
#         * two (2) RMSNorm layers
#   2. call: called when we run the forward pass of the block
#       - The call method has a few simple ideas - In no particular order:
#         * the first one to observe is the composition pattern of adding residuals
#         * the next composition pattern to note is the repeating usage of a normalization layer
#       - They can all be thought of as a stabilizer that helps with training
#       - Like their deep-learning cousins the regularizers, their main function 
#         is to keep values passing through in a sensible range–in the ball park 
#         of (-1, 1), typically

TransformerBlock(keras$layers$Layer) %py_class% {
  initialize <- function(attn_head_size, attn_n_heads,
                         norm_eps = k_epsilon(), ...,
                         block_id = NULL) {
    super$initialize(...)
    
    self$attention <- Attention(attn_head_size, attn_n_heads,
                                block_id = block_id)
    
    self$feed_forward <- FeedForward(
      hidden_dim = 4 * attn_head_size * attn_n_heads,
      block_id = block_id)
    
    self$attention_norm <- RMSNorm(eps = norm_eps,
                                   block_id = block_id,
                                   feeds_into = "attention")
    self$feed_forward_norm <- RMSNorm(eps = norm_eps,
                                      block_id = block_id,
                                      feeds_into = "ffn")
  }
  
  call <- function(x) {
    
    # norm and attention
    x2 <- x |>
      self$attention_norm() |>
      self$attention()
    
    x <- x + x2 # add residual
    
    # norm and swiglu
    x2 <- x %>%
      self$feed_forward_norm() %>%
      self$feed_forward()
    
    x <- x + x2 # residual again
    
    x
  }
}

# With the framing in place, let’s go through and take a closer look at RMSNorm, FeedForward, 
# and then with the foundation in place, we’ll turn our attention to Attention.


# RMSNorm -----------------------------------------------------------------

# RMSnorm() has a single trainable tensor w
#  - In the forward pass, each value in the input is multiplied by the reciprocal-
#    root-mean-square of all the values in the feature axis and by w
#  - Certainly a mouthful, but just a simple sequence of arithmetic transformations 
#    in the end, designed for the express purpose of adjusting the range of values 
#    passing through.

RMSNorm(keras$layers$Layer) %py_class% {
  initialize <-
    function(eps = 1e-6, ..., block_id = NULL, feeds_into = NULL) {
      super$initialize(...)
      self$eps <- eps
      self$block_id <- block_id
      self$feeds_into <- feeds_into
    }
  
  build <- function(input_shape) {
    # input_shape == (batch_size, seqlen, params$dim)
    # self$w will broadcast over batch_size and seqlen dims.
    # w_shape == (1, 1, params$dim)
    w_shape <- rep(1L, length(input_shape))
    w_shape[length(input_shape)] <- as.integer(input_shape) |> tail(1L)
    
    # define a local function that will load
    # the pretrained-weights if we supplied `block_id` and `feeds_into`
    import_from({self}, block_id, feeds_into)
    initializer <-if (is.null(block_id))
      "ones"
    else if (block_id >=0) {
      \(...) weights_path("layers.{block_id}.{feeds_into}_norm.weight.npy") |>
        np$load() |> np$expand_dims(0:1)
    } else if(block_id == -1)
      # load weights for the final output normalization layer, which is not
      # part of a TransformerBlock
      \(...) weights_path("norm.weight.npy") |>
      np$load() |> np$expand_dims(0:1)
    
    self$w <- self$add_weight(shape = w_shape,
                              initializer = initializer,
                              trainable = TRUE)
  }
  
  rrms <- function(x) {
    # reciprocal root mean square along the last axis
    x %>% # (batch_size, seqlen, n_features)
      tf$math$square() %>%
      tf$reduce_mean(axis = -1L, keepdims = TRUE) %>% # (batch_size, seqlen, 1)
      tf$math$add(self$eps) %>% # for numerical stability
      tf$math$rsqrt()
  }
  
  call <- function(x) {
    x * self$rrms(x) * self$w
  }
}

# RMSNorm experimentation
norm <- RMSNorm()
m <- matrix(c(0, 1,
              2, 3), nrow = 2)
norm(m)
norm(m*10)
norm(m*100)


# FeedForward -------------------------------------------------------------

# FeedForward consists of three Dense layers: "initialize" (1) does some simple 
# arithmetic, munging on the input value "hidden_dim" (2) to ensure the size is 
# a performant multiple of 256, and "build" (3) is mostly boiler plate for 
# creating the layers and loading the weights.

# The novelty of FeedForward() is in the call() method, where rather than composing 
# the Dense layers in a conventional sequential model with, say, ReLU activations 
# in between and maybe some dropout, the layers are composed to form a “SwiGLU” unit
# SwiGLU Source: https://arxiv.org/abs/2002.05202

# The Feedforward$call() is just a single SwiGLU followed by a linear projection. In 
# its essence, it’s a clever composition of three (learned) linear projections, an 
# element-wise multiplication, and a silu() activation function.

# Perhaps the most surprising observation to make here is the relative dearth of 
# activation functions, or even non-linearities, not just in FeedForward, but 
# overall. The silu() in this feedforward, the reciprocal-root-mean-square in 
# RMSnorm(), and a softmax() in Attention() are the only non-linear transformations 
# in the whole sequence of TransformerBlocks. Everything else is a linear transformation!

FeedForward(keras$layers$Layer) %py_class% {
  
  initialize <- function(hidden_dim, multiple_of = 256L,
                         ..., block_id = NULL) {
    super$initialize()
    
    if(!is.null(multiple_of)) {
      hidden_dim <- hidden_dim %>%
        { as.integer( . * (2/3)) } %>%
        { (. + multiple_of - 1) %/% multiple_of } %>%
        { . * multiple_of }
    }
    
    self$hidden_dim <- hidden_dim
    self$block_id <- block_id
  }
  
  build <- function(input_shape) {
    output_dim <- input_shape |> as.integer() |> tail(1)
    
    if(is.null(self$block_id))
      load_weight <- \(...) NULL
    else
      load_weight <- \(name) \(...) np$load(weights_path(
        "layers.{self$block_id}.feed_forward.{name}.weight.npy"))$`T`
    
    self$w1 <- Dense(self$hidden_dim, use_bias = FALSE,
                     kernel_initializer = load_weight("w1"))
    self$w2 <- Dense(output_dim, use_bias = FALSE,
                     kernel_initializer = load_weight("w2"))
    self$w3 <- Dense(self$hidden_dim, use_bias = FALSE,
                     kernel_initializer = load_weight("w3"))
    
    super$build(input_shape)
  }
  
  call <- function(x) {
    import_from({self}, w1, w2, w3)
    import_from(tf$nn, silu)
    
    x %>%
      { silu(w1(.)) * w3(.) } %>% # SwiGLU
      w2()
  }
}


# Attention ---------------------------------------------------------------

# Attention in LLaMA is similar but not identical to the Attention described in 
# the original Transformers paper (and available as a keras builtin under 
# keras$layers$MultiHeadAttention()). The core novelty is the addition of the 
# apply_rotary_embedding() function, which we’ll describe shortly. The additional 
# novelty is balanced by the simplicity from the fact that the layer is performing 
# self-attention—we don’t need to pass in different query, key, and value tensors 
# (or reason about what that means), since the same input serves all three roles.

Attention(keras$layers$Layer) %py_class% {
  initialize <- function(head_size, n_heads,
                         ..., block_id = NULL) {
    super$initialize(...)
    
    self$head_size <- head_size
    self$n_heads <- n_heads
    
    if (is.null(block_id))
      load_weight <- function(name) NULL
    else
      load_weight <- \(name) \(...) np$load(weights_path(
        "layers.{block_id}.attention.{name}.weight.npy"))$`T`
    
    Dense <- function(name) keras$layers$Dense(
      units = n_heads * head_size,
      use_bias = FALSE,
      kernel_initializer = load_weight(name)
    )
    
    self$wq <- Dense("wq")
    self$wk <- Dense("wk")
    self$wv <- Dense("wv")
    self$wo <- Dense("wo")
  }
  
  call <- function(x) {
    c(batch_size, seqlen, n_features) %<-% tf$unstack(tf$shape(x))
    
    # 1. project (linear transform) x into
    #    query, key, and value tensors
    # 2. reshape q k v, splitting out the last dim (n_features)
    #    into n_heads independent subspaces,
    #    each with size head_size.
    #    (n_features == head_size * n_heads)
    split_heads_shape <- c(batch_size, seqlen,
                           self$n_heads, self$head_size)
    q <- x |> self$wq() |> tf$reshape(split_heads_shape)
    k <- x |> self$wk() |> tf$reshape(split_heads_shape)
    v <- x |> self$wv() |> tf$reshape(split_heads_shape)
    
    # embed positional information in query and key
    # (bsz, seqlen, n_heads, head_size)
    q %<>% apply_rotary_embedding()
    k %<>% apply_rotary_embedding()
    
    # reshape:
    #   move heads out of the last 2 axes,
    #   so later matmuls are performed across the subspaces (heads)
    #   between (seqlen, head_size) axes
    v <- tf$transpose(v, c(0L, 2L, 1L, 3L)) # (bsz, n_heads, seqlen, head_size)
    q <- tf$transpose(q, c(0L, 2L, 1L, 3L)) # (bsz, n_heads, seqlen, head_size)
    k <- tf$transpose(k, c(0L, 2L, 3L, 1L)) # (bsz, n_heads, head_size, seqlen)
    
    # calculate and normalize attention scores
    scores <- q %*% k                       # (bsz, n_heads, seqlen, seqlen)
    scores <- scores / sqrt(self$head_size) # scale
    
    # apply causal mask, so the model can't "look ahead" during training
    mask <- make_mask(seqlen, dtype = scores$dtype)
    scores %<>% { . + mask }
    
    scores <- tf$nn$softmax(scores, axis = -1L)
    
    # adjust values tensor with attention scores
    # scores (bsz, n_heads, seqlen, seqlen)
    # v      (bsz, n_heads, seqlen, head_size)
    output <- scores %*% v   # (bsz, n_heads, seqlen, head_size)
    
    # combine heads back into a single features dim,
    # so Attention output_shape==input_shape
    output <- output |>
      tf$transpose(c(0L, 2L, 1L, 3L)) |> # (bsz, seqlen, n_heads, head_size)
      tf$reshape(tf$shape(x))            # (bsz, seqlen, n_heads * head_size)
    
    # one more trainable linear projection for good luck
    output <- self$wo(output) # (bsz, seqlen, n_heads * head_size)
    
    output
  }
}

# To develop an understanding of the mechanics in a layer like this, it’s helpful 
# to temporarily unsee some of the minutia that can act as a fog obscuring the 
# essence of the operation. In this instance, if we temporarily strip out the 
# transpose()s and reshape()s (as clever and vital as they are), this is what’s 
# left: [DON'T RUN CODE BELOW, ONLY FOR EXAMPLE]
# 
# call <- function(x) {
#   # split input into three learned linear projections
#   q <- x |> self$wq()
#   k <- x |> self$wk()
#   v <- x |> self$wv()
#   
#   # rotate q,k to inject position information.
#   # cross q,k to calculate an attention score for each token pair.
#   scores <- rotate(q) %*% rotate(k)   |>  normalize_scores()
#   
#   # adjust the 3rd projection with the attention scores
#   output <- scores %*% v
#   
#   self$wo(output) # one more learned linear projection for good luck
# }
# 
# Returning to the transpose()s and reshapes(), you can observe that their 
# purpose is to make it so that the attention calculations are performed across 
# n_heads independent subspaces, rather than in a single larger space. The same 
# reasoning drives this decision as that driving usage of depthwise-separable 
# convolutions in image models. Empirically, for the fixed compute budget, 
# factoring features into independent subspaces performs better than doing the 
# same core operations in single larger feature space. As with all things, there 
# is a balance to strike between n_heads (the number of subspaces) and head_dim 
# (the size of each subspace). The LLaMA authors have struck the balance like this 
# at the various model sizes:
p <- read_json(weights_path("params.json"))

with(p, list(llama_size = "7B",
             n_heads = n_heads,
             head_dim = dim %/% n_heads)) |> 
  dplyr::as_tibble()

# Lets turn our attention to the causal attention mask:
make_mask <- function(seqlen, dtype = k_floatx()) {
  x <- tf$range(seqlen)
  mask <- tf$where(x[, tf$newaxis] < x[tf$newaxis, ],
                   tf$constant(-Inf, dtype = dtype),
                   tf$constant(0, dtype = dtype))
  
  # broadcast over batch and heads dim
  mask[tf$newaxis, tf$newaxis, , ] # (1, 1, seqlen, seqlen)
}

make_mask(seqlen = 5L)
make_mask(seqlen = 10L)


# Rotary Position Embedding -----------------------------------------------

# Lets turn our attention to apply_rotary_embedding(): This core innovation was 
# published by Su et al. (2022) in the paper titled “RoFormer: Enhanced Transformer 
# with Rotary Position Embedding” (Source: https://arxiv.org/abs/2104.09864)
# 
# Some more context:
# 
#  1. The bare Attention() mechanism doesn’t leave any possibility for a token’s
#     position in a sequence to affect the attention scores, since only token-pairs 
#     are scored. Attention treats its input like a bag-of-tokens.
# 
#  2. The position of a token in a sequence is clearly important, and the attention 
#     layer should have access to that information.
# 
#  3. The absolute position of a token in a sequence is less important than the 
#     relative position between tokens. (Especially so for long sequences).
# 
# If we imagine the features as complex numbers, we can rotate them, and we can 
# calculate angles between them. From the Roformers paper:
#  - "Specifically, incorporating the relative position embedding is straightforward: 
#     simply rotate the affine-transformed word embedding vector by amount of angle 
#     multiples of its position index and thus interprets the intuition behind 
#     Rotary Position Embedding"
# 
# Expanding slightly: 
#  - The rotation matrix is designed so that subsequently, after rotating our q 
#    and k token sequence embedding the same way, the angle between token features 
#    is a function of the relative distance between those tokens in the token sequence
#  - The relative angle between two tokens is invariant to the absolute position of 
#    those tokens in the full sequence.
#  - In short, the rotation injects positional information
#     - The meaning or interpretability of that positional information, or how it is 
#       meant to be used, or even extracted from the result of q %*% k, is left to 
#       the model to learn
# 
# Here is some code:
apply_rotary_embedding <- function(x) {
  c(., seqlen, ., head_size) %<-%
    tf$unstack(tf$shape(x))
  
  rotation_matrix <- compute_rotation_matrix(seqlen, head_size)
  
  x %>%
    view_as_complex() %>%
    { . * rotation_matrix } %>%
    view_as_real()
  
}

compute_rotation_matrix <-
  function(seqlen, feature_dim, theta = 10000) {
    # `feature_dim` here is going to be attention$head_size
    # `seqlen` is going to match the token sequence length.
    
    t <- tf$range(seqlen, dtype = tf$float32)
    freqs <- tf$range(start = 0, limit = 1, delta = 1 / (feature_dim %/% 2),
                      dtype = tf$float32)
    tf_assert(tf$size(freqs) == feature_dim %/% 2)
    freqs <- 1.0 / (theta ^ freqs)
    
    # outer product; (seqlen, head_size/2)
    freqs <- tf$einsum('a,b->ab', t, freqs)
    
    rot_mat <- tf$complex(tf$cos(freqs), tf$sin(freqs))
    
    # the positional embedding will be broadcast across batch and heads dim
    rot_mat[tf$newaxis, , tf$newaxis, ] #(1, seqlen, 1, headdim/2)
  }

view_as_complex <- function(x) {
  tf$complex(x[all_dims(), `::2`],
             x[all_dims(), `2::2`])
}

view_as_real <- function(x) {
  # xs = (..., f);  xs2 = (..., f*2)
  xs <- tf$shape(x)
  xs2 <- tf$concat(list(xs[1:(length(xs)-1)],
                        xs[length(xs), drop = FALSE] * 2L),
                   axis = 0L)
  
  x2 <- tf$stack(list(Re(x), Im(x)), axis = -1L)
  
  # (..., f, 2) -> (..., f*2)
  tf$reshape(x2, xs2)
}

# We can quickly confirm that the rotary embeddings only rotate 
# features and don’t scale them:
near <- function (x, y, tol = 1e-6) abs(x - y) < tol
all(near(1, Mod(compute_rotation_matrix(2048L, 128L))))

# Because of some of the mathematical properties of the rotation matrix, it’s 
# possible to avoid doing a full complex multiply operation and still arrive at 
# the same result. Also, since the rotation matrix never changes, it makes sense 
# to only compute it once and cache it, like so:
precomputed_rotation_matrix <- compute_rotation_matrix(
  seqlen = 2048L, # LLaMA max seqlen
  feature_dim = with(params, dim %/% n_heads)  # head_size
)

apply_rotary_embedding_faster <- function(x) {
  
  rotate_every_two <- function(x) {
    x1 <- x[all_dims(), `::2`]
    x2 <- x[all_dims(), `2::2`]
    x_ <- tf$stack(list(-x2, x1), axis = -1L)
    tf$reshape(x_, tf$shape(x))
  }
  
  repeat_each_twice <- function(x) {
    tf$`repeat`(x, 2L, axis = -1L)
  }
  
  seqlen <- tf$shape(x)[2]
  rot <- precomputed_rotation_matrix[, NA:seqlen, , ]
  
  cos <- Re(rot) |> repeat_each_twice()
  sin <- Im(rot) |> repeat_each_twice()
  
  (x * cos) + (rotate_every_two(x) * sin)
}

rand <- tf$random$uniform(shape(3, 8, params$n_heads, 128))
all(apply_rotary_embedding(rand) == apply_rotary_embedding_faster(rand))

apply_rotary_embedding <- apply_rotary_embedding_faster

# Finally, note that the rotary positional embeddings are applied within each 
# Attention layer. This is different from the original Transformer implementation, 
# where a positional embedding was only added once at the head of the model. 
# Similar to residual connections, you can think of the presence of these repeated 
# injections of positional information as relieving the remaining trainable layers 
# from the burden of allocating some of their weights to the task of “passing 
# through” or “preserving” the positional information for later layers.


# Tying It All Together (Keras) -------------------------------------------

# With Tokenizer, Embedding, TransformerBlock (RMSNorm, Attention FeedForward and 
# apply_rotary_embedding) all covered, it’s time to tie all the pieces together 
# into a Transformer model. We could do this using %py_class% like with the other 
# layers above, but it’s just as easy to move over to using the Keras functional 
# API at this point.

layer_transformer_block <- create_layer_wrapper(TransformerBlock)
layer_rms_norm <- create_layer_wrapper(RMSNorm)

# input to the model will be output from the tokenizer
input <- layer_input(shape(NA)) #, dtype = "int32")

x <- input |>
  tok_embeddings()  # instantiated earlier in the blog-post

for(block_id in seq_len0(params$n_layers)) {
  x <- x |>
    layer_transformer_block(attn_head_size = params$dim %/% params$n_heads,
                            attn_n_heads = params$n_heads,
                            norm_eps = params$norm_eps,
                            block_id = block_id)
}

# final output projection into logits of output tokens
x <- x |>
  layer_rms_norm(block_id = -1, eps = params$norm_eps) |>
  layer_dense(
    tokenizer$vocab_size(), use_bias = FALSE,
    kernel_initializer = \(...) np$load(weights_path("output.weight.npy"))$`T`
  )

# slice out the logits for the last token
with_options(c(tensorflow.extract.warn_negatives_pythonic = FALSE), {
  output <- x[, -1, ]
})

llama <- keras_model(input, output) %>%
  compile(jit_compile = FALSE) # SET TO TRUE IF NOT USING APPLE SILICON?

# The input to the model is tokenized text and the output is the (unnormalized) 
# probabilities for each token in tokenizer$vocab_size() being the next token in 
# the sequence.

next_token_probs <- prompt %>%
  tokenizer$tokenize() %>%
  llama()

next_token_probs

# Sampling strategies for selecting a token from the token logits is a rich topic, 
# but for now, let’s just take the argmax().

sampler <- \(logits) tf$argmax(logits, axis = -1L, output_type = "int32")

(next_token <- sampler(next_token_probs))

tokenizer$detokenize(next_token) |> as.character()

# Let’s run it for a few tokens and let LLaMa finish the sentence:
prompt_tokens <- tokenizer$tokenize("The best way to attract bees")

for (i in 1:20) {
  
  next_token_probs <- prompt_tokens |> llama()
  next_token <- sampler(next_token_probs)
  
  prompt_tokens %<>% { tf$concat(c(., next_token), axis = -1L) }
  
  # end of sentence
  if (as.logical(next_token == tokenizer$string_to_id(".")))
    break
}

prompt_tokens |>
  tokenizer$detokenize() |>
  as.character() |>
  strwrap(60) |> writeLines()
