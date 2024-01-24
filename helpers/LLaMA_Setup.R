# Download LLaMA + Weights ------------------------------------------------

library(reticulate)
use_virtualenv("./.venv")

# At this point, head to https://ai.meta.com/llama/#download-the-model to
# accept the Facebook license, download LLaMA 2, and the associated weights.
# You'll receive an email with a custom link to download the model. If you're 
# following this script, set that custom URL to the env var UNIQUE_CUSTOM_URL.
system("git clone https://github.com/facebookresearch/llama.git")

# Install LLaMA dependencies wget and md5sum
# On macOS, I'm using homebrew to install these deps
system("brew install wget")
system("brew install md5sha1sum")

# Build the command and run install script:
url <- Sys.getenv("UNIQUE_CUSTOM_URL")
command <- sprintf("sh llama/download.sh | sleep 3 | echo '%s' | sleep 3 | echo '7B'", url)
system(command)
