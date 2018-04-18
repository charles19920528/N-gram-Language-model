# N-gram-Language-model

## 1. Language model
The language model we implemented is a linear interpolation of the uni, bi and tri-gram models with additive smoothing. For this assignment, we didn't use perplexity score and cross-validation to tune the hyper parameters. The hyper parameters (weights for each model) are set to be 0.3 for unigram, 0.5 for bigram and 0.2 for tri-gram models. The hyper parameters of the additive smoothing method (perturbed mass for each character) are set to be 0.0001 for all characters and all models.
## 2. Training data
With limited amount of time and coding skills, we didn't found massive amount of data for this assignment. Instead we chose five Chinese novels as our training data. Names of those novels are list in the appendix. One reason we focus on Chinese is that we are working on the character level and we are only use a tri-gram model. For English, we imagine that the model which only stores history of two previous characters is not going to perform well.
## 3. Tools
We use python as the programming language. The external library we used is NumPy. 
## 4. Appendix (Names of novels)
Demi-Gods and Semi-Devils
The Legend of the Condor Heroes
The Return of the Condor Heroes
The Heaven Sword and Dragon Saber
The Three-Body Problem
