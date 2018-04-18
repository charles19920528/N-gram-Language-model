import numpy as np

###########################################################
# Compute the number of words/pairs in the training data
###########################################################

def char_frequency(string, string_char_list, beg_char):
    # Input: string_char_list: we split the string into a list of single char.
    # Output: a dictionary containing count of each unique char in the string.

    unique_char_list = list(set(string_char_list))
    freq_dict = {}
    for char in unique_char_list:
        freq_dict[char] = string.count(char)
    freq_dict[beg_char] /= 2
    return freq_dict


def bi_char_freq(string_char_list):
    # Input: string_char_list: we split the string into a list of single char.
    # Output: a dictionary containing count of each unique (char i | char j) in the string.
    #         entry (b, c) is count for (word c | word b)
    # We add a special key ("start", word i) to denote the case where the passage starts with word i.

    freq_dict_bi = {}
    for idx, char in enumerate(string_char_list):
        if idx + 1 < len(string_char_list):
            if idx == 0:
                continue
            else:
                char_pair_bi = (string_char_list[idx - 1], char)

                if char_pair_bi in freq_dict_bi:
                    freq_dict_bi[char_pair_bi] += 1
                else:
                    freq_dict_bi[char_pair_bi] = 1

        else:
            return freq_dict_bi


def tri_char_freq(string_char_list):
    freq_dict_tri = {}
    for idx, char in enumerate(string_char_list):
        if idx + 2 < len(string_char_list):
            if idx < 2:
                continue
            else:
                char_pair_tri = (string_char_list[idx - 2], string_char_list[idx - 1], char)

                if char_pair_tri in freq_dict_tri:
                    freq_dict_tri[char_pair_tri] += 1
                else:
                    freq_dict_tri[char_pair_tri] = 1

        else:
            return freq_dict_tri


################################################################
# Compute the probability of words/pairs in the training data
################################################################

def uni_prob(char_freq_dict, vocab_size, perturb_mass, beg_char):
    # Input: char_freq_dic: contains counts of each character in the passage.
    #        perturb_mass: the alpha in the smoothing model.
    # Output: a dictionary of len(char_freq_dict)storing p(word i). word i has positive count.
    # We will map every word we didn't see to the key 'unseen'.

    prob_dict = {}
    char_freq_dict_no_beg_char = char_freq_dict.copy()
    char_freq_dict_no_beg_char.pop(beg_char, None)
    total_mass = (vocab_size * perturb_mass + sum(char_freq_dict_no_beg_char.values()))
    prob_dict['unseen'] = perturb_mass / total_mass

    for key in char_freq_dict:
        prob_dict[key] = (char_freq_dict[key] + perturb_mass) / total_mass
    prob_dict.pop(beg_char, None)
    return prob_dict


# uni_probability_dict = uni_prob(char_freq_dict, unicode_upper + 1, 0.1, beg_char)
#
# # check if sum up to 1
# sum(uni_probability_dict.values()) - uni_probability_dict["unseen"] + uni_probability_dict["unseen"] * \
# (unicode_upper + 1 - len(uni_probability_dict) + 1)


def bi_prob(char_freq_dict, freq_dict_bi, vocab_size, perturb_mass, beg_char):
    # Input: freq_dict_bi: contains counts of (char j, char i) etc.
    # Output: a dictionary of len(freq_dict_bi) - 1 with key (j, i) storing p(word i | word j).

    prob_dict = {}
    total_perturb_mass = perturb_mass * vocab_size

    for key in freq_dict_bi:
        prob_dict[key] = (perturb_mass + freq_dict_bi[key]) / (total_perturb_mass + char_freq_dict[key[0]])
    prob_dict.pop((beg_char, beg_char), None)

    for key in char_freq_dict:
        prob_dict[(key, "unseen")] = perturb_mass / (total_perturb_mass + char_freq_dict[key])

    prob_dict[("unseen", "unseen")] = perturb_mass / total_perturb_mass

    return prob_dict


def tri_prob(freq_dict_bi, freq_dict_tri, vocab_size, perturb_mass, beg_char):
    prob_dict = {}
    total_perturb_mass = perturb_mass * vocab_size

    for key in freq_dict_tri:
        prob_dict[key] = (perturb_mass + freq_dict_tri[key]) / (total_perturb_mass + freq_dict_bi[key[0: 2]])

    for key in freq_dict_bi:
        new_key = (key[0], key[1], "unseen")
        prob_dict[new_key] = perturb_mass / (total_perturb_mass + freq_dict_bi[key])

    prob_dict[('unseen', 'unseen', 'unseen')] = perturb_mass / total_perturb_mass

    return prob_dict


##########################
# Generate language model
##########################

def language_model(string, string_char_list, beg_char, vocab_size, perturb_mass):
    # Input: string: the training passages which are saved in one string.
    #        string_char_list: we split the string into a list of single char.
    #        beg_char: several special character we add at the beginning of each passage
    #        vocab_size: total number of characters which doesn't include beg_char
    #        perturb_mass: a list of size 3 including smoothing parameters for uni, bi & tri-gram models.
    # Output: Dictionaries of probabilities in uni, bi & tri-gram models.
    # This is a wrap-up function for the previous functions.

    char_freq_dict = char_frequency(string, string_char_list, beg_char)
    bi_pair_freq = bi_char_freq(string_char_list)
    tri_pair_freq = tri_char_freq(string_char_list)

    perturb_mass_uni, perturb_mass_bi, perturb_mass_tri = perturb_mass
    uni_probability_dict = uni_prob(char_freq_dict, vocab_size, perturb_mass_uni, beg_char)
    bigram_probability_dict = bi_prob(char_freq_dict, bi_pair_freq, vocab_size, perturb_mass_bi, beg_char)
    trigram_probability_dict = tri_prob(bi_pair_freq, tri_pair_freq, vocab_size, perturb_mass_tri, beg_char)

    return uni_probability_dict, bigram_probability_dict, trigram_probability_dict


###########################
# Generate language
###########################

def generate_prob_bi(prob_dict_bi, history, vocab_size):
    # Input: history: a list of length 2 [char i - 2, char i - 1].
    # Output: a list of length vocab_size storing p[word i | history[1]]

    initial_key_bi = (history[1], 'unseen')

    if initial_key_bi in prob_dict_bi:
        prob_array_bi = np.full(vocab_size, prob_dict_bi[initial_key_bi])
        for unicode in np.arange(vocab_size): # we assume that we use all the unicode from 0 to vocab_size - 1
            key_bi = (history[1], chr(unicode))
            if key_bi in prob_dict_bi:
                prob_array_bi[unicode] = prob_dict_bi[key_bi]
            else:
                continue
    else:
        prob_array_bi = np.full(vocab_size, prob_dict_bi[("unseen", 'unseen')])

    return prob_array_bi


def generate_prob_tri(prob_dict_tri, history, vocab_size):

    initial_key_tri = (history[0], history[1], "unseen")

    if initial_key_tri in prob_dict_tri:
        prob_array_tri = np.full(vocab_size, prob_dict_tri[initial_key_tri])
        for unicode in np.arange(vocab_size):
            key_tri = (history[0], history[1], chr(unicode))
            if key_tri in prob_dict_tri:
                prob_array_tri[unicode] = prob_dict_tri[key_tri]
            else:
                continue
    else:
        prob_array_tri = np.full(vocab_size, prob_dict_tri["unseen", "unseen", 'unseen'])

    return prob_array_tri



def generate_char(prob_dict_uni, prob_dict_bi, prob_dict_tri, history, lambda_list, vocab_size):
    # Input: lambda_list: a list of length 3 which contains parameters in interpolation and sum to 1.
    #        history: a list of length 2 [char i - 2, char i - 1].
    # Output: a single char.

    prob_array_uni = np.full(vocab_size, prob_dict_uni['unseen'])
    for key in prob_dict_uni:
        if key == "unseen":
            continue
        else:
            prob_array_uni[ord(key)] = prob_dict_uni[key]

    prob_array_bi = generate_prob_bi(prob_dict_bi, history, vocab_size)
    prob_array_tri = generate_prob_tri(prob_dict_tri, history, vocab_size)
    prob_array = lambda_list[0] * prob_array_uni + lambda_list[1] * prob_array_bi + lambda_list[2] * prob_array_tri

    multinomial_array = np.random.multinomial(1, prob_array)
    sampled_unicode = np.arange(vocab_size)[multinomial_array == 1]

    return chr(sampled_unicode)


def retrive_prob(prob_dict_uni, prob_dict_bi, prob_dict_tri ,new_char, history, lambda_list):
    # Input: prob_dict_x: a dictionary storing probabilities in model x.
    #        history: a list of
    # Output: base 2 log probability

    if new_char in prob_dict_uni:
        prob_uni = prob_dict_uni[new_char]
    else:
        prob_uni = prob_dict_uni["unseen"]

    bi_key = (history[1], new_char)
    if bi_key in prob_dict_bi:
        prob_bi = prob_dict_bi[bi_key]
    elif (history[1], "unseen") in prob_dict_bi:
        prob_bi = prob_dict_bi[(history[1], "unseen")]
    else:
        prob_bi = prob_dict_bi[("unseen", 'unseen')]

    if (history[0], history[1], new_char) in prob_dict_tri:
        prob_tri = prob_dict_tri[(history[0], history[1], new_char)]
    elif (history[0], history[1], "unseen") in prob_dict_tri:
        prob_tri = prob_dict_tri[(history[0], history[1], "unseen")]
    else:
        prob_tri = prob_dict_tri['unseen', 'unseen', 'unseen']

    prob = lambda_list[0] * prob_uni + lambda_list[1] * prob_bi + lambda_list[2] * prob_tri
    return np.log2(prob)