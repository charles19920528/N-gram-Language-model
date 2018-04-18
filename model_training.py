import glob
import model_training_functions as mtf
import pickle

path = './*.txt'
txt_files = glob.glob(path)
# we add 2 special begin chars and one end of text symbols at the beginning and the end of each passage.
# we only use trigram model at most.
unicode_upper = int("ffff", 16)
beg_char = chr(unicode_upper + 1)
string_list = []
for file in txt_files:
    with open(file, "r") as txt:
        sdxl = txt.read()
        if sdxl[len(sdxl) - 1] != '\x03':
            sdxl = beg_char + beg_char + sdxl + chr(3)
            string_list.append(sdxl)

merged_string = "".join(string_list)
char_list = list(merged_string)

beg_char = chr(unicode_upper + 1)
vocab_size = unicode_upper + 1
perturb_mass_list = [0.0001, 0.0001, 0.0001]
prob_dict_uni, prob_dict_bi, prob_dict_tri = \
    mtf.language_model(merged_string, char_list, beg_char, vocab_size, perturb_mass_list)


def save_dict(dict, file_name):
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)


# save the dictionary
save_dict(prob_dict_uni, "prob_dict_uni")
save_dict(prob_dict_bi, "prob_dict_bi")
save_dict(prob_dict_tri, "prob_dict_tri")


# test = list(sdxl)[0:100]
# history = ["这", "是"]
# lambda_list = [0.3, 0.5, 0.2]
# for i in np.arange(100):
#     test[i] = mtf.generate_char(prob_dict_uni, prob_dict_bi, prob_dict_tri, history, lambda_list, unicode_upper + 1)
#     history = [history[1], test[i]]
#
# test