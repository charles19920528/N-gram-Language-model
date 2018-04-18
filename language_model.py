import sys
import pickle
import numpy as np
import model_training_functions as mtf

seed = int(sys.argv[1])
np.random.seed(seed)


def load_dict(dict_name):
    # Input: the name of the pkl file without .pkl
    # Output: a dictionary
    with open(dict_name + '.pkl', 'rb') as f:
        return pickle.load(f)


prob_dict_uni = load_dict("prob_dict_uni")
prob_dict_bi = load_dict("prob_dict_bi")
prob_dict_tri = load_dict("prob_dict_tri")


unicode_upper = int("ffff", 16)
vocab_size = unicode_upper + 1
lambda_list = [0.3, 0.5, 0.2]
begin_char = chr(unicode_upper + 1)
history = [begin_char, begin_char]

standard_input = sys.stdin.read()
# with open("a1.test_input.txt", "r") as txt:
#     standard_input = txt.read()

index = 0
while index < len(standard_input):
    if standard_input[index] == "o":
        if standard_input[index] == chr(3):
            history = [begin_char, begin_char]
            print("End of text reached. Clear the history.")
            index += 2
        else:
            history = [history[1], standard_input[index + 1]]
            print("Added a character to the history!")
            index += 2

    elif standard_input[index] == "q":
        new_char = standard_input[index + 1]
        prob_new_char = mtf.retrive_prob(prob_dict_uni, prob_dict_bi, prob_dict_tri, new_char, history, lambda_list)
        print(prob_new_char)
        index += 2

    elif standard_input[index] == "g":
        new_char = mtf.generate_char(prob_dict_uni, prob_dict_bi, prob_dict_tri, history, lambda_list, vocab_size)
        prob_new_char = mtf.retrive_prob(prob_dict_uni, prob_dict_bi, prob_dict_tri, new_char, history, lambda_list)
        history = [history[1], new_char]
        print(new_char+"//  generated a character! prob of generation: {}".format(prob_new_char))
        index += 1

    elif standard_input[index] == "x":
        exit(0)

    else:
        print("there is bug!!!!!!!!!!!!!!!!!!")
        exit(1)





