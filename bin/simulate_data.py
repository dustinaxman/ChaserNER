from chaserner.data.simulator import simulate_train_dev_test
import random

train, dev, test = simulate_train_dev_test()

all_data = train + dev + test

random.shuffle(all_data)


def proc_dict(dict_val):
    return " | ".join([k + ":" + v for k, v in dict_val.items()])


# with open(working_dir/"simulated_data.txt", "w") as f:
        #     #f.write("\n".join(["\t".join([" ".join([txt+"|"+lbl for txt, lbl in zip(txt_input.split(), raw_labels)]), proc_dict(labels)]) for txt_input, labels, raw_labels in all_data]))
        #     f.write("\n".join(["\t".join(
        #         [txt_input, proc_dict(labels), " ".join(raw_labels)]) for
        #                        txt_input, labels, raw_labels in all_data]))