from itertools import product
from chaserner.data.simulator_constants import SEED, TEMPLATES, TASK_CATALOG, PERSON_CATALOG, DATE_CATALOG, TRAIN_DEV_TEST_SPLIT
import random
from chaserner.utils.logger import logger

#TODO: remove stuff like commas and periods before unrolling etc

random.seed(SEED)
def split_into_train_dev_test(data, train_dev_test_split=TRAIN_DEV_TEST_SPLIT):
    """
        Randomly split a list into three non-overlapping groups (train dev test).

        Parameters:
        - data (list): The list of elements to be split.

        Returns:
        - tuple: Three lists representing the three groups.
    """
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    train_size = int(len(shuffled_data) * train_dev_test_split[0])
    dev_size = int(len(shuffled_data) * train_dev_test_split[1])
    test_size = int(len(shuffled_data) * train_dev_test_split[2])
    train = shuffled_data[:train_size]
    dev = shuffled_data[train_size: train_size+dev_size]
    test = shuffled_data[train_size+dev_size:train_size+dev_size+test_size]
    return train, dev, test

def generate_sentences(templates, task_catalog, person_catalog, date_catalog):
    """
        Unroll the data in the templates using the catalogs

        Parameters:
        - templates (list): The list of templates to unroll.
        - task_catalog (list): task_catalog.
        - person_catalog (list): person_catalog.
        - date_catalog (list): date_catalog.

        Returns:
        - tuple: List of unrolled surface forms.
    """
    generated_sentences = []
    for template in templates:
        for task, person, date in product(task_catalog, person_catalog, date_catalog):
            sentence = template.format(task=task, person=person, date=date)
            label = []
            for tok in template.split():
                if "{task}" in tok:
                    if len(task):
                        label.append("B-task")
                        for tok_in_task in task.split()[1:]:
                            label.append("I-task")
                elif "{person}" in tok:
                    if len(person):
                        label.append("B-person")
                        for tok_in_person in person.split()[1:]:
                            label.append("I-person")
                # if "{task}" in tok:
                #     if len(task):
                #         label.append("O")
                #         for tok_in_task in task.split()[1:]:
                #             label.append("O")
                # elif "{person}" in tok:
                #     if len(person):
                #         label.append("O")
                #         for tok_in_person in person.split()[1:]:
                #             label.append("O")
                elif "{date}" in tok:
                    if len(date):
                        label.append("B-date")
                        for tok_in_date in date.split()[1:]:
                            label.append("I-date")
                else:
                    label.append("O")
            generated_sentences.append((sentence, dict(task=task, person=person, date=date), label))
    return generated_sentences

def simulate_train_dev_test():
    train_templates, dev_templates, test_templates = split_into_train_dev_test(TEMPLATES)
    train_tasks, dev_tasks, test_tasks = split_into_train_dev_test(TASK_CATALOG)
    train_persons, dev_persons, test_persons = split_into_train_dev_test(PERSON_CATALOG)
    train_dates, dev_dates, test_dates = split_into_train_dev_test(DATE_CATALOG)
    train_unrolled_sentences = generate_sentences(train_templates, train_tasks, train_persons, train_dates)
    dev_unrolled_sentences = generate_sentences(dev_templates, dev_tasks, dev_persons, dev_dates)
    test_unrolled_sentences = generate_sentences(test_templates, test_tasks, test_persons, test_dates)
    log_str = f"""
    Finished simulation.  Generated:
    TRAIN: {len(train_unrolled_sentences)}
    DEV: {len(dev_unrolled_sentences)}
    TEST: {len(test_unrolled_sentences)}
    """
    logger.info(log_str)
    return train_unrolled_sentences, dev_unrolled_sentences, test_unrolled_sentences




