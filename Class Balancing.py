import random
import numpy as np
from torch.utils.data import Dataset
from transformers import InputFeatures, T5ForConditionalGeneration, T5TokenizerFast
from copy import deepcopy

from nltk.corpus import wordnet
import re

# TrainerDataset object used as input to all models used in this study
class TrainerDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, evidences=None):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.evidences = evidences

        # Tokenize the input
        self.tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")   

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return InputFeatures(
            input_ids=self.tokenized_inputs['input_ids'][idx],
            attention_mask=self.tokenized_inputs['attention_mask'][idx],
            label=self.targets[idx])   



def add_imbalance(train_dataset, imbalance = 0.05, random_seed=123):
    """ Method used to perform random undersampling of given TrainerDataset object (by removing examples from positive class)

    Args:
        train_dataset (TrainerDataset): Dataset to imbalance
        imbalance (float, optional): Percentage of observations from positive class to be removed. Defaults to 0.05.
        random_seed (int, optional): Random state. Defaults to 123.

    Returns:
        TrainerDataset: Imbalanced dataset
    """
    np.random.seed(random_seed)
    classes, counts = np.unique(train_dataset.targets, return_counts=True)
    positive = np.array(train_dataset.inputs)[np.array(train_dataset.targets)==1]
    new_positive = [pos for pos in positive if np.random.random()<imbalance]
    new_inputs = new_positive + list(np.array(train_dataset.inputs)[np.array(train_dataset.targets)==0])
    new_targets = [1 if i<len(new_positive) else 0 for i in range(len(new_inputs))]
    np.random.seed(random_seed)
    np.random.shuffle(new_targets)
    np.random.seed(random_seed)
    np.random.shuffle(new_inputs)
    return TrainerDataset(new_inputs,
                               new_targets, train_dataset.tokenizer)




def balance_minority(train_dataset, fun, limited_range=False, random_seed=123, **kwargs):
    """ Method used to balance minority using some function fun (ex. replace_synonym, deepcopy, ...)

    Args:
        train_dataset (TrainerDataset): Dataset to balance
        fun (Callable): Function used to balance minority
        limited_range (bool, optional): In case of time consuming balancing functions, user may choose to limit number of new examples
                                        from the size difference between the samples to minimum of 500 and 5 * number of positive examples. 
                                        Defaults to False.
        random_seed (int, optional): Random state. Defaults to 123.

    Returns:
        TrainerDataset: Balanced dataset
    """
    random.seed(random_seed)
    if limited_range:
        global counter
        counter = 0
        
    positives = np.array(train_dataset.inputs)[np.array(train_dataset.targets)==1]
    n_positive = len(positives)
    n_negative = len(train_dataset.targets) - n_positive
    
    
    generation_count = np.min([n_negative-n_positive, 5*n_positive, 500]) if limited_range else n_negative-n_positive
    new_inputs = [
            fun(positives[np.random.randint(n_positive)], **kwargs) for i in range(generation_count)
        ]
    
    balanced_inputs = train_dataset.inputs + new_inputs
    balanced_targets = train_dataset.targets + [1 for _ in range(generation_count)]
    
    np.random.seed(random_seed)
    np.random.shuffle(balanced_targets)
    np.random.seed(random_seed)
    np.random.shuffle(balanced_inputs)
    
    return TrainerDataset(balanced_inputs, balanced_targets, train_dataset.tokenizer)





def get_synonym(word):
    """ Method returns randomly choosen synonym for a give word

    Args:
        word (str): Word to find synonym for

    Returns:
        str: Synonym in case there are synonyms for that word  else method returns the input word
    """
    tmp = wordnet.synsets(word)
    synonyms = np.unique([tmp[i].name().split(".")[0] for i in range(len(tmp)) if tmp[i].name().split(".")[0] != word])
    if len(synonyms)>0:
        return np.random.choice(synonyms)
    return word

def replace_synonym(sentence, prob = 0.2):
    """ Method replaces each word of a sentence with given probability into its synonym

    Args:
        sentence (str): Input sentence
        prob (float, optional): Probability of word replacemnt with its synonym. Defaults to 0.2.

    Returns:
        str: Resulting sentence
    """
    words = re.findall(r"[\w']+|[.,!?;]",sentence)
    new_words = [get_synonym(word) if np.random.random()<prob and word not in ['.',',','!','?',';'] else word for word in words]
    return re.sub(r"(?: ([.,;]))", r"\g<1>", " ".join(new_words))



# Counter used to monitor progress of paraphrasing
counter = 0

def paraphrase(sentences, paraphraser, tokenizer, cat = False, count = False):
    """ Method used for paraphrasing a text one sentence at a time

    Args:
        sentences (str): Text to paraphrase
        paraphraser (_type_): Paraphraser 
        tokenizer (_type_): Tokenizer
        cat (bool, optional): Flag for printing paraphrasing results. Defaults to False.
        count (bool, optional): Flag for printing progress counter. Defaults to False.

    Returns:
        str: Paraphrased sentence
    """
    if count:
        global counter 
        print(counter, end="\r")
        counter += 1
    # Paraphrase the sentences. Reviews are too long it's best to paraphrase one sentence at a time
    output = []
    reference = re.split(r'[.?!]', sentences)
    
    for sentence in reference:
        if len(sentence)>0 :

            # Tokenize the input sentence
            input_ids = tokenizer.encode(sentence, return_tensors='pt')

            if len(input_ids[0])>=50:
                output.append(sentence)
                continue
                
            # Generate paraphrased sentence
            paraphrase_ids = paraphraser.generate(input_ids, num_beams=5, max_length=1024, early_stopping=True)
        
            # Decode and print the paraphrased sentence
            paraphrase = tokenizer.decode(paraphrase_ids[0], skip_special_tokens=True, verbose=0)
            if cat:
                print(f"Original: {sentence}")
                print(f"Paraphrase: {paraphrase}")
                print()
            output.append(paraphrase)
        else:
            output.append(sentence)
    return " ".join(output)


paraphraser_t5small = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
tokenizer_t5small = T5TokenizerFast.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
def create_datasets(train_dataset, imbalance = 0.05, synonym_prob = 0.2, paraphraser = paraphraser_t5small, tokenizer = tokenizer_t5small, random_seed = 123):
    train_dataset_imbalanced = add_imbalance(train_dataset, imbalance, random_seed = random_seed)
    train_dataset_synonym = balance_minority(train_dataset_imbalanced, replace_synonym, **{"prob":synonym_prob}, random_seed = random_seed)
    train_dataset_ros = balance_minority(train_dataset_imbalanced, deepcopy, random_seed = random_seed)
    train_dataset_paraphrase = balance_minority(train_dataset_imbalanced, paraphrase, False, random_seed = random_seed, **{"paraphraser":paraphraser, "tokenizer":tokenizer, "count":True})
    return train_dataset_imbalanced, train_dataset_synonym, train_dataset_ros, train_dataset_paraphrase
    
