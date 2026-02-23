import pandas as pd
from datasets import load_dataset


def load_and_concat(dataset_name):
    # loading the dataset with the name
    ds = load_dataset(dataset_name)
    # see what sets are in there
    print("Keys:", ds.keys())
    # print("Len of sets:", [len(ds[key]) for key in ds.keys()])
    # collect all sets to pandas dfs in a list
    dfs = [ds[key].to_pandas() for key in ds.keys()]
    # concatenate
    final = pd.concat(dfs)
    print("Final len:", len(final))
    # return the result
    return final