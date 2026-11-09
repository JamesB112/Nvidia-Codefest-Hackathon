import json
import pandas as pd 

# Load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Parse the JSON file
    return data

# Example usage
file_path = '../data/raw/raw_scraped_dataset.json'  # Replace with the path to your JSON file
json_data = load_json(file_path)

# Print loaded JSON data
# print(json_data)
print(json_data[:5])  # This will print only the first 5 items if it's a list

# Now load into a dataframe
# Flatten the nested structure and create a DataFrame
flattened_data = [item[0] for item in json_data]  # Flatten out the inner list
df_all = pd.DataFrame(flattened_data)


# Add a unique column to each URL_MAIN (you can use a hash function or unique string)
df_all['URL_ID'] = pd.factorize(df_all['URL_MAIN'])[0] + 1


# Add an overall row ID (this will be the index of the DataFrame, starting from 1)
df_all['Row_ID'] = df_all.index + 1  # Adding a simple row ID, starting from 1

df_all = df_all[['Row_ID', 'URL_ID'] + [col for col in df_all.columns if col not in ['URL_ID', 'Unique_URL']]]




from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
from nemo_curator.utils.file_utils import get_batched_files
from nemo_curator.modules.modify import Modify
from nemo_curator.modifiers.pii_modifier import PiiModifier
import pandas as pd

import dask.dataframe as dd
import pandas as pd


modifier = PiiModifier(
    language="en",
    supported_entities=["PERSON", "EMAIL_ADDRESS"],
    anonymize_action="replace",
    batch_size=1000,
    device="gpu")

# Sample data with 'text' column
sample_data = [
    {"text": "Alice is a person and her email is alice@example.com"}]
# sample_data = json_data[0][0]
df = pd.DataFrame(sample_data, index=[0])
# text_df = df["Raw_Scraped_Text"]
# text_df = text_df.rename(columns={"Raw_Scraped_Text": "text"}, inplace=True)

# Convert pandas DataFrame to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=2)


dataset = DocumentDataset(ddf)
modify = Modify(modifier)
modified_dataset = modify(dataset)
print(modified_dataset)


# Extract the modified DataFrame (the underlying data)
modified_df = modified_dataset.df

# Now access the 'text' column and print the modified text
modified_texts = modified_df['text']
print(modified_texts)