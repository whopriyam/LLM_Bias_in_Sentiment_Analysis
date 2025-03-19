from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
dataset = load_dataset("LabHC/moji")

# Convert to Pandas DataFrame (assuming 'train', 'validation', 'test' splits exist)
train_df = pd.DataFrame(dataset['train'])
print (len(train_df))
train_df['text'] = train_df['text'].apply(lambda x: x.replace("_TWITTER-ENTITY_", ""))

# Filter based on 'sa' values
df_sa0 = train_df[train_df['sa'] == 0]
df_sa1 = train_df[train_df['sa'] == 1]

# Use .isin() to select n samples of label 1 and n samples of label 0 for each 'sa' value
df_sa0_balanced = df_sa0[df_sa0['label'].isin([0, 1])].groupby('label', group_keys=False).apply(lambda x: x.sample(n=200, random_state=42))
df_sa1_balanced = df_sa1[df_sa1['label'].isin([0, 1])].groupby('label', group_keys=False).apply(lambda x: x.sample(n=200, random_state=42))

# Concatenate and shuffle
final_dataset = pd.concat([df_sa0_balanced, df_sa1_balanced]).sample(frac=1, random_state=42)

#get dataframe description
print (len(final_dataset))
print (final_dataset.describe())
print (final_dataset['label'].value_counts())
print (final_dataset['sa'].value_counts())

# Reset index
final_dataset = final_dataset.reset_index(drop=True)
final_dataset.to_csv('final_data_balanced.csv', index=False)
