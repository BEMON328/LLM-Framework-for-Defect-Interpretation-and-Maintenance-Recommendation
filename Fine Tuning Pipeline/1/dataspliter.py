import pandas as pd

# Load training data
df = pd.read_csv('mistral_training_data_simple.csv')

# Extract prompt and response from [INST]...[/INST] format
test_data = []

for idx, row in df.iterrows():
    text = row['text']

    # Split on [INST] and [/INST]
    if '[INST]' in text and '[/INST]' in text:
        prompt = text.split('[INST]')[1].split('[/INST]')[0].strip()
        ground_truth = text.split('[/INST]')[1].strip()

        test_data.append({
            'UserInput': prompt,
            'Response': ground_truth,
            'Generated': ''  # Will be filled by model
        })

# Save as test CSV
test_df = pd.DataFrame(test_data)
test_df.to_csv('training_data_as_test_set.csv', index=False)

print(f"Created test set with {len(test_df)} samples")