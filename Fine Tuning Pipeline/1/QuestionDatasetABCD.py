# import pandas as pd
# import os
# from collections import Counter
# import re
#
#
# def read_dataset(csv_file):
#     """Read the bridge inspection dataset"""
#     try:
#         # Try different encodings and handle potential CSV issues
#         try:
#             df = pd.read_csv(separated_llama3_response_counted_v2.csv, encoding='utf-8')
#         except UnicodeDecodeError:
#             df = pd.read_csv(csv_file, encoding='latin-1')
#         except:
#             df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # Handle quotes properly
#
#         print(f"✅ Successfully loaded dataset: {len(df)} rows, {len(df.columns)} columns")
#         print(f"Columns: {list(df.columns)}")
#
#         # Check for required columns
#         if 'UserInput' not in df.columns or 'Response' not in df.columns:
#             print(f"❌ Error: Required columns 'UserInput' and 'Response' not found.")
#             print(f"Available columns: {list(df.columns)}")
#             return None
#
#         # Clean data
#         df = df.dropna(subset=['UserInput', 'Response'])  # Remove rows with missing values
#         df['UserInput'] = df['UserInput'].astype(str).str.strip()
#         df['Response'] = df['Response'].astype(str).str.strip()
#
#         print(f"✅ Data cleaned: {len(df)} valid rows remaining")
#         return df
#
#     except FileNotFoundError:
#         print(f"❌ Error: File '{csv_file}' not found.")
#         print(f"Current directory: {os.getcwd()}")
#         print(f"Files in directory: {os.listdir('.')}")
#         return None
#     except Exception as e:
#         print(f"❌ Error reading CSV: {e}")
#         print(f"Try checking if the CSV file has proper formatting or contains special characters")
#         return None
#
#
# def analyze_dataset(df):
#     """Analyze the dataset structure and content"""
#     print("\n" + "=" * 60)
#     print("DATASET ANALYSIS")
#     print("=" * 60)
#
#     # Basic statistics
#     print(f"Total rows: {len(df)}")
#     print(f"Unique responses: {df['Response'].nunique()}")
#
#     # Response frequency analysis
#     response_counts = df['Response'].value_counts()
#     print(f"\nResponse frequency distribution:")
#     print(f"Most frequent response: {response_counts.max()} occurrences")
#     print(f"Least frequent response: {response_counts.min()} occurrences")
#
#     # Show top 10 most frequent responses
#     print(f"\nTop 10 most frequent responses:")
#     for i, (response, count) in enumerate(response_counts.head(10).items(), 1):
#         print(f"  {i}. [{count}x] {response[:80]}{'...' if len(response) > 80 else ''}")
#
#     # Component analysis
#     component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck', 'bearing', 'joint', 'splice',
#                           'stiffener']
#     component_counts = {}
#
#     for component in component_keywords:
#         count = df['UserInput'].str.contains(component, case=False, na=False).sum()
#         if count > 0:
#             component_counts[component] = count
#
#     print(f"\nBridge components found in dataset:")
#     for component, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"  {component}: {count} instances")
#
#     return response_counts, component_counts
#
#
# def create_question_set_a(df):
#     """Create Question Set A: Technical Accuracy Assessment (Deduplicated)"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET A: Technical Accuracy Assessment")
#     print("=" * 60)
#
#     # Remove duplicates based on Response (keep first occurrence)
#     set_a = df.drop_duplicates(subset=['Response'], keep='first').reset_index(drop=True)
#
#     print(f"Original dataset: {len(df)} instances")
#     print(f"Question Set A: {len(set_a)} unique defect-repair pairs")
#     print(f"Reduction: {len(df) - len(set_a)} duplicate instances removed")
#
#     print(f"\nSample from Question Set A:")
#     for i, row in set_a.head(5).iterrows():
#         print(f"  {i + 1}. \"{row['UserInput']}\" → \"{row['Response']}\"")
#
#     return set_a
#
#
# def create_question_set_b(df):
#     """Create Question Set B: Linguistic Robustness Testing (All paraphrases)"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET B: Linguistic Robustness Testing")
#     print("=" * 60)
#
#     # Use full dataset (all paraphrases)
#     set_b = df.copy().reset_index(drop=True)
#
#     print(f"Question Set B: {len(set_b)} instances (full dataset with all paraphrases)")
#
#     # Show example of paraphrases for same response
#     response_counts = df['Response'].value_counts()
#     example_response = response_counts[response_counts > 1].index[0]  # Get a response with multiple instances
#
#     paraphrases = df[df['Response'] == example_response]
#     print(f"\nExample paraphrase group:")
#     print(f"Repair: \"{example_response}\"")
#     print(f"Paraphrases ({len(paraphrases)} total):")
#     for i, row in paraphrases.head(5).iterrows():
#         print(f"  {i + 1}. \"{row['UserInput']}\"")
#
#     return set_b
#
#
# def create_question_set_c(df, min_frequency=3):
#     """Create Question Set C: Decision Transparency Evaluation (High frequency defects)"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET C: Decision Transparency Evaluation")
#     print("=" * 60)
#
#     # Get responses that appear at least min_frequency times
#     response_counts = df['Response'].value_counts()
#     frequent_responses = response_counts[response_counts >= min_frequency].index.tolist()
#
#     # Filter dataset to include only frequent responses
#     set_c = df[df['Response'].isin(frequent_responses)].reset_index(drop=True)
#
#     print(f"Frequency threshold: ≥{min_frequency} occurrences")
#     print(f"Question Set C: {len(set_c)} instances")
#     print(f"Includes {len(frequent_responses)} different repair types")
#
#     print(f"\nHigh-frequency repairs:")
#     for i, (response, count) in enumerate(response_counts[response_counts >= min_frequency].head(10).items(), 1):
#         print(f"  {i}. [{count}x] {response[:70]}{'...' if len(response) > 70 else ''}")
#
#     return set_c
#
#
# def create_question_set_d(df):
#     """Create Question Set D: Cross-Component Generalisation"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET D: Cross-Component Generalisation")
#     print("=" * 60)
#
#     # Define bridge components
#     component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck', 'bearing', 'joint', 'splice',
#                           'stiffener']
#
#     # Find instances that contain component keywords
#     component_mask = df['UserInput'].str.contains('|'.join(component_keywords), case=False, na=False)
#     component_instances = df[component_mask].copy()
#
#     # Analyze which repairs work across multiple components
#     repair_to_components = {}
#
#     for _, row in component_instances.iterrows():
#         user_input_lower = row['UserInput'].lower()
#         found_components = [comp for comp in component_keywords if comp in user_input_lower]
#
#         if found_components:
#             response = row['Response']
#             if response not in repair_to_components:
#                 repair_to_components[response] = set()
#             repair_to_components[response].update(found_components)
#
#     # Filter to repairs that work on multiple components
#     cross_component_repairs = {repair: components for repair, components in repair_to_components.items()
#                                if len(components) > 1}
#
#     # Create Set D with instances suitable for cross-component testing
#     cross_component_responses = list(cross_component_repairs.keys())
#     set_d = component_instances[component_instances['Response'].isin(cross_component_responses)].reset_index(drop=True)
#
#     print(f"Total instances with component keywords: {len(component_instances)}")
#     print(f"Question Set D: {len(set_d)} instances suitable for cross-component testing")
#     print(f"Cross-component repairs: {len(cross_component_repairs)}")
#
#     print(f"\nSame repair methods across different components:")
#     for i, (repair, components) in enumerate(list(cross_component_repairs.items())[:5], 1):
#         components_str = ', '.join(sorted(components))
#         print(f"  {i}. \"{repair[:60]}{'...' if len(repair) > 60 else ''}\"")
#         print(f"     Components: {components_str}")
#
#     # Component distribution in Set D
#     component_distribution = {}
#     for component in component_keywords:
#         count = set_d['UserInput'].str.contains(component, case=False, na=False).sum()
#         if count > 0:
#             component_distribution[component] = count
#
#     print(f"\nComponent distribution in Set D:")
#     for component, count in sorted(component_distribution.items(), key=lambda x: x[1], reverse=True):
#         print(f"  {component}: {count} instances")
#
#     return set_d
#
#
# def save_datasets(set_a, set_b, set_c, set_d, output_dir="question_sets"):
#     """Save all question sets to CSV files"""
#     print("\n" + "=" * 60)
#     print("SAVING QUESTION SETS")
#     print("=" * 60)
#
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
#
#     datasets = {
#         'question_set_a_technical_accuracy.csv': set_a,
#         'question_set_b_linguistic_robustness.csv': set_b,
#         'question_set_c_decision_transparency.csv': set_c,
#         'question_set_d_cross_component_generalisation.csv': set_d
#     }
#
#     for filename, dataset in datasets.items():
#         filepath = os.path.join(output_dir, filename)
#         dataset.to_csv(filepath, index=False)
#         print(f"✅ Saved: {filepath} ({len(dataset)} rows)")
#
#     # Create summary file
#     summary_data = {
#         'Question_Set': ['Set A: Technical Accuracy', 'Set B: Linguistic Robustness',
#                          'Set C: Decision Transparency', 'Set D: Cross-Component Generalisation'],
#         'Size': [len(set_a), len(set_b), len(set_c), len(set_d)],
#         'Purpose': [
#             'Test basic correctness (deduplicated)',
#             'Test consistency across paraphrases',
#             'Test traceability for frequent defects',
#             'Test knowledge transfer across components'
#         ]
#     }
#
#     summary_df = pd.DataFrame(summary_data)
#     summary_path = os.path.join(output_dir, 'question_sets_summary.csv')
#     summary_df.to_csv(summary_path, index=False)
#     print(f"✅ Saved: {summary_path}")
#
#     return output_dir
#
#
# def main():
#     """Main function to orchestrate the dataset creation"""
#     print("🏗️ BRIDGE INSPECTION QUESTION SET GENERATOR")
#     print("=" * 60)
#
#     # Configuration - Update this filename to match your CSV file
#     csv_file = "separated_llama3_response_counted_v3.csv"  # Change this to your actual filename
#     min_frequency_threshold = 3  # Minimum frequency for Set C
#
#     # Alternative: Auto-detect CSV files in directory
#     if not os.path.exists(csv_file):
#         print(f"❌ File '{csv_file}' not found. Looking for CSV files in current directory...")
#         csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
#         if csv_files:
#             print(f"Available CSV files: {csv_files}")
#             if len(csv_files) == 1:
#                 csv_file = csv_files[0]
#                 print(f"✅ Using: {csv_file}")
#             else:
#                 print("Multiple CSV files found. Please specify the correct filename in the script.")
#                 return
#         else:
#             print("No CSV files found in current directory.")
#             return
#
#     # Step 1: Read dataset
#     df = read_dataset(csv_file)
#     if df is None:
#         return
#
#     # Step 2: Analyze dataset
#     response_counts, component_counts = analyze_dataset(df)
#
#     # Step 3: Create question sets
#     set_a = create_question_set_a(df)
#     set_b = create_question_set_b(df)
#     set_c = create_question_set_c(df, min_frequency_threshold)
#     set_d = create_question_set_d(df)
#
#     # Step 4: Save datasets
#     output_dir = save_datasets(set_a, set_b, set_c, set_d)
#
#     # Step 5: Final summary
#     print("\n" + "=" * 60)
#     print("FINAL SUMMARY")
#     print("=" * 60)
#     print(f"📁 All question sets saved to: ./{output_dir}/")
#     print(f"📊 Question Set A (Technical Accuracy): {len(set_a)} instances")
#     print(f"📊 Question Set B (Linguistic Robustness): {len(set_b)} instances")
#     print(f"📊 Question Set C (Decision Transparency): {len(set_c)} instances")
#     print(f"📊 Question Set D (Cross-Component Generalisation): {len(set_d)} instances")
#     print(f"🎯 Ready for evaluation against: Correctness, Robustness, Traceability, Generalisation")
#     print("\n✅ Question set generation completed successfully!")
#
#
# if __name__ == "__main__":
#     main()
# import pandas as pd
# import os
# from collections import Counter
# import re
# import hashlib

# def create_cluster_id(df):
#     """Create a cluster ID column based on the hash of the Response"""
#     print("\nCreating Cluster IDs...")
#     df['Cluster_ID'] = df['Response'].apply(
#         lambda x: hashlib.md5(x.encode()).hexdigest()[:8]  # Use first 8 chars of MD5 hash
#     )
#     print(f"✅ Created {df['Cluster_ID'].nunique()} unique cluster IDs")
#     return df
#
# def read_dataset(csv_file):
#     """Read the bridge inspection dataset"""
#     try:
#         # Try different encodings and handle potential CSV issues
#         try:
#             df = pd.read_csv(csv_file, encoding='utf-8')
#         except UnicodeDecodeError:
#             df = pd.read_csv(csv_file, encoding='latin-1')
#         except:
#             df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # Handle quotes properly
#
#         print(f"✅ Successfully loaded dataset: {len(df)} rows, {len(df.columns)} columns")
#         print(f"Columns: {list(df.columns)}")
#
#         # Check for required columns
#         if 'UserInput' not in df.columns or 'Response' not in df.columns:
#             print(f"❌ Error: Required columns 'UserInput' and 'Response' not found.")
#             print(f"Available columns: {list(df.columns)}")
#             return None
#
#         # Clean data
#         df = df.dropna(subset=['UserInput', 'Response'])  # Remove rows with missing values
#         df['UserInput'] = df['UserInput'].astype(str).str.strip()
#         df['Response'] = df['Response'].astype(str).str.strip()
#
#         # Add cluster IDs
#         df = create_cluster_id(df)
#
#         print(f"✅ Data cleaned: {len(df)} valid rows remaining")
#         return df
#
#     except FileNotFoundError:
#         print(f"❌ Error: File '{csv_file}' not found.")
#         print(f"Current directory: {os.getcwd()}")
#         print(f"Files in directory: {os.listdir('.')}")
#         return None
#     except Exception as e:
#         print(f"❌ Error reading CSV: {e}")
#         print(f"Try checking if the CSV file has proper formatting or contains special characters")
#         return None
#
# def analyze_dataset(df):
#     """Analyze the dataset structure and content"""
#     print("\n" + "=" * 60)
#     print("DATASET ANALYSIS")
#     print("=" * 60)
#
#     # Basic statistics
#     print(f"Total rows: {len(df)}")
#     print(f"Unique responses: {df['Response'].nunique()}")
#     print(f"Unique clusters: {df['Cluster_ID'].nunique()}")
#
#     # Response frequency analysis
#     response_counts = df['Response'].value_counts()
#     print(f"\nResponse frequency distribution:")
#     print(f"Most frequent response: {response_counts.max()} occurrences")
#     print(f"Least frequent response: {response_counts.min()} occurrences")
#
#     # Show top 10 most frequent responses
#     print(f"\nTop 10 most frequent responses:")
#     for i, (response, count) in enumerate(response_counts.head(10).items(), 1):
#         print(f"  {i}. [{count}x] {response[:80]}{'...' if len(response) > 80 else ''}")
#
#     # Component analysis
#     component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck', 'bearing', 'joint', 'splice',
#                           'stiffener']
#     component_counts = {}
#
#     for component in component_keywords:
#         count = df['UserInput'].str.contains(component, case=False, na=False).sum()
#         if count > 0:
#             component_counts[component] = count
#
#     print(f"\nBridge components found in dataset:")
#     for component, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"  {component}: {count} instances")
#
#     return response_counts, component_counts
#
# def create_question_set_a(df):
#     """Create Question Set A: Technical Accuracy Assessment (Deduplicated)"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET A: Technical Accuracy Assessment")
#     print("=" * 60)
#
#     # Remove duplicates based on Cluster_ID (keep first occurrence)
#     set_a = df.drop_duplicates(subset=['Cluster_ID'], keep='first').reset_index(drop=True)
#
#     print(f"Original dataset: {len(df)} instances")
#     print(f"Question Set A: {len(set_a)} unique defect-repair pairs")
#     print(f"Reduction: {len(df) - len(set_a)} duplicate instances removed")
#
#     print(f"\nSample from Question Set A:")
#     for i, row in set_a.head(5).iterrows():
#         print(f"  {i + 1}. Cluster {row['Cluster_ID']}: \"{row['UserInput']}\" → \"{row['Response']}\"")
#
#     return set_a
#
# def create_question_set_b(df):
#     """Create Question Set B: Linguistic Robustness Testing (All paraphrases)"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET B: Linguistic Robustness Testing")
#     print("=" * 60)
#
#     # Use full dataset (all paraphrases)
#     set_b = df.copy().reset_index(drop=True)
#
#     print(f"Question Set B: {len(set_b)} instances (full dataset with all paraphrases)")
#
#     # Show example of paraphrases for same cluster
#     cluster_counts = df['Cluster_ID'].value_counts()
#     example_cluster = cluster_counts[cluster_counts > 1].index[0]  # Get a cluster with multiple instances
#
#     paraphrases = df[df['Cluster_ID'] == example_cluster]
#     print(f"\nExample paraphrase group (Cluster {example_cluster}):")
#     print(f"Repair: \"{paraphrases.iloc[0]['Response']}\"")
#     print(f"Paraphrases ({len(paraphrases)} total):")
#     for i, row in paraphrases.head(5).iterrows():
#         print(f"  {i + 1}. \"{row['UserInput']}\"")
#
#     return set_b
#
# def create_question_set_c(df, min_frequency=3):
#     """Create Question Set C: Decision Transparency Evaluation (High frequency defects)"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET C: Decision Transparency Evaluation")
#     print("=" * 60)
#
#     # Get responses that appear at least min_frequency times
#     response_counts = df['Response'].value_counts()
#     frequent_responses = response_counts[response_counts >= min_frequency].index.tolist()
#
#     # Filter dataset to include only frequent responses
#     set_c = df[df['Response'].isin(frequent_responses)].reset_index(drop=True)
#
#     print(f"Frequency threshold: ≥{min_frequency} occurrences")
#     print(f"Question Set C: {len(set_c)} instances")
#     print(f"Includes {len(frequent_responses)} different repair types")
#
#     print(f"\nHigh-frequency repairs:")
#     for i, (response, count) in enumerate(response_counts[response_counts >= min_frequency].head(10).items(), 1):
#         print(f"  {i}. [{count}x] {response[:70]}{'...' if len(response) > 70 else ''}")
#
#     return set_c
#
# def create_question_set_d(df):
#     """Create Question Set D: Cross-Component Generalisation"""
#     print("\n" + "=" * 60)
#     print("CREATING QUESTION SET D: Cross-Component Generalisation")
#     print("=" * 60)
#
#     # Define bridge components
#     component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck', 'bearing', 'joint', 'splice',
#                           'stiffener']
#
#     # Find instances that contain component keywords
#     component_mask = df['UserInput'].str.contains('|'.join(component_keywords), case=False, na=False)
#     component_instances = df[component_mask].copy()
#
#     # Analyze which repairs work across multiple components
#     repair_to_components = {}
#
#     for _, row in component_instances.iterrows():
#         user_input_lower = row['UserInput'].lower()
#         found_components = [comp for comp in component_keywords if comp in user_input_lower]
#
#         if found_components:
#             response = row['Response']
#             if response not in repair_to_components:
#                 repair_to_components[response] = set()
#             repair_to_components[response].update(found_components)
#
#     # Filter to repairs that work on multiple components
#     cross_component_repairs = {repair: components for repair, components in repair_to_components.items()
#                                if len(components) > 1}
#
#     # Create Set D with instances suitable for cross-component testing
#     cross_component_responses = list(cross_component_repairs.keys())
#     set_d = component_instances[component_instances['Response'].isin(cross_component_responses)].reset_index(drop=True)
#
#     print(f"Total instances with component keywords: {len(component_instances)}")
#     print(f"Question Set D: {len(set_d)} instances suitable for cross-component testing")
#     print(f"Cross-component repairs: {len(cross_component_repairs)}")
#
#     print(f"\nSame repair methods across different components:")
#     for i, (repair, components) in enumerate(list(cross_component_repairs.items())[:5], 1):
#         components_str = ', '.join(sorted(components))
#         print(f"  {i}. Cluster {df[df['Response'] == repair].iloc[0]['Cluster_ID']}: \"{repair[:60]}{'...' if len(repair) > 60 else ''}\"")
#         print(f"     Components: {components_str}")
#
#     # Component distribution in Set D
#     component_distribution = {}
#     for component in component_keywords:
#         count = set_d['UserInput'].str.contains(component, case=False, na=False).sum()
#         if count > 0:
#             component_distribution[component] = count
#
#     print(f"\nComponent distribution in Set D:")
#     for component, count in sorted(component_distribution.items(), key=lambda x: x[1], reverse=True):
#         print(f"  {component}: {count} instances")
#
#     return set_d
#
# def save_datasets(set_a, set_b, set_c, set_d, output_dir="question_sets"):
#     """Save all question sets to CSV files"""
#     print("\n" + "=" * 60)
#     print("SAVING QUESTION SETS")
#     print("=" * 60)
#
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
#
#     datasets = {
#         'question_set_a_technical_accuracy.csv': set_a,
#         'question_set_b_linguistic_robustness.csv': set_b,
#         'question_set_c_decision_transparency.csv': set_c,
#         'question_set_d_cross_component_generalisation.csv': set_d
#     }
#
#     for filename, dataset in datasets.items():
#         filepath = os.path.join(output_dir, filename)
#         dataset.to_csv(filepath, index=False)
#         print(f"✅ Saved: {filepath} ({len(dataset)} rows, {dataset['Cluster_ID'].nunique()} clusters)")
#
#     # Create summary file
#     summary_data = {
#         'Question_Set': ['Set A: Technical Accuracy', 'Set B: Linguistic Robustness',
#                          'Set C: Decision Transparency', 'Set D: Cross-Component Generalisation'],
#         'Size': [len(set_a), len(set_b), len(set_c), len(set_d)],
#         'Clusters': [set_a['Cluster_ID'].nunique(), set_b['Cluster_ID'].nunique(),
#                      set_c['Cluster_ID'].nunique(), set_d['Cluster_ID'].nunique()],
#         'Purpose': [
#             'Test basic correctness (deduplicated)',
#             'Test consistency across paraphrases',
#             'Test traceability for frequent defects',
#             'Test knowledge transfer across components'
#         ]
#     }
#
#     summary_df = pd.DataFrame(summary_data)
#     summary_path = os.path.join(output_dir, 'question_sets_summary.csv')
#     summary_df.to_csv(summary_path, index=False)
#     print(f"✅ Saved: {summary_path}")
#
#     return output_dir
#
# def main():
#     """Main function to orchestrate the dataset creation"""
#     print("🏗️ BRIDGE INSPECTION QUESTION SET GENERATOR")
#     print("=" * 60)
#
#     # Configuration - Update this filename to match your CSV file
#     csv_file = "separated_llama3_response_counted_v3.csv"  # Change this to your actual filename
#     min_frequency_threshold = 3  # Minimum frequency for Set C
#
#     # Alternative: Auto-detect CSV files in directory
#     if not os.path.exists(csv_file):
#         print(f"❌ File '{csv_file}' not found. Looking for CSV files in current directory...")
#         csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
#         if csv_files:
#             print(f"Available CSV files: {csv_files}")
#             if len(csv_files) == 1:
#                 csv_file = csv_files[0]
#                 print(f"✅ Using: {csv_file}")
#             else:
#                 print("Multiple CSV files found. Please specify the correct filename in the script.")
#                 return
#         else:
#             print("No CSV files found in current directory.")
#             return
#
#     # Step 1: Read dataset
#     df = read_dataset(csv_file)
#     if df is None:
#         return
#
#     # Step 2: Analyze dataset
#     response_counts, component_counts = analyze_dataset(df)
#
#     # Step 3: Create question sets
#     set_a = create_question_set_a(df)
#     set_b = create_question_set_b(df)
#     set_c = create_question_set_c(df, min_frequency_threshold)
#     set_d = create_question_set_d(df)
#
#     # Step 4: Save datasets
#     output_dir = save_datasets(set_a, set_b, set_c, set_d)
#
#     # Step 5: Final summary
#     print("\n" + "=" * 60)
#     print("FINAL SUMMARY")
#     print("=" * 60)
#     print(f"📁 All question sets saved to: ./{output_dir}/")
#     print(f"📊 Question Set A (Technical Accuracy): {len(set_a)} instances, {set_a['Cluster_ID'].nunique()} clusters")
#     print(f"📊 Question Set B (Linguistic Robustness): {len(set_b)} instances, {set_b['Cluster_ID'].nunique()} clusters")
#     print(f"📊 Question Set C (Decision Transparency): {len(set_c)} instances, {set_c['Cluster_ID'].nunique()} clusters")
#     print(f"📊 Question Set D (Cross-Component Generalisation): {len(set_d)} instances, {set_d['Cluster_ID'].nunique()} clusters")
#     print(f"🎯 Ready for evaluation against: Correctness, Robustness, Traceability, Generalisation")
#     print("\n✅ Question set generation completed successfully!")
#
# if __name__ == "__main__":
#     main()

import pandas as pd
import hashlib
import os
import sys
from typing import List, Dict, Tuple


def create_cluster_id(df):
    """Create a cluster ID column based on the hash of the Response"""
    print("\nCreating Cluster IDs...")
    df['Cluster_ID'] = df['Response'].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()[:8]  # Use first 8 chars of MD5 hash
    )
    print(f"✅ Created {df['Cluster_ID'].nunique()} unique cluster IDs")
    return df


def read_dataset(csv_file):
    """Read the bridge inspection dataset"""
    try:
        # Try different encodings and handle potential CSV issues
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1')
        except:
            df = pd.read_csv(csv_file, encoding='utf-8', quoting=1)  # Handle quotes properly

        print(f"✅ Successfully loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        # Check for required columns
        if 'UserInput' not in df.columns or 'Response' not in df.columns:
            print(f"❌ Error: Required columns 'UserInput' and 'Response' not found.")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Clean data
        df = df.dropna(subset=['UserInput', 'Response'])  # Remove rows with missing values
        df['UserInput'] = df['UserInput'].astype(str).str.strip()
        df['Response'] = df['Response'].astype(str).str.strip()

        # Add cluster IDs
        df = create_cluster_id(df)

        print(f"✅ Data cleaned: {len(df)} valid rows remaining")
        return df

    except FileNotFoundError:
        print(f"❌ Error: File '{csv_file}' not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return None
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        print(f"Try checking if the CSV file has proper formatting or contains special characters")
        return None


def analyze_dataset(df):
    """Analyze the dataset structure and content"""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    # Basic statistics
    print(f"Total rows: {len(df)}")
    print(f"Unique responses: {df['Response'].nunique()}")
    print(f"Unique clusters: {df['Cluster_ID'].nunique()}")

    # Response frequency analysis
    response_counts = df['Response'].value_counts()
    print(f"\nResponse frequency distribution:")
    print(f"Most frequent response: {response_counts.max()} occurrences")
    print(f"Least frequent response: {response_counts.min()} occurrences")

    # Show top 10 most frequent responses
    print(f"\nTop 10 most frequent responses:")
    for i, (response, count) in enumerate(response_counts.head(10).items(), 1):
        print(f"  {i}. [{count}x] {response[:80]}{'...' if len(response) > 80 else ''}")

    # Component analysis
    component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck', 'bearing', 'joint', 'splice',
                          'stiffener']
    component_counts = {}

    for component in component_keywords:
        count = df['UserInput'].str.contains(component, case=False, na=False).sum()
        if count > 0:
            component_counts[component] = count

    print(f"\nBridge components found in dataset:")
    for component, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {component}: {count} instances")

    return response_counts, component_counts


def create_question_set_a(df):
    """Create Question Set A: Technical Accuracy Assessment (Deduplicated)"""
    print("\n" + "=" * 60)
    print("CREATING QUESTION SET A: Technical Accuracy Assessment")
    print("=" * 60)

    # Remove duplicates based on Cluster_ID (keep first occurrence)
    set_a = df.drop_duplicates(subset=['Cluster_ID'], keep='first').reset_index(drop=True)

    print(f"Original dataset: {len(df)} instances")
    print(f"Question Set A: {len(set_a)} unique defect-repair pairs")
    print(f"Reduction: {len(df) - len(set_a)} duplicate instances removed")

    print(f"\nSample from Question Set A:")
    for i, row in set_a.head(5).iterrows():
        print(f"  {i + 1}. Cluster {row['Cluster_ID']}: \"{row['UserInput'][:60]}...\" → \"{row['Response'][:60]}...\"")

    return set_a


def generate_paraphrases_with_llama(defect_description: str, num_paraphrases: int = 3) -> List[str]:
    """
    Generate paraphrased versions of defect descriptions using LLaMA-3

    This function uses a local LLaMA-3 model to create linguistic variations
    of the same defect description using varied technical vocabulary.

    Args:
        defect_description: Original defect description
        num_paraphrases: Number of paraphrases to generate

    Returns:
        List of paraphrased descriptions
    """
    try:
        # Try to import ollama for local LLaMA-3 inference
        try:
            import ollama

            prompt = f"""You are a bridge inspection expert. Paraphrase the following defect description {num_paraphrases} times using different technical vocabulary and phrasing, while maintaining the same technical meaning.

Original defect description: {defect_description}

Generate {num_paraphrases} paraphrased versions, each on a new line starting with a number (1., 2., 3., etc.). Use varied technical terminology but preserve the exact technical meaning."""

            response = ollama.generate(model='llama3', prompt=prompt)

            # Parse the response to extract paraphrases
            paraphrases = []
            lines = response['response'].strip().split('\n')
            for line in lines:
                # Remove numbering and clean up
                cleaned = line.strip()
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                        break
                if cleaned and len(cleaned) > 10:  # Valid paraphrase
                    paraphrases.append(cleaned)

            return paraphrases[:num_paraphrases]

        except ImportError:
            print("⚠️  Warning: ollama not installed. Install with: pip install ollama")
            print("⚠️  Falling back to rule-based paraphrasing...")
            return generate_paraphrases_rule_based(defect_description, num_paraphrases)

    except Exception as e:
        print(f"⚠️  Warning: LLaMA-3 paraphrase generation failed: {e}")
        print("⚠️  Falling back to rule-based paraphrasing...")
        return generate_paraphrases_rule_based(defect_description, num_paraphrases)


def generate_paraphrases_rule_based(defect_description: str, num_paraphrases: int = 3) -> List[str]:
    """
    Fallback rule-based paraphrasing when LLaMA-3 is not available

    Creates linguistic variations by:
    - Synonym substitution for common bridge terms
    - Structural rearrangement of sentences
    - Adding/removing technical details
    """
    # Define synonym mappings for bridge inspection terminology
    synonyms = {
        'corrosion': ['rust', 'oxidation', 'deterioration', 'material degradation'],
        'crack': ['fracture', 'fissure', 'split', 'break'],
        'damage': ['defect', 'deterioration', 'degradation', 'impairment'],
        'severe': ['significant', 'major', 'substantial', 'considerable'],
        'minor': ['small', 'slight', 'limited', 'minimal'],
        'section': ['area', 'portion', 'region', 'zone'],
        'observed': ['detected', 'identified', 'noted', 'found'],
        'present': ['evident', 'visible', 'apparent', 'occurring'],
    }

    paraphrases = []
    original_lower = defect_description.lower()

    for i in range(num_paraphrases):
        paraphrase = defect_description

        # Apply synonym substitutions
        for original, replacements in synonyms.items():
            if original in original_lower and i < len(replacements):
                paraphrase = paraphrase.replace(original, replacements[i % len(replacements)])
                paraphrase = paraphrase.replace(original.capitalize(), replacements[i % len(replacements)].capitalize())

        # Add variation markers
        if i == 0:
            paraphrase = f"Inspection shows: {paraphrase}"
        elif i == 1:
            paraphrase = f"Visual examination reveals {paraphrase.lower()}"
        elif i == 2:
            paraphrase = f"Field assessment indicates {paraphrase.lower()}"

        paraphrases.append(paraphrase)

    return paraphrases


def create_question_set_b(set_a: pd.DataFrame, num_paraphrases: int = 3) -> pd.DataFrame:
    """
    Create Question Set B: Linguistic Robustness Testing

    According to the paper:
    "Question Set B is selected for linguistic robustness testing by taking the
    deduplicated Question Set A and introducing linguistic variation. Local LLM
    (i.e. LLaMA-3) is used to generate multiple paraphrased versions of each
    defect description, creating different ways to describe the same defect using
    varied technical vocabulary."

    Args:
        set_a: Deduplicated Question Set A
        num_paraphrases: Number of paraphrases to generate per defect

    Returns:
        DataFrame with original + paraphrased defect descriptions
    """
    print("\n" + "=" * 60)
    print("CREATING QUESTION SET B: Linguistic Robustness Testing")
    print("=" * 60)
    print(f"Generating {num_paraphrases} paraphrases per defect using LLaMA-3...")

    set_b_data = []

    total = len(set_a)
    for idx, row in set_a.iterrows():
        # Add original description
        set_b_data.append(row.to_dict())

        # Generate paraphrases
        print(f"  Progress: {idx + 1}/{total} - Generating paraphrases for Cluster {row['Cluster_ID']}", end='\r')

        paraphrases = generate_paraphrases_with_llama(row['UserInput'], num_paraphrases)

        # Add each paraphrase as a new row with same Cluster_ID and Response
        for i, paraphrase in enumerate(paraphrases, 1):
            paraphrase_row = row.to_dict()
            paraphrase_row['UserInput'] = paraphrase
            paraphrase_row['Paraphrase_Number'] = i
            set_b_data.append(paraphrase_row)

    print()  # New line after progress
    set_b = pd.DataFrame(set_b_data).reset_index(drop=True)

    print(f"\n✅ Question Set B created:")
    print(f"   Original descriptions: {len(set_a)}")
    print(f"   Total with paraphrases: {len(set_b)}")
    print(f"   Paraphrases per defect: {num_paraphrases}")
    print(f"   Unique clusters: {set_b['Cluster_ID'].nunique()}")

    # Show example paraphrase group
    example_cluster = set_a.iloc[0]['Cluster_ID']
    paraphrase_group = set_b[set_b['Cluster_ID'] == example_cluster]
    print(f"\nExample paraphrase group (Cluster {example_cluster}):")
    print(f"Expected repair: \"{paraphrase_group.iloc[0]['Response'][:70]}...\"")
    print(f"Linguistic variations ({len(paraphrase_group)} total):")
    for i, row in paraphrase_group.iterrows():
        marker = "[ORIGINAL]" if i == 0 else f"[PARA-{row.get('Paraphrase_Number', '?')}]"
        print(f"  {marker} \"{row['UserInput'][:70]}...\"")

    return set_b


def create_question_set_c(df, min_frequency=3):
    """Create Question Set C: Decision Transparency Evaluation (High frequency defects)"""
    print("\n" + "=" * 60)
    print("CREATING QUESTION SET C: Decision Transparency Evaluation")
    print("=" * 60)

    # Get responses that appear at least min_frequency times
    response_counts = df['Response'].value_counts()
    frequent_responses = response_counts[response_counts >= min_frequency].index.tolist()

    # Filter dataset to include only frequent responses
    set_c = df[df['Response'].isin(frequent_responses)].reset_index(drop=True)

    print(f"Frequency threshold: ≥{min_frequency} occurrences")
    print(f"Question Set C: {len(set_c)} instances")
    print(f"Includes {len(frequent_responses)} different repair types")

    print(f"\nHigh-frequency repairs:")
    for i, (response, count) in enumerate(response_counts[response_counts >= min_frequency].head(10).items(), 1):
        print(f"  {i}. [{count}x] {response[:70]}{'...' if len(response) > 70 else ''}")

    return set_c


def create_question_set_d(df: pd.DataFrame,
                          holdout_ratio: float = 0.3,
                          use_primary_component_strategy: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create Question Set D: Cross-Component Generalisation Testing

    According to the paper:
    "Question Set D is selected for cross-component generalisation testing by
    identifying defect instances that occur on different bridge components but
    require the same repair methods. Certain defect instances are withheld from
    the system during training/knowledge base construction, and the system is
    then tested on defect types that have been previously encountered but
    occurring on unseen bridge components."

    Strategy (UPDATED for multi-component defects):
    1. Identify repairs that work across multiple components
    2. For each repair, hold out specific components for testing
    3. Use PRIMARY COMPONENT (first mentioned) to assign train/test
       - This handles multi-component defects (e.g., "girder flange")
    4. Training set: Instances where primary component is in training set
    5. Test set: Instances where primary component is held-out

    Args:
        df: Full dataset
        holdout_ratio: Proportion of components to hold out per repair (default: 0.3)
        use_primary_component_strategy: If True, use first-mentioned component for splitting

    Returns:
        Tuple of (training_set, test_set, metadata)
    """
    print("\n" + "=" * 60)
    print("CREATING QUESTION SET D: Cross-Component Generalisation")
    print("=" * 60)

    if use_primary_component_strategy:
        print("📋 Strategy: PRIMARY COMPONENT (handles multi-component defects)")
        print("   Split based on first-mentioned component in defect description")

    # Define bridge components
    component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck',
                          'bearing', 'joint', 'splice', 'stiffener']

    # Step 1: Find instances that contain component keywords
    component_mask = df['UserInput'].str.contains('|'.join(component_keywords), case=False, na=False)
    component_instances = df[component_mask].copy()

    # Step 2: Build repair-to-components mapping and identify primary components
    repair_to_components = {}
    instance_component_map = {}  # Track which components each instance has

    for idx, row in component_instances.iterrows():
        user_input_lower = row['UserInput'].lower()

        # Find all components and their positions
        found_components = []
        component_positions = []
        for comp in component_keywords:
            pos = user_input_lower.find(comp)
            if pos != -1:
                found_components.append(comp)
                component_positions.append(pos)

        if found_components:
            response = row['Response']
            cluster_id = row['Cluster_ID']

            # Determine primary component (first mentioned in text)
            if use_primary_component_strategy and len(found_components) > 1:
                # Sort by position to get first-mentioned component
                sorted_components = [comp for _, comp in sorted(zip(component_positions, found_components))]
                primary_component = sorted_components[0]
            else:
                primary_component = found_components[0]

            # Track repair-to-components
            if response not in repair_to_components:
                repair_to_components[response] = set()
            repair_to_components[response].update(found_components)

            # Track instance-to-components
            instance_component_map[idx] = {
                'components': found_components,
                'primary_component': primary_component,
                'cluster_id': cluster_id,
                'response': response
            }

    # Step 3: Filter to repairs that work on multiple components (≥2)
    cross_component_repairs = {
        repair: components
        for repair, components in repair_to_components.items()
        if len(components) >= 2
    }

    print(f"Total instances with component keywords: {len(component_instances)}")
    print(f"Cross-component repairs identified: {len(cross_component_repairs)}")

    # Step 4: Create train/test split with component holdout
    train_indices = []
    test_indices = []
    holdout_info = {}

    for repair, components in cross_component_repairs.items():
        components_list = sorted(list(components))
        n_components = len(components_list)
        n_holdout = max(1, int(n_components * holdout_ratio))

        # Hold out components for testing (e.g., last N components alphabetically)
        # This ensures reproducibility
        held_out_components = set(components_list[-n_holdout:])
        training_components = set(components_list[:-n_holdout])

        holdout_info[repair] = {
            'all_components': components_list,
            'training_components': list(training_components),
            'held_out_components': list(held_out_components),
            'n_components': n_components,
            'n_holdout': n_holdout
        }

        # Assign instances to train or test based on PRIMARY component
        for idx, instance_data in instance_component_map.items():
            if instance_data['response'] == repair:
                primary_comp = instance_data['primary_component']

                # Test set: primary component is held-out
                if primary_comp in held_out_components:
                    test_indices.append(idx)
                # Training set: primary component is in training set
                elif primary_comp in training_components:
                    train_indices.append(idx)

    # Create datasets
    set_d_train = component_instances.loc[train_indices].reset_index(drop=True)
    set_d_test = component_instances.loc[test_indices].reset_index(drop=True)

    # Add split indicator and primary component info
    set_d_train['Split'] = 'Train'
    set_d_train['Primary_Component'] = [instance_component_map[idx]['primary_component']
                                        for idx in train_indices]

    set_d_test['Split'] = 'Test (Held-out Components)'
    set_d_test['Primary_Component'] = [instance_component_map[idx]['primary_component']
                                       for idx in test_indices]

    # Combine for full Question Set D
    set_d_full = pd.concat([set_d_train, set_d_test], ignore_index=True)

    print(f"\n✅ Question Set D created with train/test split:")
    print(f"   Training instances: {len(set_d_train)} (seen components)")
    print(f"   Test instances: {len(set_d_test)} (held-out components)")
    print(f"   Total: {len(set_d_full)}")
    print(f"   Unique clusters: {set_d_full['Cluster_ID'].nunique()}")

    if len(set_d_test) == 0:
        print(f"\n⚠️  WARNING: Test set is empty!")
        print(f"   This may indicate insufficient cross-component repairs in the dataset.")
    else:
        print(
            f"\n✅ Test set successfully created ({len(set_d_test)}/{len(set_d_full)} = {len(set_d_test) / len(set_d_full) * 100:.1f}%)")

        # Show test set examples
        print(f"\nTest set examples (held-out component defects):")
        for i, row in set_d_test.head(3).iterrows():
            print(f"  {i + 1}. Primary: [{row['Primary_Component']}] \"{row['UserInput'][:55]}...\"")

    # Show examples of held-out strategy
    print(f"\nComponent holdout strategy examples:")
    for i, (repair, info) in enumerate(list(holdout_info.items())[:3], 1):
        print(f"\n  {i}. Repair: \"{repair[:60]}...\"")
        print(f"     All components: {', '.join(info['all_components'])}")
        print(f"     ✓ Training components: {', '.join(info['training_components'])}")
        print(f"     ✗ Held-out components: {', '.join(info['held_out_components'])}")

    # Component distribution
    print(f"\nComponent distribution in Set D:")
    for split_name in ['Train', 'Test (Held-out Components)']:
        split_df = set_d_full[set_d_full['Split'] == split_name]
        if len(split_df) > 0:
            print(f"\n  {split_name}:")
            for component in component_keywords:
                count = split_df['UserInput'].str.contains(component, case=False, na=False).sum()
                if count > 0:
                    print(f"    {component}: {count} instances")
        else:
            print(f"\n  {split_name}: (empty)")

    return set_d_full, set_d_train, set_d_test, holdout_info


def save_datasets(set_a, set_b, set_c, set_d_full, set_d_train, set_d_test,
                  holdout_info, output_dir="question_sets"):
    """Save all question sets to CSV files"""
    print("\n" + "=" * 60)
    print("SAVING QUESTION SETS")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save main datasets
    datasets = {
        'question_set_a_technical_accuracy.csv': set_a,
        'question_set_b_linguistic_robustness.csv': set_b,
        'question_set_c_decision_transparency.csv': set_c,
        'question_set_d_cross_component_generalisation_full.csv': set_d_full,
        'question_set_d_train.csv': set_d_train,
        'question_set_d_test.csv': set_d_test,
    }

    for filename, dataset in datasets.items():
        filepath = os.path.join(output_dir, filename)
        dataset.to_csv(filepath, index=False)
        clusters = dataset['Cluster_ID'].nunique() if 'Cluster_ID' in dataset.columns else 'N/A'
        print(f"✅ Saved: {filepath} ({len(dataset)} rows, {clusters} clusters)")

    # Save Set D holdout strategy metadata
    holdout_df = []
    for repair, info in holdout_info.items():
        holdout_df.append({
            'Repair': repair[:100] + ('...' if len(repair) > 100 else ''),
            'Total_Components': info['n_components'],
            'Training_Components': ', '.join(info['training_components']),
            'Held_Out_Components': ', '.join(info['held_out_components']),
            'N_Held_Out': info['n_holdout']
        })

    holdout_metadata = pd.DataFrame(holdout_df)
    holdout_path = os.path.join(output_dir, 'question_set_d_holdout_strategy.csv')
    holdout_metadata.to_csv(holdout_path, index=False)
    print(f"✅ Saved: {holdout_path}")

    # Create summary file
    summary_data = {
        'Question_Set': [
            'Set A: Technical Accuracy',
            'Set B: Linguistic Robustness',
            'Set C: Decision Transparency',
            'Set D: Cross-Component (Train)',
            'Set D: Cross-Component (Test)'
        ],
        'Size': [
            len(set_a),
            len(set_b),
            len(set_c),
            len(set_d_train),
            len(set_d_test)
        ],
        'Clusters': [
            set_a['Cluster_ID'].nunique(),
            set_b['Cluster_ID'].nunique(),
            set_c['Cluster_ID'].nunique(),
            set_d_train['Cluster_ID'].nunique(),
            set_d_test['Cluster_ID'].nunique()
        ],
        'Purpose': [
            'Test basic correctness (deduplicated)',
            'Test consistency across LLaMA-3 generated paraphrases',
            'Test traceability for frequent defects',
            'Training set with seen bridge components',
            'Test set with held-out bridge components'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'question_sets_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved: {summary_path}")

    return output_dir


def main():
    """Main function to orchestrate the dataset creation"""
    print("🏗️ BRIDGE INSPECTION QUESTION SET GENERATOR")
    print("=" * 60)

    # Configuration
    csv_file = "separated_gpt4_response_v2.csv"
    min_frequency_threshold = 3
    num_paraphrases_per_defect = 3  # For Set B
    component_holdout_ratio = 0.3  # For Set D (30% of components held out)

    # Auto-detect CSV files if specified file not found
    if not os.path.exists(csv_file):
        print(f"❌ File '{csv_file}' not found. Looking for CSV files in current directory...")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"Available CSV files: {csv_files}")
            if len(csv_files) == 1:
                csv_file = csv_files[0]
                print(f"✅ Using: {csv_file}")
            else:
                print("Multiple CSV files found. Please specify the correct filename in the script.")
                return
        else:
            print("No CSV files found in current directory.")
            return

    # Step 1: Read dataset
    df = read_dataset(csv_file)
    if df is None:
        return

    # Step 2: Analyze dataset
    response_counts, component_counts = analyze_dataset(df)

    # Step 3: Create question sets according to paper methodology

    # Set A: Deduplicated for technical accuracy
    set_a = create_question_set_a(df)

    # Set B: Paraphrased versions using LLaMA-3 for linguistic robustness
    set_b = create_question_set_b(set_a, num_paraphrases=num_paraphrases_per_defect)

    # Set C: High-frequency defects for decision transparency
    set_c = create_question_set_c(df, min_frequency_threshold)

    # Set D: Cross-component generalization with train/test split
    set_d_full, set_d_train, set_d_test, holdout_info = create_question_set_d(
        df,
        holdout_ratio=component_holdout_ratio
    )

    # Step 4: Save datasets
    output_dir = save_datasets(set_a, set_b, set_c, set_d_full,
                               set_d_train, set_d_test, holdout_info)

    # Step 5: Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"📁 All question sets saved to: ./{output_dir}/")
    print(f"\n📊 Question Set A (Technical Accuracy):")
    print(f"   {len(set_a)} instances, {set_a['Cluster_ID'].nunique()} clusters")
    print(f"   Purpose: Deduplicated defect-repair pairs for accuracy testing")

    print(f"\n📊 Question Set B (Linguistic Robustness):")
    print(f"   {len(set_b)} instances, {set_b['Cluster_ID'].nunique()} clusters")
    print(f"   Purpose: LLaMA-3 generated paraphrases ({num_paraphrases_per_defect} per defect)")

    print(f"\n📊 Question Set C (Decision Transparency):")
    print(f"   {len(set_c)} instances, {set_c['Cluster_ID'].nunique()} clusters")
    print(f"   Purpose: High-frequency repairs (≥{min_frequency_threshold} occurrences)")

    print(f"\n📊 Question Set D (Cross-Component Generalisation):")
    print(f"   Training: {len(set_d_train)} instances (seen components)")
    print(f"   Testing: {len(set_d_test)} instances (held-out components)")
    print(f"   Holdout ratio: {component_holdout_ratio * 100:.0f}% of components per repair")
    print(f"   Purpose: Test knowledge transfer across bridge components")

    print(f"\n🎯 Evaluation Metrics Supported:")
    print(f"   ✓ Correctness (Set A)")
    print(f"   ✓ Robustness (Set B)")
    print(f"   ✓ Traceability (Set C)")
    print(f"   ✓ Generalisation (Set D)")

    print("\n✅ Question set generation completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Review generated datasets in ./question_sets/")
    print("   2. If Set B paraphrases need improvement, ensure LLaMA-3 is accessible via ollama")
    print("   3. Use Set D train/test split for cross-component evaluation")
    print("   4. Refer to question_sets_summary.csv for overview")


if __name__ == "__main__":
    main()