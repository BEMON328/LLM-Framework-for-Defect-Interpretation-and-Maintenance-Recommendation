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
import pandas as pd
import os
from collections import Counter
import re
import hashlib

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
        print(f"  {i + 1}. Cluster {row['Cluster_ID']}: \"{row['UserInput']}\" → \"{row['Response']}\"")

    return set_a

def create_question_set_b(df):
    """Create Question Set B: Linguistic Robustness Testing (All paraphrases)"""
    print("\n" + "=" * 60)
    print("CREATING QUESTION SET B: Linguistic Robustness Testing")
    print("=" * 60)

    # Use full dataset (all paraphrases)
    set_b = df.copy().reset_index(drop=True)

    print(f"Question Set B: {len(set_b)} instances (full dataset with all paraphrases)")

    # Show example of paraphrases for same cluster
    cluster_counts = df['Cluster_ID'].value_counts()
    example_cluster = cluster_counts[cluster_counts > 1].index[0]  # Get a cluster with multiple instances

    paraphrases = df[df['Cluster_ID'] == example_cluster]
    print(f"\nExample paraphrase group (Cluster {example_cluster}):")
    print(f"Repair: \"{paraphrases.iloc[0]['Response']}\"")
    print(f"Paraphrases ({len(paraphrases)} total):")
    for i, row in paraphrases.head(5).iterrows():
        print(f"  {i + 1}. \"{row['UserInput']}\"")

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

def create_question_set_d(df):
    """Create Question Set D: Cross-Component Generalisation"""
    print("\n" + "=" * 60)
    print("CREATING QUESTION SET D: Cross-Component Generalisation")
    print("=" * 60)

    # Define bridge components
    component_keywords = ['girder', 'flange', 'web', 'angle', 'plate', 'deck', 'bearing', 'joint', 'splice',
                          'stiffener']

    # Find instances that contain component keywords
    component_mask = df['UserInput'].str.contains('|'.join(component_keywords), case=False, na=False)
    component_instances = df[component_mask].copy()

    # Analyze which repairs work across multiple components
    repair_to_components = {}

    for _, row in component_instances.iterrows():
        user_input_lower = row['UserInput'].lower()
        found_components = [comp for comp in component_keywords if comp in user_input_lower]

        if found_components:
            response = row['Response']
            if response not in repair_to_components:
                repair_to_components[response] = set()
            repair_to_components[response].update(found_components)

    # Filter to repairs that work on multiple components
    cross_component_repairs = {repair: components for repair, components in repair_to_components.items()
                               if len(components) > 1}

    # Create Set D with instances suitable for cross-component testing
    cross_component_responses = list(cross_component_repairs.keys())
    set_d = component_instances[component_instances['Response'].isin(cross_component_responses)].reset_index(drop=True)

    print(f"Total instances with component keywords: {len(component_instances)}")
    print(f"Question Set D: {len(set_d)} instances suitable for cross-component testing")
    print(f"Cross-component repairs: {len(cross_component_repairs)}")

    print(f"\nSame repair methods across different components:")
    for i, (repair, components) in enumerate(list(cross_component_repairs.items())[:5], 1):
        components_str = ', '.join(sorted(components))
        print(f"  {i}. Cluster {df[df['Response'] == repair].iloc[0]['Cluster_ID']}: \"{repair[:60]}{'...' if len(repair) > 60 else ''}\"")
        print(f"     Components: {components_str}")

    # Component distribution in Set D
    component_distribution = {}
    for component in component_keywords:
        count = set_d['UserInput'].str.contains(component, case=False, na=False).sum()
        if count > 0:
            component_distribution[component] = count

    print(f"\nComponent distribution in Set D:")
    for component, count in sorted(component_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {component}: {count} instances")

    return set_d

def save_datasets(set_a, set_b, set_c, set_d, output_dir="question_sets"):
    """Save all question sets to CSV files"""
    print("\n" + "=" * 60)
    print("SAVING QUESTION SETS")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'question_set_a_technical_accuracy.csv': set_a,
        'question_set_b_linguistic_robustness.csv': set_b,
        'question_set_c_decision_transparency.csv': set_c,
        'question_set_d_cross_component_generalisation.csv': set_d
    }

    for filename, dataset in datasets.items():
        filepath = os.path.join(output_dir, filename)
        dataset.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath} ({len(dataset)} rows, {dataset['Cluster_ID'].nunique()} clusters)")

    # Create summary file
    summary_data = {
        'Question_Set': ['Set A: Technical Accuracy', 'Set B: Linguistic Robustness',
                         'Set C: Decision Transparency', 'Set D: Cross-Component Generalisation'],
        'Size': [len(set_a), len(set_b), len(set_c), len(set_d)],
        'Clusters': [set_a['Cluster_ID'].nunique(), set_b['Cluster_ID'].nunique(),
                     set_c['Cluster_ID'].nunique(), set_d['Cluster_ID'].nunique()],
        'Purpose': [
            'Test basic correctness (deduplicated)',
            'Test consistency across paraphrases',
            'Test traceability for frequent defects',
            'Test knowledge transfer across components'
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

    # Configuration - Update this filename to match your CSV file
    csv_file = "separated_llama3_response_counted_v3.csv"  # Change this to your actual filename
    min_frequency_threshold = 3  # Minimum frequency for Set C

    # Alternative: Auto-detect CSV files in directory
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

    # Step 3: Create question sets
    set_a = create_question_set_a(df)
    set_b = create_question_set_b(df)
    set_c = create_question_set_c(df, min_frequency_threshold)
    set_d = create_question_set_d(df)

    # Step 4: Save datasets
    output_dir = save_datasets(set_a, set_b, set_c, set_d)

    # Step 5: Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"📁 All question sets saved to: ./{output_dir}/")
    print(f"📊 Question Set A (Technical Accuracy): {len(set_a)} instances, {set_a['Cluster_ID'].nunique()} clusters")
    print(f"📊 Question Set B (Linguistic Robustness): {len(set_b)} instances, {set_b['Cluster_ID'].nunique()} clusters")
    print(f"📊 Question Set C (Decision Transparency): {len(set_c)} instances, {set_c['Cluster_ID'].nunique()} clusters")
    print(f"📊 Question Set D (Cross-Component Generalisation): {len(set_d)} instances, {set_d['Cluster_ID'].nunique()} clusters")
    print(f"🎯 Ready for evaluation against: Correctness, Robustness, Traceability, Generalisation")
    print("\n✅ Question set generation completed successfully!")

if __name__ == "__main__":
    main()