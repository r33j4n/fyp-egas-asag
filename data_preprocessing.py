import pandas as pd
import json
DATASET_SOURCE = 'Data/mohler_dataset_edited.csv'
OUTPUT_PATH = 'Data/mohler_dataset_augmented.csv'
JSON_OUTPUT_PATH = 'Data/mohler_dataset_augmented.json'
JSON_PATH_TO_BE_BEAUTIFIED = 'Data/mohler_expanded_answer.json.txt'

def augment_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    # Remove duplicate rows based on 'question' and 'desired_answer'
    df = df.drop_duplicates(subset=['question', 'desired_answer'])
    augmented_df = df[['id', 'question', 'desired_answer']].copy()
    augmented_df['id'] = range(1, len(augmented_df) + 1)
    augmented_df.to_csv(output_path, index=False)
    print(augmented_df.head())

# New function to convert CSV to JSON
def csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)
    # Save as a list of dicts (array of objects) for valid JSON
    df.to_json(json_path, orient='records', lines=False)
    print(f"JSON file saved to {json_path}")

def beautify_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def modify_and_beautify_json(json_path):
    """
    Reads a JSON file, modifies its content if needed, and beautifies it.
    """
    try:
        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Perform any modifications to the JSON data here
        # For example, ensure all keys are strings or add a new field
        # Example modification: Add a timestamp to each object if it doesn't exist
        from datetime import datetime
        for obj in data:
            if 'timestamp' not in obj:
                obj['timestamp'] = datetime.now().isoformat()

        # Beautify and save the JSON file
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"JSON file at {json_path} has been modified and beautified.")
    except Exception as e:
        print(f"An error occurred while processing the JSON file: {e}")

augment_dataset(DATASET_SOURCE, OUTPUT_PATH)
csv_to_json(OUTPUT_PATH, JSON_OUTPUT_PATH)
beautify_json(JSON_PATH_TO_BE_BEAUTIFIED)
modify_and_beautify_json(JSON_PATH_TO_BE_BEAUTIFIED)
