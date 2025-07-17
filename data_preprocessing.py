import pandas as pd
import json
DATASET_SOURCE = 'Data/mohler_dataset_edited.csv'
OUTPUT_PATH = 'Data/mohler_dataset_augmented.csv'
JSON_OUTPUT_PATH = 'Data/mohler_dataset_augmented.json'
JSON_PATH_TO_BE_BEAUTIFIED = 'Data/mohler_expanded_answer.json'

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

def merge_questions_with_topics(question_group_path, listed_topic_path, expanded_answer_path, output_path):
    # Load question groups
    with open(question_group_path, 'r') as file:
        question_groups = json.load(file)

    # Load listed topics
    with open(listed_topic_path, 'r') as file:
        listed_topics = json.load(file)

    # Load expanded answers
    with open(expanded_answer_path, 'r') as file:
        expanded_answers = json.load(file)

    # Create a mapping of question IDs to topics from listed topics
    id_to_topics = {item['id']: item['topics'] for item in listed_topics}

    # Merge topics into expanded answers
    for answer in expanded_answers:
        qid = answer['id']
        answer['topics'] = id_to_topics.get(qid, [])

    # Save the merged data
    with open(output_path, 'w') as file:
        json.dump(expanded_answers, file, indent=4)

    print(f"Merged data saved to {output_path}")

augment_dataset(DATASET_SOURCE, OUTPUT_PATH)
csv_to_json(OUTPUT_PATH, JSON_OUTPUT_PATH)
beautify_json(JSON_PATH_TO_BE_BEAUTIFIED)
modify_and_beautify_json(JSON_PATH_TO_BE_BEAUTIFIED)

merge_questions_with_topics(
    'Data/question_group.json',
    'Data/listed_topic.json',
    'Data/mohler_expanded_answer.json',
    'Data/mohler_merged_questions.json'
)

