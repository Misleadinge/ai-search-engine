import pandas as pd
import requests

# Load the Excel file with test data
file_path = 'Book3.xlsx' 
data = pd.read_excel(file_path)

# URL of your Flask app's search endpoint
url = "http://localhost:5002/search" 

# Initialize counters for test results
total_tests = len(data)
passed_tests = 0
failed_tests = 0

# Loop through each row in the Excel file and send requests to /search
for index, row in data.iterrows():
    example_question = row['Örnek soru']  # Example Question
    target_catalog_item = row['Ulaşılması hedeflenen katalog öğesi']  # Target Catalog Item

    # Prepare the payload for the POST request
    payload = {
        "input": example_question,
        "prompt_type": "keyword_generation"
    }

    # Send the POST request to the /search endpoint
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        # Extract the GPT response
        gpt_response = response.json()['response']
        
        # Collect all suggested descriptions for comparison
        suggested_descriptions = []

        for line in gpt_response:
            if line.startswith("En Uygun DESCRIPTION:"):
                # Extract the first suggested description
                first_suggested_description = line.split(":")[1].strip().lower()
                suggested_descriptions.append(first_suggested_description)
            elif line.startswith("   - Alternatif"):
                # Extract alternative descriptions
                alternative_description = line.split(":")[1].strip().lower()
                suggested_descriptions.append(alternative_description)

        # Perform a case-insensitive comparison
        target_catalog_item_lower = target_catalog_item.lower()
        if target_catalog_item_lower in suggested_descriptions:
            print(f"Test {index + 1} PASSED")
            passed_tests += 1
        else:
            print(f"Test {index + 1} FAILED - Expected: {target_catalog_item}, Got: {suggested_descriptions}")
            failed_tests += 1
    else:
        print(f"Test {index + 1} FAILED - HTTP Status Code: {response.status_code}")
        failed_tests += 1

# Print summary of test results
print(f"\nTotal Tests: {total_tests}")
print(f"Passed Tests: {passed_tests}")
print(f"Failed Tests: {failed_tests}")
