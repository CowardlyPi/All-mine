import re

# Read the main.py file
with open('main.py', 'r') as file:
    content = file.read()

# Find and replace the OpenAI client initialization
pattern = r"self\.openai_client = OpenAI\(\s*api_key=openai_api_key,\s*organization=openai_org_id,\s*project=openai_project_id\s*\)"
replacement = "self.openai_client = OpenAI(\n            api_key=openai_api_key,\n            organization=openai_org_id\n        )"

# Apply the replacement
modified_content = re.sub(pattern, replacement, content)

# Write the modified content back to the file
with open('main.py', 'w') as file:
    file.write(modified_content)

print("Successfully patched main.py to fix OpenAI client initialization!")
