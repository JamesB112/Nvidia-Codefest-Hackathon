import pandas as pd
import asyncio
from openai import AsyncOpenAI
import logging
from datetime import datetime
import json
from pydantic import BaseModel, ValidationError
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Excel source file to load
source_file = 'Complaints Register Aug23-Jul24.xlsx' #'cc_test20.xlsx' #Complaints Register 2024-05-Jul24.xlsx'

# Excel output file to save results
output_file = "cc_test_result2024-09-02_all.xlsx"

prompt = """
## Instructions
1. Customer Complaint: Extract the customer complaint from the description exactly as it was stated by the customer. Restate only the complaint without any additional text or formatting. Do not include the customer name or contact details.
2. Complaint Category: Categorise the extracted Customer Complaint into one, and only one, of the following categories: "Billing", "Billing - High Usage", "Billing - Charges", "Billing - Meter Readings", "Billing - Did Not Receive Bill", "Account Payments & Collections - Payments", "Account Payments & Collections - Debt Recovery Action", "Account Payments & Collections - Unable to Pay", "Customer Details - Move In / Move Out", 
     "Customer Details - Account/Access", "Customer Details - Update Customer/Person", "Customer Details - Concession", "Customer Details - Returned Account", "Faults and Emergencies - Water", 
     "Faults and Emergencies - Sewer", "Construction - Noise or Vibration", "Construction - Traffic Management", "Construction - Timing", "Construction - Disruption to Water Supply", 
     "Construction - Damage to Property", "Construction - Site Clean Up / Stock Pile Areas", "Customer Service", "Sales & Service - Service Issues", "Sales & Service - Campaigns & Programs", 
     "Property - Meter Information", "Meter - Meter Information", "Transfer - Call transfer", take a breath and check **only** the most appropriate category has been chosen.
3. Complaint Topics: Extract a comma separated list of topics mentioned in the customer complaint text enclosed in [] brackets.
4. Resolution: Extract the resolution details, if and only if, the description includes details as to actions we have taken to resolve the complaint, leave blank otherwise.
5. Resolution Category: Categorise the resolution action taken e.g. "Information Provided", "Leak Allowance/Unexplained Usage", "Damage Claim", "Further Action Required", "Account Balance/Due Date", "Tenancy Information", "Billing Adjustment", "Plumbing Claim", "Telephone Issue", "Organised Refund" or otherwise if more appropriate.
6. Generate points 1-5 in a JSON format, ensure the JSON generated is valid and consistent, output well formatted JSON without comments. Respond with only the JSON, do not output any other comments or details other than the JSON.

## Description
{0}

## JSON Output
"""

# Define the JSON schema for structured output
json_schema = {
    "type": "object",
    "properties": {
        "customer_complaint": {"type": "string"},
        "complaint_category": {"type": "string"},
        "complaint_topics": {"type": "array", "items": {"type": "string"}},
        "resolution": {"type": "string"},
        "resolution_category": {"type": "string"}
    },
    "required": ["customer_complaint", "complaint_category", "complaint_topics"]
}

# Setup logging
log_filename = datetime.now().strftime("script_execution_%Y%m%d_%H%M%S.log")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Async function to create a chat completion with a dynamic user prompt
async def create_chat_completion(user_input, system_message, model_name, retries=6):
    for attempt in range(retries):
        try:
            logging.info(f"Model used: {model_name}, Attempt: {attempt + 1}")
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt.format(user_input)},
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "complaint_response",
                        "strict": "true",
                        "schema": json_schema
                    }
                }
            )

            # Try to parse the response as JSON
            response_json = json.loads(response.choices[0].message.content.strip())
            return response_json

        except (json.JSONDecodeError) as e:
            logging.error(f"Invalid JSON or Validation error: {str(e)}")
            if attempt + 1 == retries:
                return None  # Return None after retries are exhausted
            await asyncio.sleep(2 ** attempt)  # Exponential backoff before retrying
        except Exception as e:
            logging.error(f"Error in create_chat_completion: {str(e)}")
            if attempt + 1 == retries:
                return None
            await asyncio.sleep(2 ** attempt)  # Exponential backoff before retrying

# Async function to process each row asynchronously
async def process_row(semaphore, row, system_message, model_name):
    async with semaphore:
        user_input = row["Description"]
        response_obj = await create_chat_completion(user_input, system_message, model_name)
    return response_obj

# Function to generate embeddings
async def get_embedding(text, model_name="second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q8_0.gguf"):
    response = await client.embeddings.create(input=text, model=model_name)
    return response.data[0].embedding

# Function to compute cosine similarity
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Main async function to handle the spreadsheet processing
async def gather():
    semaphore = asyncio.Semaphore(3)

    df = pd.read_excel(source_file)

    models = [
        "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
        "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf:2",
        "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf:3"
    ]

    tasks_get_responses = [
        process_row(semaphore, row, "", models[i % len(models)])
        for i, row in df.iterrows()
    ]

    responses = await asyncio.gather(*tasks_get_responses)

    # Filter out None responses and create a DataFrame from the JSON responses
    valid_responses = [res for res in responses if res is not None]
    response_df = pd.json_normalize(valid_responses)

    # Merge the response DataFrame with the original DataFrame
    df = pd.concat([df.reset_index(drop=True), response_df.reset_index(drop=True)], axis=1)

        # Initialize empty lists to store embeddings and similarity scores
    complaint_category_embeddings = []
    original_text_embeddings = []
    similarity_scores = []
    customer_complaint_embeddings = []
    complaint_topics_embeddings = []

    # Generate embeddings and compute similarity
    for row in df.itertuples():
        # Await the embeddings since they are coroutines
        complaint_category_embedding = await get_embedding(row.complaint_category.strip())

        if str(row.What).strip() in (None, "", "0"):
            if str(row.Subject).strip() not in (None, "", "0"):
                logging.info("Original text embedding: Subject")
                original_text_embedding = await get_embedding((row.Subject).strip())
            else:
                logging.info("Original text embedding: None")
                original_text_embedding = None
        else:
            logging.info("Original text embedding: What + Why")
            original_text_embedding = await get_embedding((str(row.What) + ' - ' + str(row.Why)).strip())  

        complaint_category_embeddings.append(complaint_category_embedding)
        original_text_embeddings.append(original_text_embedding)
        
        similarity_score = compute_cosine_similarity(complaint_category_embedding, original_text_embedding)
        similarity_scores.append(similarity_score)

        customer_complaint_embedding = await get_embedding(row.customer_complaint.strip())
        customer_complaint_embeddings.append(customer_complaint_embedding)

        complaint_topics_embedding = await get_embedding(''.join(row.complaint_topics).strip())
        complaint_topics_embeddings.append(complaint_topics_embedding)
        

    # Add the embeddings and similarity scores to the DataFrame
    df["Complaint_Category_Embedding"] = complaint_category_embeddings
    df["Original_Text_Embedding"] = original_text_embeddings
    df["Similarity_Score"] = similarity_scores
    df["Customer_Complaint_Embedding"] = customer_complaint_embeddings
    df["Complaint_Topic_Embedding"] = complaint_topics_embeddings

    # Convert embeddings to string for Excel output
    df["Complaint_Category_Embedding"] = [json.dumps(embedding) for embedding in complaint_category_embeddings]
    df["Original_Text_Embedding"] = [json.dumps(embedding) for embedding in original_text_embeddings]
    df["Customer_Complaint_Embedding"] = [json.dumps(embedding) for embedding in customer_complaint_embeddings]
    df["Complaint_Topic_Embedding"] = [json.dumps(embedding) for embedding in complaint_topics_embeddings]

    # Save the updated DataFrame back to Excel
    df.to_excel(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

# Setup base URL and API key for AsyncOpenAI client
client = AsyncOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not_needed")

# Fix for Windows event loop issue
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the async event loop
    asyncio.run(gather())