import nest_asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from pyngrok import ngrok
import torch
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import TrainingArguments, Trainer



# Allow FastAPI to run in Jupyter environment
#nest_asyncio.apply()

# Initialize the FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; replace with specific URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the input structure
class RequestBody(BaseModel):
    input_text: str

# Load the model and tokenizer
model_name = "faq_bert_model"
model2 = BertForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True, num_labels=3)  # Adjust num_labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class FAQDataset(Dataset):
    def __init__(self, questions, answers, labels, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.labels = labels  # Ensure labels are a list of integers (or appropriate class labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Tokenize the question and answer pair
        item = self.tokenizer(
            self.questions[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove the extra batch dimension by squeezing the tensors
        item = {key: value.squeeze(0) for key, value in item.items()}  # Fix: Add squeeze(0)

        # Ensure labels are properly shaped (tensor with correct shape)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Print shapes for debugging
        print(f"input_ids shape: {item['input_ids'].shape}")
        print(f"attention_mask shape: {item['attention_mask'].shape}")
        print(f"labels shape: {label.shape}")

        # Add the label to the output
        item['labels'] = label

        return item
        
trainer = Trainer(
    model=model2,
)
@app.get("/")
def read_root():
    return {"message": "Welcome to the FAQ prediction API! Use the /predict endpoint."}


@app.post("/predict")
def predict(request: RequestBody):
    # Tokenize input text
    my_string = request.input_text
    new_questions = [my_string]
    new_answers = ["You can return items within 30 days."]
    new_dataset = FAQDataset(new_questions, new_answers, [0], tokenizer)

    # Make prediction
    prediction_output = trainer.predict(new_dataset)

    # Access the outputs
    predictions = prediction_output.predictions
    label_ids = prediction_output.label_ids
    metrics = prediction_output.metrics

    # For classification, you may want to convert logits to probabilities using softmax
    import torch.nn.functional as F

    probabilities = F.softmax(torch.tensor(predictions), dim=-1)
    # Get the predicted label
    predicted_label = torch.argmax(probabilities, dim=-1).item()

    # Return the response with prediction probabilities and predicted label
    return {
        "input_text": request.input_text,
        "predicted_label": predicted_label,
        "probabilities": probabilities.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


