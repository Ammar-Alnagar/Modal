import modal
import os 
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

# Configure the container image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("transformers", "datasets", "accelerate", "peft", "huggingface_hub" , "sentencepiece","bitsandbytes")
)

app = modal.App("finetune_llm", image=image)


@app.function(gpu="H100:3" ,timeout=86400)  # Adjust GPU type and timeout as needed
def main():
    from huggingface_hub import Repository
    from transformers import (
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
        AutoTokenizer
    )
    from datasets import load_dataset
    import torch

    # Configurations
    MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct"
    DATASET_NAME = "qingy2024/FineQwQ-142k"
    OUTPUT_DIR = "./llama_finetuned"
    HUGGINGFACE_REPO_NAME = "Daemontatox/Llama-3.3-70B-Instruct-ZWZ"  # Change to your desired repo
    HUGGINGFACE_TOKEN = "HF_TOKEN"  # Replace with your Hugging Face API token
    MAX_LENGTH = 512
    BATCH_SIZE = 8

    # Load dataset
    dataset = load_dataset(DATASET_NAME)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_data(example):
        prompt = example["prompt"]
        response = example["response"]
        source = f"### Prompt:\n{prompt}\n### Response:\n{response}"
        tokenized = tokenizer(
            source, truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_data, batched=True, remove_columns=["prompt", "response","source"]
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=MODEL_NAME, padding=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=500,
        logging_strategy="steps",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=HUGGINGFACE_REPO_NAME,
        hub_token=HUGGINGFACE_TOKEN,
    )
    
    
    

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["50K"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Fine-tuning
    trainer.train()

    # Save the model locally and upload to Hugging Face
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    repo = Repository(local_dir=OUTPUT_DIR, token=HUGGINGFACE_TOKEN)
    repo.push_to_hub()

    print(f"Fine-tuned model uploaded to Hugging Face: {HUGGINGFACE_REPO_NAME}")
