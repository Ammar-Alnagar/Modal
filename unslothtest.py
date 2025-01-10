import modal
import os 
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")
# Configure the container image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install( "unsloth","peft","xformers<0.0.27" ,"accelerate","bitsandbytes" ,"transformers", "datasets",  "huggingface_hub" , "trl<0.9.0" ).run_commands(
        "pip install --upgrade --no-cache-dir unsloth[colab-new] "
    )
)
app = modal.App("finetune_llm", image=image)
@app.function(gpu="H100" ,timeout=86400)
def main():
    from unsloth import FastLanguageModel 
    import torch
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from transformers import TextStreamer
    
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.3-70B-Instruct", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    )
    tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    pass

    dataset = load_dataset("gghfez/QwQ-LongCoT-130K-cleaned", split = "train")
    dataset = dataset.select(range(10000))
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    unsloth_template = \
        "{{ bos_token }}"\
        "{{ 'You are a helpful assistant to the user\n' }}"\
        "{% for message in messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ '>>> User: ' + message['content'] + '\n' }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}"\
            "{{ '>>> Assistant: ' }}"\
        "{% endif %}"
    unsloth_eos_token = "eos_token"
    
    
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 16,
            warmup_steps = 500,
            num_train_epochs=3,
            # max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    
    
    trainer_stats = trainer.train()
    
    
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    
    tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )

   
    
    
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    
    
    
    if True: model.save_pretrained_merged("Llama3.3-70B-COT", tokenizer, save_method = "merged_16bit",)
    if True: model.push_to_hub_merged("Llama3.3-70B-COT", tokenizer, save_method = "merged_16bit", token = "HF_TOKEN")
    
    
    print("Done")