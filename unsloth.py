import modal

# Configure the container image
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("unsloth[cu124-torch250]" , "peft","xformers<0.0.27" ,"accelerate","bitsandbytes" ,"transformers", "datasets",  "huggingface_hub" , "trl<0.9.0" )
)
app = modal.App("finetune_llm", image=image)
@app.function()
def main():
    from unsloth import FastLanguageModel 
    
    
    print("hello world")