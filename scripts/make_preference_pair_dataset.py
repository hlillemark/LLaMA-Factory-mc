from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
from datasets import Dataset, load_dataset
import pandas as pd
import argparse

def format_prompt_to_template(prompt_data):
    """Format prompt data to match the instruction/input/output template"""
    
    if isinstance(prompt_data, dict):
        # If it already has the correct format
        if 'instruction' in prompt_data and 'input' in prompt_data:
            return prompt_data
        
        # Convert other formats to the template
        elif 'prompt' in prompt_data and 'input' in prompt_data:
            return {
                "instruction": prompt_data['prompt'],
                "input": prompt_data['input'],
                "output": prompt_data.get('output', '')
            }
        
        # Handle case where there's just instruction
        elif 'instruction' in prompt_data:
            return {
                "instruction": prompt_data['instruction'],
                "input": prompt_data.get('input', ''),
                "output": prompt_data.get('output', '')
            }
        
        # Fallback for other dictionary formats
        else:
            # Try to guess the instruction from common keys
            instruction = (prompt_data.get('question') or 
                          prompt_data.get('task') or 
                          prompt_data.get('prompt') or 
                          'Complete the following task:')
            
            input_text = (prompt_data.get('input') or 
                         prompt_data.get('context') or 
                         prompt_data.get('text') or '')
            
            return {
                "instruction": instruction,
                "input": input_text,
                "output": prompt_data.get('output', '')
            }
    
    # If it's a string, treat it as instruction
    elif isinstance(prompt_data, str):
        return {
            "instruction": prompt_data,
            "input": "",
            "output": ""
        }
    
    else:
        return {
            "instruction": str(prompt_data),
            "input": "",
            "output": ""
        }

def create_formatted_prompt_string(formatted_data, include_output=False):
    """Convert the formatted data into a string for the model"""
    
    prompt_parts = []
    
    # Add instruction
    if formatted_data.get('instruction'):
        prompt_parts.append(f"Instruction: {formatted_data['instruction']}")
    
    # Add input if it exists and is not empty
    if formatted_data.get('input') and formatted_data['input'].strip():
        prompt_parts.append(f"Input: {formatted_data['input']}")
    
    # Add output section
    if include_output and formatted_data.get('output'):
        prompt_parts.append(f"Output: {formatted_data['output']}")
    else:
        prompt_parts.append("Output:")
    
    return '\n\n'.join(prompt_parts)

def process_conversation_input(conversation_data):
    """Special handler for conversation-style inputs like in your example"""
    
    if isinstance(conversation_data, list):
        # Extract the conversation context
        conversation_text = ""
        for turn in conversation_data:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role and content:
                conversation_text += f"{role}: {content}\n"
        
        return conversation_text.strip()
    
    return str(conversation_data)

def prepare_dataset(data_file_path):
    """Load and prepare dataset from JSON file"""
    
    # Load JSON data
    with open(data_file_path) as f:
        # take only the first 1000 samples for testing
        json_data = json.load(f)[:1000]
    
    # Convert to Dataset
    if isinstance(json_data, list):
        dataset = Dataset.from_list(json_data)
    else:
        # If it's a dict, assume it's already in dataset format
        dataset = Dataset.from_dict(json_data)
    
    return dataset

def format_prompt_for_generation(example):
    """Function to apply to each example in the dataset"""
    
    # Handle the specific format from your example
    if 'instruction' in example and 'input' in example:
        instruction = example['instruction']
        
        # Process input (might be conversation format)
        input_data = example['input']
        if isinstance(input_data, str):
            try:
                # Try to parse as JSON if it's a string representation
                parsed_input = json.loads(input_data)
                formatted_input = process_conversation_input(parsed_input)
            except:
                # If not JSON, use as is
                formatted_input = input_data
        else:
            formatted_input = process_conversation_input(input_data)
        
        # Create the full prompt
        full_prompt = f"{instruction}\n\nConversation History:\n{formatted_input}\n\nResponse:"
        
    else:
        # Use the standard template formatting
        formatted_template = format_prompt_to_template(example)
        full_prompt = create_formatted_prompt_string(formatted_template)
    
    # Add the formatted prompt to the example
    example['formatted_prompt'] = full_prompt
    return example

def generate_batch_responses(examples, generator, batch_size=8):
    """Generate responses for a batch of examples"""
    
    prompts = examples['formatted_prompt']
    
    # Generate responses in batch
    generated_outputs = generator(
        prompts,
        max_length=8000,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id,
        batch_size=batch_size
    )
    
    # Extract just the responses (remove the input prompts)
    responses = []
    for i, output in enumerate(generated_outputs):
        if isinstance(output, list):
            output = output[0]  # Take first if multiple returns
        
        full_output = output['generated_text']
        prompt_length = len(prompts[i])
        response_only = full_output[prompt_length:].strip()
        responses.append(response_only)
    
    examples['dispreferred'] = responses
    return examples

def generate_with_dataset_format(data_file_path, model_name="microsoft/DialoGPT-small", batch_size=8, num_proc=4):
    """Generate responses using Dataset for efficient processing"""
    
    print("Loading and preparing dataset...")
    # Load dataset
    dataset = prepare_dataset(data_file_path)
    
    print(f"Dataset loaded with {len(dataset)} examples")
    print("Sample data:", dataset[0])
    
    # Format prompts for generation
    print("Formatting prompts...")
    dataset = dataset.map(
        format_prompt_for_generation, 
        num_proc=num_proc,
        desc="Formatting prompts"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

    num_gpus = torch.cuda.device_count()

    # Configuration for multi-GPU setup
    device_config = {
        "device_map": "auto" if num_gpus > 1 else (0 if num_gpus == 1 else -1),
        "torch_dtype": torch.float16 if num_gpus > 0 else torch.float32
    }

    # Add model_kwargs for multi-GPU scenarios
    if num_gpus > 1:
        device_config["model_kwargs"] = {
            "max_memory": {i: "auto" for i in range(min(num_gpus, 8))}
        }

    generator = pipeline(
        'text-generation', 
        model=model_name, 
        tokenizer=model_name,
        **device_config
    )
    
    # Initialize model and tokenizer
    print(f"Loading model: {model_name}")
    generator = pipeline(
        'text-generation', 
        model=model_name, 
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Generate responses in batches
    print("Generating responses...")
    def generate_batch(examples):
        return generate_batch_responses(examples, generator, batch_size)
    
    # Process in batches
    dataset_with_responses = dataset.map(
        generate_batch,
        batched=True,
        batch_size=batch_size,
        desc="Generating responses"
    )
    
    # Add preferred responses and calculate metrics
    def add_preferred_and_metrics(example):
        example['preferred'] = example.get('output', '')
        example['same_response'] = 1 if example['dispreferred'] == example['preferred'] else 0
        return example
    
    dataset_with_responses = dataset_with_responses.map(
        add_preferred_and_metrics,
        desc="Adding preferred responses"
    )
    
    # Calculate same response count
    same_response_count = sum(dataset_with_responses['same_response'])
    total_count = len(dataset_with_responses)
    
    print(f"Same responses: {same_response_count}/{total_count} ({same_response_count/total_count*100:.2f}%)")
    
    return dataset_with_responses

def save_dataset_results(dataset, save_path, format='json'):
    """Save dataset results to file"""
    
    if format == 'json':
        # Convert to list of dictionaries and save as JSON
        data_list = []
        for example in dataset:
            # Create result dictionary
            result = {
                "instruction": example.get('instruction', ''),
                "input": example.get('input', ''),
                "dispreferred": example.get('dispreferred', ''),
                "preferred": example.get('preferred', ''),
                "formatted_prompt": example.get('formatted_prompt', '')
            }
            data_list.append(result)
        
        with open(f"{save_path}", 'w') as f:
            json.dump(data_list, f, indent=2)
            
    elif format == 'parquet':
        # Save as parquet for efficiency
        dataset.save_to_disk(save_path)
        
    elif format == 'csv':
        # Convert to pandas and save as CSV
        df = dataset.to_pandas()
        df.to_csv(f"{save_path}.csv", index=False)
    
    print(f"Results saved to {save_path}")

# Example usage
if __name__ == "__main__":

    def parse_arguments():
        parser = argparse.ArgumentParser(description='Generate preference pair dataset')
        parser.add_argument('--data_file_path', type=str, default="data/custom/cooking_sft_success_new_mem.json", help='Path to the input data file')
        parser.add_argument('--save_path', type=str, default="data/custom/dpo_pairs.json", help='Path to save the results file')
        parser.add_argument('--model_name', type=str, default="izzcw/large_cooking_sft_fail", help='Name of the model to use')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
        parser.add_argument('--num_proc', type=int, default=2, help='Number of processes for generation')
        
        return parser.parse_args()

    if __name__ == "__main__":
        args = parse_arguments()
        data_file_path = args.data_file_path
        save_path = args.save_path
        model_name = args.model_name
        batch_size = args.batch_size
        num_proc = args.num_proc

        # Generate responses using dataset
        results_dataset = generate_with_dataset_format(
            data_file_path=data_file_path,
            model_name=model_name,
            batch_size=batch_size,
            num_proc=num_proc
        )
        
        # Save results
        save_dataset_results(results_dataset, save_path, format='json')
        
        # Print some sample results
        print("\nSample results:")
        for i in range(min(3, len(results_dataset))):
            result = results_dataset[i]
            print(f"\nExample {i+1}:")
            print(f"Instruction: {result['instruction'][:100]}...")
            print(f"Input: {str(result['input'])[:100]}...")
            print(f"Dispreferred: {result['dispreferred'][:100]}...")
            print(f"Preferred: {result['preferred'][:100]}...")
            print("-" * 80)
    
    # Generate responses using dataset
    results_dataset = generate_with_dataset_format(
        data_file_path=data_file_path,
        model_name=model_name,
        batch_size=4,  # Adjust based on your GPU memory
        num_proc=2     # Adjust based on your CPU cores
    )
    
    # Save results
    save_dataset_results(results_dataset, save_path, format='json')
    
    # Print some sample results
    print("\nSample results:")
    for i in range(min(3, len(results_dataset))):
        result = results_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"Instruction: {result['instruction'][:100]}...")
        print(f"Input: {str(result['input'])[:100]}...")
        print(f"Dispreferred: {result['dispreferred'][:100]}...")
        print(f"Preferred: {result['preferred'][:100]}...")
        print("-" * 80)
