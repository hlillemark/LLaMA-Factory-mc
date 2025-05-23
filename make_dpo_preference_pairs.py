from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

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

def generate_with_template_format(prompts, model_name="gpt2"):
    """Generate text using the instruction/input/output template format"""
    
    generator = pipeline('text-generation', 
                        model=model_name, 
                        tokenizer=model_name,
                        device=0 if torch.cuda.is_available() else -1)
    
    results = []
    
    for prompt_data in prompts:
        # Format to template structure
        formatted_template = format_prompt_to_template(prompt_data)
        
        # Create the prompt string for generation
        prompt_string = create_formatted_prompt_string(formatted_template, include_output=False)
        
        # Generate response
        generated = generator(prompt_string, 
                            max_length=300,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            truncation=True,
                            pad_token_id=generator.tokenizer.eos_token_id)
        
        # Extract just the generated output (remove the input prompt)
        full_output = generated[0]['generated_text']
        output_only = full_output[len(prompt_string):].strip()
        
        # Create result with template format
        result = {
            "instruction": formatted_template['instruction'],
            "input": formatted_template['input'],
            "output": output_only,
            "original_prompt": prompt_data,
            "formatted_prompt_string": prompt_string,
            "full_generated_text": full_output
        }
        
        results.append(result)
    
    return results

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

def generate_with_minecraft_format(prompts, model_name="microsoft/DialoGPT-small"):
    """Specialized function for handling Minecraft bot conversation format"""
    
    generator = pipeline('text-generation', 
                        model=model_name, 
                        tokenizer=model_name,
                        device=0 if torch.cuda.is_available() else -1)
    
    results = []
    
    for prompt_data in prompts:
        # Handle the specific format from your example
        if isinstance(prompt_data, dict) and 'instruction' in prompt_data and 'input' in prompt_data:
            instruction = prompt_data['instruction']
            
            # Process input (might be conversation format)
            if isinstance(prompt_data['input'], str):
                try:
                    import json
                    # Try to parse as JSON if it's a string representation
                    input_data = json.loads(prompt_data['input'])
                    formatted_input = process_conversation_input(input_data)
                except:
                    # If not JSON, use as is
                    formatted_input = prompt_data['input']
            else:
                formatted_input = process_conversation_input(prompt_data['input'])
            
            # Create the full prompt
            full_prompt = f"{instruction}\n\nConversation History:\n{formatted_input}\n\nResponse:"
            
        else:
            # Use the standard template formatting
            formatted_template = format_prompt_to_template(prompt_data)
            full_prompt = create_formatted_prompt_string(formatted_template)
        
        # Generate response
        generated = generator(full_prompt, 
                            max_length=400,  # Longer for complex instructions
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            truncation=True,
                            pad_token_id=generator.tokenizer.eos_token_id)
        
        # Extract just the response
        full_output = generated[0]['generated_text']
        response_only = full_output[len(full_prompt):].strip()
        
        # Format result to match your template
        result = {
            "instruction": prompt_data.get('instruction', ''),
            "input": prompt_data.get('input', ''),
            "output": response_only,
            "full_prompt": full_prompt,
            "full_generated": full_output
        }
        
        results.append(result)
    
    return results

# Example usage with your format:
minecraft_prompts = [
    {
        "instruction": "You are a task-focused Minecraft bot named Jill_0. You have to collaborate with other agents in the world to complete the current task...",
        "input": """[
  {
    "role": "user",
    "content": "Andy_0: (FROM OTHER BOT)"
  },
  {
    "role": "assistant", 
    "content": "!attack(\\"rabbit\\", 1)"
  },
  {
    "role": "system",
    "content": "Code output:\\nFound sugar_cane at (1135, -60, -2292).\\nYou have reached at 1135, -60, -2292.\\n"
  }
]""",
        "output": "!collectBlocks(\"sugar_cane\", 3)"
    }
]

# Generate responses
results = generate_with_minecraft_format(minecraft_prompts)

# Print results in your template format
for result in results:
    print("Generated result:")
    print(f"Instruction: {result['instruction'][:100]}...")  # Truncated for display
    print(f"Input: {result['input'][:100]}...")
    print(f"Output: {result['output']}")
    print("-" * 80)