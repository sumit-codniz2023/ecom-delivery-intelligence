import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# 1. Load your datasets
print("Loading project data and models...")
modelling_df = pd.read_csv("data/modelling_table.csv")
new_shipments_df = pd.read_csv("data/new_shipments.csv")

# 2. Initialize the Groq LLM
llm = ChatGroq(
    temperature=0, 
    model_name="openai/gpt-oss-20b"
)

# 3. Custom Robust Query Assistant (Bypasses fragile agent tool wrappers)
def ask_delivery_assistant(query: str):
    print(f"\nUser Query: {query}")
    
    prompt = f"""
You are an expert data analyst assistant. You have access to two pandas DataFrames in Python:
1. `modelling_df` (shape: {modelling_df.shape}): Historical shipment modeling table. Columns: {list(modelling_df.columns)}
2. `new_shipments_df` (shape: {new_shipments_df.shape}): Incoming future shipments table. Columns: {list(new_shipments_df.columns)}

User Question: {query}

Write a Python code snippet using pandas to answer the user question. 
You must store the final answer, summary, or dataframe result in a variable named `result`.
Provide ONLY the executable Python code inside a markdown code block (```python ... ```). Do not include any extra conversational text outside the code block.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Extract python code block safely
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            code = content.split("```")[1].split("```")[0].strip()
        else:
            code = content.strip()
            
        print(f"\n[Generated Code]:\n{code}\n")
        
        # Execute the generated code in a safe local namespace
        local_vars = {
            "modelling_df": modelling_df,
            "new_shipments_df": new_shipments_df,
            "result": None
        }
        
        exec(code, {}, local_vars)
        
        result = local_vars.get("result")
        print(f"AI Assistant:\n{result}\n")
        return result
        
    except Exception as e:
        print(f"Error executing query: {e}")

if __name__ == "__main__":
    print("\n--- Capability Foundry Delivery Intelligence Agent Initialized ---")
    print("Type your questions below (type 'exit' to quit):\n")
    
    # Test initial query
    try:
        ask_delivery_assistant("How many incoming future shipments do we have in new_shipments.csv, and what is the breakdown by service type?")
    except Exception as e:
        print(f"Initial query error: {e}")

    # Interactive loop for live prompting in CMD
    while True:
        user_input = input("Ask about shipments: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue
        try:
            ask_delivery_assistant(user_input)
        except Exception as e:
            print(f"Error: {e}")