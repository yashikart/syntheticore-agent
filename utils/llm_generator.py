import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def generate_llm_summary(prompt, system_role="You are a helpful assistant.", api_key=None, temperature=0.7, max_tokens=150):
    try:
        api_key = "open_api_key"
        if not api_key:
            raise ValueError("❌ OpenAI API key not set!")

        # Create LangChain model instance
        model = ChatOpenAI(
            temperature=temperature,
            api_key=api_key,
            model_name="gpt-3.5-turbo"
        )

        # Prepare prompt messages
        messages = [
            SystemMessage(content=system_role),
            HumanMessage(content=prompt)
        ]

        # Invoke the model with config (max_tokens)
        response = model.invoke(messages, config={"max_tokens": max_tokens})

        return response.content.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"

