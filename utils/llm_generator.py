import openai  

def generate_llm_summary(prompt, system_role="You are a helpful assistant.", api_key=None, temperature=0.7, max_tokens=150):
    try:
        client = openai(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"
