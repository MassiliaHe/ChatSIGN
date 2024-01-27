from openai import OpenAI

class ChatGPT:
    def __init__(self, api_key, model="gpt-3.5-turbo-1106"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation = [{"role": "system", "content": "The conversation with ChatGPT has started in English."}]

    def __call__(self, user_input):
        self.conversation.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation
        )

        assistant_response = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_response})
        return assistant_response