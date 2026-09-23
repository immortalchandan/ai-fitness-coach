from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class FitnessCoachLLM:
    def __init__(self, api_key):
        # Updated to the currently active Gemini 2.5 Flash model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key, 
            temperature=0.7
        )
        self.output_parser = StrOutputParser()
        
        # Define the AI persona
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an elite AI personal trainer. Provide concise, highly technical, and actionable fitness advice based on biomechanics and modern exercise science. Keep responses under 3 paragraphs."),
            ("user", "{user_query}")
        ])
        
        self.chain = self.prompt | self.llm | self.output_parser

    def get_advice(self, user_query):
        try:
            return self.chain.invoke({"user_query": user_query})
        except Exception as e:
            return f"Error connecting to AI Coach: {str(e)}"