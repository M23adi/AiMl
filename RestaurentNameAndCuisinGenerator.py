import os
os.environ['GOOGLE_API_KEY']='APIkey'
from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7)
name=llm.invoke("i want to open a restaurent for indian food. Suggest a fancy name for this.")
print(name)
