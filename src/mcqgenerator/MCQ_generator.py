import os
import json
import pandas as pd
import traceback
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
#IMPORTING NECESSARY PACKAGES FROM LANGCHAIN
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2

# Load envrionment variables from the .env file
load_dotenv()

#Access the environment virables just like you would with os.environment
key =os.getenv("OPENAI_API_KEY")

#Call Load LLM model and call the API
llm=ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo", temperature=0.5)


# Create a prompt template or template for INPUT prompt
TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

#Quiz generation PromptTemplate with Langcnhain
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
    )


#LLMChain to pass the quiz_generation_prompt to the model and get the output into the variable quiz
quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)


#2ND INPUT TEMPLATE
TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

# Quiz evaluation PromptTemplate with LangChain
quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
    )

#LLMChain to pass the quiz_evaluation_prompt to the model and get the output into the variables ["text", "number", "subject", "tone",] to 
# be collected from the user side and "response_json" is going to be the output
review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

# Combining the 2 chains with SequentialChain
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)