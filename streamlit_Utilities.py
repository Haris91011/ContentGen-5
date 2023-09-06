import openai
import requests
from PIL import Image
from io import BytesIO
# import api_key
import re
# openai.api_key = api_key.OPENAI_API_KEY
import ast
import os
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import load_tools
import tempfile
import io
import streamlit as st
openapi_key = st.secrets["open_ai_key"]
openai.api_key = openapi_key
# OpenAI = openapi_key
segmind_API=st.secrets["api_key"]
serpapi_api_key = st.secrets["serp_api_key"]
# SerpAPIWrapper.api_key=serpapi_api_key
# print(serpapi_api_key)
#function for blog structure
def generate_Blog_Structure(Title):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a {Title} and generate the structure of a blog post depending upon the Title.First analyze the (Language) of {Title} and responsed must be in same language.Response must be in the same language as the language of {Title}."""},
    {"role": "user", "content": f"""Analyze the (Title) and generate the structure of a blog post.The Title is {Title}
                                    First analyze the (Language) of {Title} and responsed must be in same language.If {Title} is (country) specific, Do not follow the country language,Just follow the overall language of {Title}.Generate a structure to help generate a post on that.Structure must contain (Reference) Heading. The structure would varry on the Title.Response must be in the same language as the language of {Title}"""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages, 
                        max_tokens= 4000, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()
    print(response_text)
    return response_text

#function for blog content
def generate_Blog_Content(Title, structure):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a {Title} and {structure} to generate a blog post depending on the given {structure}.First analyze the (Language) of {Title} and {structure} and responsed must be in same language.If {Title} and {structure} is (country) specific, Do not follow the country language,Just follow the overall language of {Title}.Response should be in the same language as {Title} and {structure}.Add (HTTP links) in (Reference) heading of Blog post."""},
    {"role": "user", "content": f"""Analyze the Title and Structure and generate a blog post. The Title is {Title}. The Structure is: {structure}.Blog post must contain 3000 to 4000 words.
                                    (Reference) heading in blog post must contain (HTTP links).Add (HTTP links) in (Reference) heading of Blog post. Strictly follow the structure given to you.First analyze the (Language) of {Title} and responsed must be in same language.If {Title} and {structure} is (country) specific, Do not follow the country language,Just follow the overall language of {Title}.Response should be in the same language as {Title} and {structure}"""}
    ]


    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        max_tokens= 4000, 
                        n=1, 
                        stop=None, 
                        temperature=0.5)

    response_text = response.choices[0].message.content.strip()

    return response_text

#function for twitter content

def generate_Twitter_content(topic):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a topic and generate a Twitter post without adding (emojis) in the post.First analyze the (Language) of {topic} and responsed must be in same language.If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.follow this instruction Rigorously.Response should be in the same language as {topic}."""},
    {"role": "user", "content": f"""Analyze the topic and generate a twitter post without any (emojis). The Topic is: {topic}.
                                    Follow the intructions:
                                    1.Response should be in the same language as {topic}.
                                    2.If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.
                                    3. Shape it like a tweet without containing any (emojis).  
                                    4. Do not Add any type of (emojis).                                                                    
                                    5. If relevant, you can include hashtags to categorize your tweet and make it more discoverable but do not add (emojis).
                                    6. It should not generate any harmful text.                             
                                    7.Rigorously follow (2) instruction."""}
    ]
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 60, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()

    return response_text

#function for Instagram content
def generate_Instagram_content(topic):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a topic and generate an Instagram caption without adding (emojis) in the caption.First analyze the (Language) of {topic} and responsed must be in same language.If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.follow this instruction precisely.Response must be in the same language as {topic}"""},
    {"role": "user", "content": f"""Analyze the topic and generate an instagram caption without any kind of (emojis). The Topic is: {topic}.
                                    Follow the instruction:
                                    1. Response should be in the same language as {topic}.
                                    2. If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.
                                    3. Shape it like an instagram caption without containing any (emojis).
                                    4. Do not Add the (emojis) in caption.
                                    5. Add a catchy opening line (Not more than one line, not any emojis).
                                    6. Generate text relevant to the topic.
                                    7. If relevant, you can include hashtags to categorize your post and make it more discoverable but not (emojis).
                                    8. It should not generate any harmful text.
                                    9. Rigorously Follow (2) instruction."""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 440, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()

    return response_text

#function for facebook content
def generate_Facebook_content(topic):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a topic and generate a Facebook post without containing any (emojis) in the post.First analyze the (Language) of {topic} and responsed must be in same language.If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}. Follow this instruction precisely.Response should be in the same language as {topic}"""},
    {"role": "user", "content": f"""Analyze the topic and generate a Facebook post. The topic is: {topic}.
                                    Follow the instructions:
                                    1. Response should be in the same language as {topic}.
                                    2. If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.
                                    3. Shape it like a Facebook Post without containing any kind of (emojis).
                                    4. Do not add the (emojis).
                                    5. Add a catchy opening line (Not more than one line).
                                    6. Generate text relevant to the topic. (Decide the size of the text depending upon the topic).
                                    7. If relevant, you can include hashtags to categorize your post and make it more discoverable but not (emojis).
                                    8. It should not generate any harmful text.
                                    9. Rigorously follow instruction (2)."""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 3000, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()

    return response_text
#function for Linkedin content
def generate_LinkedIn_content(topic):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a topic and generate an LinkedIn post.First analyze the (Language) of {topic} and responsed must be in same language.If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.And post should not contain any type of (emojis).Response should be in the same language as {topic}."""},
    {"role": "user", "content": f"""Analyze the topic and generate an LinkedIn post without any type of (emojis). The topic is: {topic}.
                                    Follow the instruction:
                                    1. Response should be in the same language as the language of {topic}.
                                    2. If {topic} is (country) specific, Do not follow the country language,Just follow the overall language of {topic}.
                                    3. Shape it like a LinkedIn Post without containing any kind of (emojis).
                                    4. Do not Add the (emojis).
                                    5. Start your post with an engaging and attention-grabbing opening sentence. This should be concise and highlight the main point or message you want to convey.
                                    6. Generate text relevent to the topic. (Decide the size of the text depending upon the topic).
                                    7. If relevant, you can include hashtags to categorize your post and make it more discoverable but not (emojis).
                                    8. It should not generate any harmful text.                                   
                                    9. Rigorously follow instruction (2)."""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 600, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()

    return response_text



#function for better image prompt
def TextRefine(topic):
    messages = [
    {"role": "system", "content": f"""Your role is to write the prompts that can generate stunning, realistic and high qulaity 4k images. Get the idea from {topic} and write prompts in atmost 30 words. The prompts should be relevant to {topic}."""},
    {"role": "user", "content": f"""Write a prompt that can generate the Stunning, realistic, high quality 4k photography and showcasing the beauty images based on the {topic}.Portray every detail of {topic} with stunning realism.The prompt should clearly define the image content and mention the quality of image.The prompt should be less than 30 words"""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 3000, 
                        n=1, 
                        stop=None, 
                        temperature=1)

    response_text = response.choices[0].message.content.strip()
   # print(response_text)
    return response_text

#function to remove emojis
def remove_emojis(text):
    # Regular expression to match emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical Symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A"
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    
    # Remove emojis from the text
    return emoji_pattern.sub(r'', text)


#Function for blog image prompt
def blogPromptGenerator(topic):
    messages = [
    {"role": "system", "content": f"""Your role is to generate some prompts that can generate stunning, realistic and high qulaity 4k images for (blog). Get the idea from {topic} of (blog) and write prompts in atmost 30 words. The prompts should be relevant to {topic}.prompt should focus on a distinct point and contribute to a cohesive set of images for (blog)'s multi-point structure.As we need 3 prompt in return.So, The list of prompt should be in the form: ['Prompt1','Prompt2','Prompt3']"""},
    {"role": "user", "content": f"""Write a prompt that can generate the Stunning, realistic, high quality 4k photography and showcasing the beauty images based on the {topic}.Portray every detail of {topic} with stunning realism because.Craft prompts to generate images that capture different facets of {topic} for (blog)'s post.The prompt should clearly define the image content and mention the quality of image.The prompt should clearly mention that there should not any text in the generated images. The prompt should be less than 30 words"""}
    ]
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 3000, 
                        n=1, 
                        stop=None, 
                        temperature=1)

    response_text = response.choices[0].message.content.strip()
   # print(response_text)
    return response_text

    
def blogMultiPromptGenerator(topic,content):
    prompt = f"Description: {topic},{content}"
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages = [
        {"role": "system", "content": f"""Your role is to generate some prompts that can generate stunning, realistic and high qulaity 4k images for (blog). Get the idea from {topic} and {content} of (blog) and write prompts in atmost 30 words. The prompts should be relevant to {topic} and {content}.prompt should focus on a distinct point and contribute to a cohesive set of images for (blog)'s multi-point structure.the response must not be empty like [''].As we need 3 prompt in return.So, The list of prompt should be in the form: ['Prompt1','Prompt2','Prompt3']"""},
        {"role": "user", "content": f"""Write a prompt that can generate the Stunning, realistic, high quality 4k photography and showcasing the beauty images based on the {topic} and {content}.Portray every detail of {topic} and {content} with stunning realism.Craft prompts to generate images that capture different facets of {topic} and {content} for (blog)'s post.The prompt should clearly define the image content and mention the quality of image. The prompt should be less than 30 words"""}
        ],
        temperature =1
        )
    except Exception as e:
        print(f'Error : {str(e)}')
    prompt_list = response['choices'][0]['message']['content']
    prompt_list = prompt_list.split("\n")
    if prompt_list is not None:
        print(prompt_list)
        return prompt_list
        
    else:
        return None
    
#function for splitting text
def split_text(text):
    length = len(text)
    third = length // 3

    part1 = text[:third]
    part2 = text[third:2*third]
    part3 = text[2*third:]

    return part1, part2, part3



#function for multi blogs titles

def blogMultiTitleGenerator(topic):
    prompt = f"Description: {topic}"
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
        {"role": "system", "content": f"""You are trained to analyze a {topic} and generate (5) (Titles) of a blog post depending upon the {topic}. Get the idea from {topic} of (blog) and write (5) (Titles). The Titles should be relevant to {topic}.As we need 5 Title in return.So, The List of Title should be in the form: ['Title1','Title2','Title3','Title4','Title5']. Do not add (\) in Titles.Response should be in the same language as {topic} """},
        {"role": "user", "content": f"""Analyze the {topic} and generate 5 Titles of a blog post. The topic is {topic}
                                        All 5 (Titles) should be vary from each other.The Titles should be relevant to {topic}.Do not add (\) in response.Response should be in the same language as {topic}"""}
        ],
        temperature =1
        )
    except Exception as e:
        print(f'Error : {str(e)}')
    prompt_list = response['choices'][0]['message']['content']
    prompt_list = prompt_list.replace("\\","")
    prompt_list = prompt_list.split("\n")
    # response_without_backslashes = prompt_list.replace("\\", "")
    if prompt_list is not None:
        text = [item for item in prompt_list if item]
        print(text)
        return text
        
    else:
        return None

#Seo keywords generation for Blog
def generate_Blog_SEO(topic):
    messages = [
    {"role": "system", "content": f"""You are trained to analyze a {topic} and generate (8) (SEO) keywords of a blog post depending upon the topic.Only generate (SEO) keywords.Response should be in the same language as {topic}"""},
    {"role": "user", "content": f"""Analyze the topic and generate the (SEO) of a blog post. The topic is {topic}
                                    Generate (8) (SEO) keywords for blog post.The (SEO) keywords would varry on the topic.And generate Seo keywords in the form of this '#SEO,#SEO,#SEO'.Only generate (SEO) keywords.Response should be in the same language as {topic}"""}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 3000, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()
    print(response_text)
    return response_text


#function for repos and blog topic
def topic_generate(prompt):
    messages = [
    {"role": "system", "content": f"""Your role is to Analyze the {prompt} and generate (one) main (Topic) related to {prompt}. Get the idea from {prompt} and write (Topic).The (Topic) should be relevant to {prompt}.And (Topic) must be only (one).Response should contain only topic and focus on the main (topic) of {prompt}.Response contain at most 2 words."""},
    {"role": "user", "content": f"""Write a (one) main (Topic) based on the {prompt}.Craft Topic facets of {prompt}.And (Topic) must be only (one).Response should focus on the main (topic) of {prompt}.Response contain at most 2 words."""}
    ]
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages, 
                        max_tokens= 3000, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()
    print(response_text)
    return response_text

def blog_repo_links(topic):
    llm = OpenAI(temperature=0, 
                 openai_api_key=openapi_key)
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools = [Tool(
                name="Intermediate Answer",
                func=search.run,
                description="useful for when you need to ask with search",)]
    self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    response = self_ask_with_search.run(f"Give me valid http links for news articles,videos,blog posts on the topic {topic}")
    # tools = load_tools(["serpapi"], llm=llm )
    # agent = initialize_agent(search ,llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    # response=agent.run(f"Give me http links for news articles, blog posts on the topic {topic}")
    print(response)
    return response


#speech to text
def speechToText(filename):
    transcribed_text = openai.Audio.transcribe("whisper-1", filename)
    return transcribed_text['text']


#function for comment replier
#function for comment replier
def commentReplier(comment):
    messages = [
    {"role": "system", "content": f"""You are trained to Analyze the {comment} and generate the (Reply) depending on {comment}.First analyze a {comment} (Tone) and (Topic) and generate (Reply) according to {comment}.Response only contain (Reply) with no heading.Response should be in the same language as {comment}"""},
    {"role": "user", "content": f"""Analyze the {comment} and generate Reply.The comment is: {comment}.
                                    Follow the instruction:
                                    1. Response should be in the same language as {comment}
                                    2. If {comment} is (Rude) than (reply) should be (neutral).
                                    3. If {comment} is (Jolly) than (reply) should be (Jolly). 
                                    4. If {comment} is (Neutral) than (reply) should be (Neutral).
                                    5. If {comment} is (Appreciation) than reply with ("Thanks").
                                    6. If you do not understand the {comment} than reply with ("Thanks for sharing your FeedBack.").
                                    7. If {comment} is (greeting) than greet accordingly but do not give any suggestions or assistance.
                                    8. If {comment} is a (suggestion) just reply with "We appreciate your suggestion".
                                    9. If {comment} is (opinion) Do not agree with it just reply with ("Thanks for Sharing your feedback)".
                                    10.If  {comment} is (Promotional) just reply with ("Thanks for Sharing your feedback").
                                    11.If {comment}  is (Request) just reply with ("I Appreciate Your comment").
                                    12. If {comment} is (informative and business-oriented) just reply with (""Thanks for Sharing your feedback").
                                    13. If {comment} contain (some key points). Do not Explain it.Just reply with ("I Appreciate Your comment.").
                                    14.If {comment} is (Defining) someone or something just reply with "Thank you for sharing your perspective". 
                                    15.If {comment} is (instructive) than just reply with "Thanks."
                                    16. If {comment} is (Toxic) just reply with "Thank you for sharing your perspective".
                                    17.If {comment} is (abusive) just reply with "Thank you for sharing your perspective".
                                    18.Do not add {comment} in response.
                                    19. If {comment} is containing some personal Insights or sharing than just reply with "Thanks For sharing".
                                    20. Ananlyze the (Topic) and (Tone) of {comment} properly.
                                    21. Reply Should be (Concise) and to the point.
                                    22. Reply should contain (At most) 20 words.
                                    23. Response only contain (Reply) with no (heading).
                                    24. It should not generate any harmful text.                                   
                                    """}
    ]

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        messages=messages, 
                        max_tokens= 600, 
                        n=1, 
                        stop=None, 
                        temperature=0.6)

    response_text = response.choices[0].message.content.strip()
    print(response_text)
    return (response_text)


#function to generte Images
def generate_thumbnail_background(text):
    url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
    data = {
    "prompt": text,
    "negative_prompt": "low quality, grainy, anime, cartoon, non-realistic,logo",
    "style": "base",
    "samples": 1,
    "scheduler": "dpmpp_sde_ancestral",
    "num_inference_steps": 25,
    "guidance_scale": 8,
    "strength": 1,
    "seed": 468685,
    "img_width": 1024,
    "img_height": 1024,
    "refiner": "yes",
    "base64": False
        }
    response = requests.post(url, json=data, headers={'x-api-key': segmind_API})
    image_data = response.content
    image = Image.open(BytesIO(image_data))
    image.save("output_image.png")
    return image

#function to generte multi images
def generate_multi_thumbnail_background(text1):
    generated_images = []
    text = [item for item in text1 if item]
    for i in range(0,len(text)):
        punctuation_list = '0123456789,;.:?\/"'
        text[i] = text[i].replace(text[i][:3], "")
        for punctuation in punctuation_list:
            text[i] = text[i].replace(punctuation, "")
        new=text[i]
        url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
        data = {
        "prompt": new,
        "negative_prompt": "low quality, grainy, anime, cartoon, non-realistic,logo",
        "style": "base",
        "samples": 1,
        "scheduler": "dpmpp_sde_ancestral",
        "num_inference_steps": 25,
        "guidance_scale": 8,
        "strength": 1,
        "seed": 468685,
        "img_width": 1024,
        "img_height": 1024,
        "refiner": "yes",
        "base64": False
            }
        response = requests.post(url, json=data, headers={'x-api-key': segmind_API})
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        generated_images.append(image)
        image.save(f"Add_gen{i}.jpg")
    return generated_images


