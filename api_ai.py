from flask import Flask, request, jsonify, render_template
import chromadb
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is missing.")

client = AsyncOpenAI(api_key=api_key)

# Load the data from the Excel file
file_path = 'data.xlsx'  # path to Excel file
data = pd.read_excel(file_path)

# Combine relevant columns into a single text format
data_to_process = [
    # .lower added to make all options lowercased
    f"DESCRIPTION: {row['DESCRIPTION'].lower()}\nAçıklama: {row['Açıklama - GenAI'].lower()}\nKeywords: {row['Keywords - GenAI'].lower()}"
    for _, row in data.iterrows()
]

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(async_client=client)

# Create the Chroma Vector Store
vector_store = Chroma(
    collection_name="descriptions",
    embedding_function=embeddings,
    client=chroma_client
)

# Add documents to ChromaDB
vector_store.add_texts(data_to_process)

# Chat completion settings
settings = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    # more settings if needed
}

def construct_prompt(user_input, search_results, prompt_type):
    if prompt_type == "default":
        # First Prompt
        prompt = (
            f"Kullanıcı girdisi: {user_input}\n"
            f"Aşağıdaki DESCRIPTIONlardan hangisi alakalı ve uygundur? "
            f"Aşağıda verilen tüm opsiyonları ve opsiyonlardaki DESCRIPTION, "
            f"Açıklama ve Keywordsleri dikkate alarak cevap veriniz:\n\n"
        )
        for i, result in enumerate(search_results, start=1):
            prompt += f"Opsiyon {i}:\n{result.page_content}\n\n"
        print(prompt)

        prompt += (
            "Kullanıcının bulmak istediğini düşündüğün DESCRIPTION ismini paylaşır mısın? Ve olabileceğini düşündüğün en uygun 5 tane daha DESCRIPTION ismini paylaşır mısın? "
            "Paylaştığın DESCRIPTION isimlerinde herhangi bir değişim yapma sana sunulan neyse onlardan seç ve kullanıcıya sun, kendin asla oluşturma. "
            "Eğer ucu çok açık bir şey yazıldıysa daha detaylandırmak için soru sorabilirsin. Ama sadece ucu çok açıksa."
        )
    elif prompt_type == "custom":
        # Second prompt
        prompt = (
            f"Kullanıcı girdisi: {user_input}\n"
            f"Aşağıda verilen DESCRIPTION seçeneklerinden kullanıcı girdisine en uygun olanları seçiniz. "
            f"Seçimleriniz sadece sağlanan DESCRIPTION'lar arasından olmalıdır. Kendiniz yeni bir DESCRIPTION oluşturmayınız.\n\n"
        )
        
        for i, result in enumerate(search_results, start=1):
            prompt += f"Opsiyon {i}:\n{result.page_content}\n\n"
        print(prompt)

        prompt += (
            "Yukarıda verilen seçenekler arasından, kullanıcının aradığına en uygun olduğunu düşündüğünüz DESCRIPTION ismini seçiniz. "
            "Eğer varsa, kullanıcıya sunabileceğiniz 5 alternatif DESCRIPTION daha seçiniz. "
            "Eğer seçeneklerin hiçbiri tam olarak uygun değilse veya kullanıcının girdisi çok belirsizse, "
            "daha fazla bilgi istemek için bir soru sorun."
            "\n\n"
            "Cevabınız şu formatta olmalıdır:\n"
            "En Uygun DESCRIPTION: [En uygun DESCRIPTION ismini buraya yazın]\n"
            "Alternatifler:\n"
            "   - Alternatif 1: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 2: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 3: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 4: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 5: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "Eğer seçenekler uygun değilse, daha fazla bilgi istemek için: 'Kullanıcının girdisi daha net olmalı, lütfen daha fazla ayrıntı verin.'"
        )
    elif prompt_type == "keyword_generation":
        # Third prompt, keyword gen
        prompt = (
            f"Kullanıcı girdisi: {user_input}\n"
            f"Aşağıda verilen DESCRIPTION seçeneklerinden kullanıcı girdisine en uygun olanları seçiniz ve gerektiğinde kullanıcı girdisine ek anahtar kelimeler oluşturunuz. "
            f"Oluşturduğunuz anahtar kelimelerle ve kullanıcı girdisiyle alakalı en uygun DESCRIPTION'ları seçiniz. "
            f"Seçimleriniz sadece sağlanan DESCRIPTION'lar arasından olmalıdır. Kendiniz yeni bir DESCRIPTION oluşturmayınız.\n\n"
        )

        for i, result in enumerate(search_results, start=1):
            prompt += f"Opsiyon {i}:\n{result.page_content}\n\n"
        print(prompt)

        prompt += (
            "Yukarıda verilen seçenekler arasından, en çok kullanıcının girdisini dikkate alarak en uygun olduğunu düşündüğünüz DESCRIPTION ismini seçiniz. "
            "Eğer varsa, kullanıcıya sunabileceğiniz 5 alternatif DESCRIPTION daha seçiniz. "
            "Eğer seçeneklerin hiçbiri tam olarak uygun değilse veya kullanıcının girdisi çok belirsizse, "
            "daha fazla bilgi istemek için bir soru sorun."
            "\n\n"
            "Cevabınız şu formatta olmalıdır:\n"
            "En Uygun DESCRIPTION: [En uygun DESCRIPTION ismini buraya yazın]\n"
            "Alternatifler:\n"
            "   - Alternatif 1: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 2: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 3: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 4: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "   - Alternatif 5: [Alternatif DESCRIPTION ismini buraya yazın]\n"
            "Eğer seçenekler uygun değilse, daha fazla bilgi istemek için: 'Kullanıcının girdisi daha net olmalı, lütfen daha fazla ayrıntı verin.'"
        )
    return prompt

async def get_additional_keywords(client, user_input):
    """Generate additional keywords using GPT."""
    response = await client.chat.completions.create(
        messages=[
            {"content": "Aşağıdaki metne dayalı en fazla 10 ek anahtar kelimeler oluşturun:", "role": "system"},
            {"content": user_input, "role": "user"}
        ],
        **settings
    )
    keywords = response.choices[0].message.content.strip().split(',')
    return keywords

@app.route('/search', methods=['POST'])
async def search():
    # Convert user input to lowercase
    user_input = request.json.get('input').lower()
    prompt_type = request.json.get('prompt_type', 'default')  # Default prompt type is "default"
    if not user_input:
        return jsonify({"error": "Missing input data"}), 400

    if prompt_type == "keyword_generation":
        # generate additional keywords for more accurate answers
        #additional_keywords = await get_additional_keywords(client, user_input)
        # lowercase for all keywords
        additional_keywords = [kw.lower() for kw in await get_additional_keywords(client, user_input)]
        print(additional_keywords)
        # similarity search with user input and additional keywords
        #sim_search = 
        #search_results = vector_store.similarity_search("0. " + user_input + "\n" + ", ".join(additional_keywords), k=40)
        #search_results = vector_store.similarity_search(user_input, k=30)
        #print("0. " + user_input + "\n" + ", ".join(additional_keywords))

        # Perform similarity search with user input only, get top 20 results
        search_results_input_only = vector_store.similarity_search(user_input, k=20)

        # Perform similarity search with additional keywords, get top 20 results
        search_results_keywords = vector_store.similarity_search(", ".join(additional_keywords), k=20)

        # Combine the two sets of search results
        combined_search_results = search_results_input_only + search_results_keywords

        # Remove duplicates based on content or similarity scores
        unique_combined_search_results = list({result.page_content: result for result in combined_search_results}.values())
        
        # Sort the combined results based on similarity scores or any other metric if needed
        search_results = unique_combined_search_results[:30]  # Ensures up to 30 results
    else:
        # Search only for user_inputs
        search_results = vector_store.similarity_search(user_input, k=30)

    # Construct the prompt using the relevant search results
    prompt = construct_prompt(user_input, search_results, prompt_type)

    # Get the response from the OpenAI API
    response = await client.chat.completions.create(
        messages=[
            {
                "content": "Sen yardımcı bir arama asistanısın, her zaman Türkçe yanıt veriyorsun.",
                "role": "system"
            },
            {
                "content": prompt,
                "role": "user"
            }
        ],
        **settings
    )

    # Extract results from OpenAI response
    response_text = response.choices[0].message.content

    # Return the response in JSON format
    return jsonify({"response": response_text.split('\n')})  # Assuming response is split by lines which is correct for last 2 prompts 

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
