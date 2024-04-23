import os
import requests
import fitz
from tqdm.auto import tqdm
import pandas as pd
import random
import numpy as np
from spacy.lang.en import English
import re
# Requires !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)

# Open PDF and load target page
folder_path = "./media/ncert/class_11/history" # requires PDF to be downloaded
# doc = fitz.open(pdf_path)
# page = doc.load_page(5 + 41) # number of page (our doc starts page numbers on page 41)

# # Get the image of the page
# img = page.get_pixmap(dpi=300)

# # Optional: save the image
# #img.save("output_filename.png")
# doc.close()

# # Convert the Pixmap to a numpy array
# img_array = np.frombuffer(img.samples_mv, 
#                           dtype=np.uint8).reshape((img.h, img.w, img.n))

# Display the image using Matplotlib
# import matplotlib.pyplot as plt
# plt.figure(figsize=(13, 10))
# plt.imshow(img_array)
# plt.title(f"Query: '{query}' | Most relevant page:")
# plt.axis('off') # Turn off axis
# plt.show()





def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(folder_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    pages_and_texts = []
    for file_name in os.listdir(folder_path):
        pdf_path = os.path.join(folder_path, file_name)
        doc = fitz.open(pdf_path)  # open a document
        file_name = os.path.basename(pdf_path)
        
        for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
            text = page.get_text()  # get plain text encoded as UTF-8
            text = text_formatter(text)
            pages_and_texts.append({"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                                    "page_char_count": len(text),
                                    "page_word_count": len(text.split(" ")),
                                    "page_sentence_count_raw": len(text.split(". ")),
                                    "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                    "text": text,
                                    "book_name": file_name})
    return pages_and_texts
    # files = []
    # pages_and_texts = []
    # result_arr = []
    # pages_and_texts = open_and_read_pdf('./media/ncert/class_11/history/Class11_World History_An Empire Across Three Continents.pdf')
    # for file_name in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, file_name)
    #     if os.path.isfile(file_path):
    #         files.append(file_path)
    #     print("filepath is ", file_path)    
    #     result_arr = open_and_read_pdf(file_path);    
    #     pages_and_texts = np.vstack((pages_and_texts, result_arr))
    #     # pages_and_texts.append(...result_arr)
    #     print("Files in folder:", pages_and_texts)
    #     # pages_and_texts[:2]
    #     print("Files in folder:", pages_and_texts)        
    # return pages_and_texts


pages_and_texts = open_and_read_pdf(folder_path)
# print("Files in folder:", files)

# print("random sample", random.sample(pages_and_texts, k=3))

df = pd.DataFrame(pages_and_texts)
print(df.head())

print('Get stats', df.describe().round(2))

# Ingest text -> split it into groups/chunks -> embed the groups/chunks -> use the embeddings
# Further text processing (splitting pages into sentences)

# We don't necessarily need to use spaCy, however, it's an open-source library designed to do NLP tasks like this at scale.

# So let's run our small sentencizing pipeline on our pages of text.

nlp = English()

# All-mpnet-base-v2 can take at max 384 tokens
# LLms cant take infinite tokens Eg: chatgpt4 -> token limit
# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    
    # Make sure all sentences are strings
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    
    # Count the sentences 
    item["page_sentence_count_spacy"] = len(item["sentences"])
    
df = pd.DataFrame(pages_and_texts)
print('sentencizer',df.describe())    
# print(df.columns) 
# df.describe(include=[np.number])


# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10 

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, 
               slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Loop through pages and texts and split sentences into chunks
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])
    
df = pd.DataFrame(pages_and_texts)
print("After chunking", df.describe())   


# Split each chunk into its own item
# To handle the embedding effectively -> Good granularity for the text sample used in our model
pages_and_chunks = []
ids = []
count = 1
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        chunk_dict["book_name"] = item["book_name"]
        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
        
        pages_and_chunks.append(chunk_dict)
        
        
print(random.sample(pages_and_chunks, k=1))    
df = pd.DataFrame(pages_and_chunks)
print(df.describe().round(2))

# Show random chunks with under 30 tokens in length
min_token_length = 30
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

# Meaning rather than directly mapping words/tokens/characters to numbers directly (e.g. {"a": 0, "b": 1, "c": 3...}), 
# the numerical representation of tokens is learned by going through large corpuses of text and figuring out how different tokens relate to each other.

# Send the model to the GPU
embedding_model.to("cpu") # requires a GPU installed, for reference on my local machine, I'm using a NVIDIA RTX 4090

# Create embeddings one by one on the GPU
for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    
    
# # Turn text chunks into a single list
# text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]    
    
# # Embed all texts in batches
# text_chunk_embeddings = embedding_model.encode(text_chunks,
#                                                batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
#                                                convert_to_tensor=True) # optional to return embeddings as tensor instead of array

# print(text_chunk_embeddings)    

# Save embeddings to file
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

# import chromadb
# from chromadb.config import Settings


# client = chromadb.Client()

# collection = client.get_or_create_collection("class_11")

# # print("item lenght is ", pages_and_chunks_over_min_token_len)
# for item in tqdm(pages_and_chunks):
#     print("item is ", item)
#     print("item id is ", item["book_name"] + str(item["page_number"]) + str(count))
#     ids = [item["book_name"] + str(item["page_number"]) + str(count)]
#     count = count+1
#     collection.add(
#     ids=ids,
#     documents=[item["sentence_chunk"]]
#     )


# results = collection.query(
#     query_texts=["Who is genghis khan?"],
#     n_results=2
# )

# print(results)