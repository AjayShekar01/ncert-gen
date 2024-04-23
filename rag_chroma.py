import chromadb
from chromadb.config import Settings


client = chromadb.Client()

collection = client.get_or_create_collection("class_11")

# print("item lenght is ", pages_and_chunks_over_min_token_len)
for item in tqdm(pages_and_chunks):
    print("item is ", item)
    print("item id is ", item["book_name"] + str(item["page_number"]) + str(count))
    ids = [item["book_name"] + str(item["page_number"]) + str(count)]
    count = count+1
    collection.add(
    ids=ids,
    documents=[item["sentence_chunk"]]
    )


results = collection.query(
    query_texts=["Who is genghis khan?"],
    n_results=2
)

print(results)