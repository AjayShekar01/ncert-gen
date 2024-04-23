## RAG
Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

In other words, it fills a gap in how LLMs work. Under the hood, LLMs are neural networks, typically measured by how many parameters they contain. An LLM’s parameters essentially represent the general patterns of how humans use words to form sentences.

That deep understanding, sometimes called parameterized knowledge, makes LLMs useful in responding to general prompts at light speed. However, it does not serve users who want a deeper dive into a current or more specific topic.


# Create embeddings from the list of pdf files stored in the media folder.

```
python3 ./rag_data_pipeline.py

```

PDF files contents are broken down into chunks with all the metadata used to identify them.
Embeddings are created to give the context to llm that we can choose.
text_chunks_and_embeddings_df.csv is the out put of the rag data processing.

# Import an LLM, Hard Prompt with RAG context from the embedding file to get specific answers

```
python3 ./rag.py
```