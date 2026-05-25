#!/usr/bin/env python
# coding: utf-8

# # Lesson 2: Tool Calling

# ## Setup

# In[1]:


from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()


# In[2]:


import nest_asyncio
nest_asyncio.apply()


# ## 1. Define a Simple Tool

# In[12]:


from llama_index.core.tools import FunctionTool

def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)

def multiplication(x: int, y:int) ->int:
    """Multiplies two integers"""
    return x*y



add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)
multiply_tool=FunctionTool.from_defaults(fn=multiplication)


# In[14]:


from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
response = llm.predict_and_call(
    [add_tool, mystery_tool,multiply_tool], 
    "multiply on 8 and 5", 
    verbose=True
)
print(str(response))


# ## 2. Define an Auto-Retrieval Tool

# ### Load Data

# To download this paper, below is the needed code:
# 
# #!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
# 
# **Note**: The pdf file is included with this lesson. To access it, go to the `File` menu and select`Open...`.

# In[7]:


from llama_index.core import SimpleDirectoryReader
# load documents
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()


# In[ ]:


from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)


# In[ ]:


print(nodes[0].get_content(metadata_mode="all"))


# In[ ]:


from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)


# In[ ]:


from llama_index.core.vector_stores import MetadataFilters

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query(
    "What are some high-level results of MetaGPT?", 
)


# In[ ]:


print(str(response))


# In[ ]:


for n in response.source_nodes:
    print(n.metadata)


# ### Define the Auto-Retrieval Tool

# In[ ]:


from typing import List
from llama_index.core.vector_stores import FilterCondition


def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response
    

vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)


# In[ ]:


llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.predict_and_call(
    [vector_query_tool], 
    "What are the high-level results of MetaGPT as described on page 2?", 
    verbose=True
)


# In[ ]:


for n in response.source_nodes:
    print(n.metadata)


# ## Let's add some other tools!

# In[ ]:


from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool

summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get a summary of MetaGPT"
    ),
)


# In[ ]:


response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What are the MetaGPT comparisons with ChatDev described on page 8?", 
    verbose=True
)


# In[ ]:


for n in response.source_nodes:
    print(n.metadata)


# In[ ]:


response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What is a summary of the paper?", 
    verbose=True
)

