# Agentic RAG with LlamaIndex - Learning Path

A comprehensive learning project for building **Agentic Retrieval-Augmented Generation (RAG)** systems using LlamaIndex. This repository contains interactive lessons and practical implementations of advanced RAG patterns.

## 📚 Overview

This project explores the core concepts behind building intelligent agents that can reason over knowledge bases and invoke tools dynamically. Learn how to create systems that:

- **Route queries intelligently** to appropriate processing engines
- **Call external tools and APIs** based on reasoning
- **Maintain context** through multi-turn conversations
- **Execute complex workflows** combining retrieval and computation

### What is Agentic RAG?

Agentic RAG goes beyond traditional RAG by adding:
- **Dynamic routing** - Intelligently choose between different retrieval strategies
- **Tool integration** - Agents can call functions, APIs, and external services
- **Reasoning loops** - Multi-step reasoning with feedback and refinement
- **Context awareness** - Maintain state across interactions
- **Adaptive behavior** - Learn and adjust strategies based on outcomes

## 🏗️ Project Structure

```
llama_index/
├── L1_Router_Engine/              # Lesson 1: Intelligent Query Routing
│   ├── L1_Router_Engine.ipynb      # Interactive notebook with visualizations
│   └── L1_Router_Engine.py         # Python script version
├── L2_Tool_Calling/                # Lesson 2: Agent Tool Invocation
│   ├── L2_Tool_Calling.ipynb       # Interactive notebook
│   └── L2_Tool_Calling.py          # Python script version
├── requirements.txt                # Project dependencies
├── helper.py                       # Utility functions and helpers
└── metagpt.pdf                     # Sample document for RAG exercises
```

## 📖 Lessons

### Lesson 1: Router Engine (L1_Router_Engine)

**Learning Objectives:**
- Understand query routing in RAG systems
- Implement multiple query engines (Summary vs Vector)
- Use LlamaIndex's QueryEngineTool abstraction
- Build intelligent routers that select the best retrieval strategy

**Key Topics:**
- `SummaryIndex` - Document summarization and abstract retrieval
- `VectorStoreIndex` - Semantic similarity-based retrieval
- `QueryEngineTool` - Wrapping query engines as callable tools
- `QueryEngineTool.from_defaults()` - Creating tools with metadata
- Router selection logic based on query type

**What You'll Learn:**
```python
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool

# Create multiple retrieval strategies
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Wrap as tools for agent routing
summary_tool = QueryEngineTool.from_defaults(query_engine=...)
vector_tool = QueryEngineTool.from_defaults(query_engine=...)

# Agent routes between them based on query characteristics
```

### Lesson 2: Tool Calling (L2_Tool_Calling)

**Learning Objectives:**
- Define custom functions as tools
- Implement function-based tool calling
- Build auto-retrieval tools for document querying
- Create agents that choose and invoke tools

**Key Topics:**
- `FunctionTool` - Converting Python functions into agent tools
- `FunctionTool.from_defaults()` - Creating tools from functions
- Tool metadata and descriptions for agent selection
- `llm.predict_and_call()` - LLM-based tool invocation
- Auto-retrieval patterns for efficient document access

**What You'll Learn:**
```python
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Define tools as functions
def add(x: int, y: int) -> int:
    """Adds two integers."""
    return x + y

# Convert to callable tools
add_tool = FunctionTool.from_defaults(fn=add)

# LLM invokes tools based on reasoning
llm = OpenAI(model="gpt-3.5-turbo")
response = llm.predict_and_call([add_tool], "what is 5 + 3?")
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible LLM provider)
- pip or conda for package management

### Installation

1. **Clone or download the repository:**
```bash
cd llama_index
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

### Running the Lessons

**Option 1: Interactive Jupyter Notebooks** (Recommended for learning)
```bash
jupyter notebook L1_Router_Engine/L1_Router_Engine.ipynb
jupyter notebook L2_Tool_Calling/L2_Tool_Calling.ipynb
```

**Option 2: Python Scripts**
```bash
python L1_Router_Engine/L1_Router_Engine.py
python L2_Tool_Calling/L2_Tool_Calling.py
```

## 🔑 Key Concepts

### Query Routing
Agents don't query all data sources equally. They analyze the query and route to appropriate engines:
- **Summary queries** → Use SummaryIndex for high-level overviews
- **Specific queries** → Use VectorIndex for precise information retrieval
- **Hybrid queries** → Combine multiple strategies

### Tool Calling Pattern
```
Input → LLM Reasoning → Tool Selection → Tool Execution → Response
```

The LLM analyzes the request, selects appropriate tools, executes them, and synthesizes results.

### Agent Loop
```
1. Parse user query
2. Route to appropriate engine/tool
3. Execute retrieval/function
4. Check if answer is sufficient
5. If not, refine and retry (loop)
6. Return final response
```

### Tool Metadata
Tools need clear descriptions so agents can select them:
```python
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Use this tool for summarizing topics or getting an overview.",
    name="summary_tool"
)
```

## 📊 Learning Flow

```
Start (Basic RAG Understanding)
    ↓
L1: Router Engine (Query Routing Strategy)
    • Multiple index types
    • Tool wrapping
    • Router selection logic
    ↓
L2: Tool Calling (Function Invocation)
    • Function tools
    • LLM-based selection
    • Auto-retrieval
    ↓
Advanced: Multi-Agent Systems
    • Orchestrating multiple agents
    • Complex workflows
    • State management
```

## 💡 Usage Examples

### Example 1: Simple Query Routing
```python
# The router decides which tool to use
user_query = "What is the main topic of the paper?"

# Could route to summary_tool
# → Uses SummaryIndex for quick overview

# vs query = "How does the technique work in section 3?"
# Could route to vector_tool
# → Uses VectorIndex for specific information
```

### Example 2: Tool Invocation
```python
# Define a tool
def search_documents(query: str) -> str:
    """Search documents for relevant content."""
    return vector_index.as_query_engine().query(query)

# Agent uses it
search_tool = FunctionTool.from_defaults(fn=search_documents)
agent.invoke([search_tool], "Find information about embeddings")
```

### Example 3: Complex Workflow
```python
# Tools available to agent
tools = [
    summary_tool,           # Get overview
    vector_tool,           # Search specific info
    add_tool,              # Perform calculations
    retrieval_tool         # Auto-retrieve documents
]

# Agent: "Find papers on topic X, count them, summarize each"
# → Uses multiple tools in sequence
```

## 🛠️ Development

### Modifying Lessons

1. **Edit the notebooks** for interactive exploration:
   - Add new code cells
   - Experiment with parameters
   - Visualize results

2. **Edit the Python scripts** for standalone execution:
   - Test different configurations
   - Run on servers/clusters
   - Integrate with pipelines

3. **Helper utilities** (in `helper.py`):
   - API key management
   - Common functions
   - Utility methods

### Extending the Project

```python
# Add new lesson (L3_Agent_Orchestration)
L3_Agent_Orchestration/
├── L3_Agent_Orchestration.ipynb
├── L3_Agent_Orchestration.py
└── requirements.txt

# Add new tools
def custom_tool(param: str) -> str:
    """Tool description for agent selection."""
    return "result"

custom = FunctionTool.from_defaults(fn=custom_tool)
```

## 📦 Dependencies

Key packages used in this project:

```
llama-index              # Core RAG and agent framework
llama-index-llms-openai # OpenAI LLM integration
llama-index-embeddings-openai # OpenAI embeddings
nest-asyncio            # Async support in notebooks
python-dotenv           # Environment variables
```

See `requirements.txt` for complete list.

## 🔗 Resources

### LlamaIndex Documentation
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Query Engines](https://docs.llamaindex.ai/en/stable/module_guides/querying/query_engine/)
- [Tools & Agents](https://docs.llamaindex.ai/en/stable/module_guides/agents/tools/)
- [Routers](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/)

### RAG & Agents
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/use_cases/rag/)
- [Building Agents](https://docs.llamaindex.ai/en/stable/module_guides/agents/)
- [Tool Integration](https://docs.llamaindex.ai/en/stable/module_guides/tools/)

### LLM Foundations
- [OpenAI API](https://platform.openai.com/docs/)
- [Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Function Calling](https://platform.openai.com/docs/guides/function-calling)

## 🎯 Learning Objectives Summary

By completing this course, you'll be able to:

✅ Understand RAG fundamentals and advanced patterns
✅ Build intelligent query routers
✅ Implement tool calling systems
✅ Create agents that reason over knowledge bases
✅ Combine multiple retrieval strategies
✅ Design workflow orchestration
✅ Deploy production-ready systems

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new lessons and exercises
- Improve documentation
- Create example implementations
- Share insights and improvements

## 📝 License

This project is for educational purposes.

## 💬 Support

For questions about:
- **LlamaIndex**: Check the [official documentation](https://docs.llamaindex.ai/)
- **OpenAI**: Visit [OpenAI API docs](https://platform.openai.com/docs/)
- **This course**: Review the lesson notebooks and helper code

## 🗓️ Version History

- **v1.0** (May 2026) - Initial release with L1 and L2 lessons

---

**Happy Learning! 🚀**

Start with Lesson 1 (Router Engine) to understand query routing, then move to Lesson 2 (Tool Calling) to learn about agent tool invocation. These foundations will prepare you for building sophisticated agentic systems!
