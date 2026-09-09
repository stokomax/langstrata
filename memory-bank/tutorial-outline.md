LangChain vs LangGraph — What Each Package Actually Is
LangChain (langchain, langchain-core, langchain-community)

LangChain is a toolkit for building LLM-powered components — the building blocks layer. It gives you:

Model abstractions — a unified interface to call any LLM (OpenAI, Anthropic, Bedrock, etc.) via init_chat_model()
Chains — linear sequences of steps composed via LCEL (LangChain Expression Language), e.g. prompt → model → parser
Retrievers and RAG utilities — document loaders, text splitters, vector store integrations
Tool/function calling abstractions — @tool decorator, bind_tools(), structured output parsing
Memory primitives — ConversationBufferMemory and similar (mostly legacy now)
Community integrations — langchain-community and provider packages like langchain-anthropic, langchain-openai

Think of LangChain as Lego bricks — individual components you assemble. It has no concept of state machines, cycles, or long-running agent coordination.

LangGraph (langgraph)

LangGraph is a runtime for stateful, cyclical agent workflows, built on top of LangChain primitives. It gives you:

StateGraph — a graph where nodes are Python functions and edges control flow; state flows through the graph and accumulates via reducers
Persistence / checkpointing — every step is checkpointed to Postgres/SQLite; you can resume interrupted runs, support human-in-the-loop, and replay
Cycles and loops — unlike LCEL chains which are DAGs, LangGraph graphs can loop back (e.g. supervisor → worker → supervisor)
Send API — dynamic fan-out: a node can spawn multiple parallel branches at runtime
Subgraphs — embed one compiled graph as a node inside another
Streaming — token-level and node-level streaming of agent state as it evolves
LangGraph Server (Agent Server) — a production HTTP server that wraps your graphs, exposes REST + SSE + A2A endpoints, manages a process pool, and backs everything with Postgres + Redis

Think of LangGraph as the orchestration runtime — it controls how agents run, persist, branch, and communicate.

The Dependency Relationship
langgraph
    └── depends on → langchain-core
                         └── provides: BaseMessage, ChatModel,
                                       RunnableConfig, @tool, bind_tools()

langchain          ← optional, adds community integrations,
                     LCEL chains, retrievers, legacy memory

LangGraph depends on langchain-core (the minimal, stable base), not on the full langchain package. When you write a LangGraph agent you're typically using:

langchain-core / langchain-anthropic / langchain-openai — for model calls and tool binding
langgraph — for the graph, state, checkpointing, and server

The full langchain package (LCEL, retrievers, document loaders) is only needed if you're building RAG pipelines or using its higher-level chain abstractions alongside your graph.

Practical Distinction in Your Driftline Context

In your codebase the split is visible directly:

_build_agent() uses LangChain primitives — init_chat_model(), bind_tools(), @tool, BaseMessage
build_supervisor_graph() uses LangGraph primitives — StateGraph, Send, CompiledGraph, checkpointing, langgraph_sdk

The workers are LangChain agents; the graph that coordinates them is LangGraph. That's the intended division of labor between the two packages.
