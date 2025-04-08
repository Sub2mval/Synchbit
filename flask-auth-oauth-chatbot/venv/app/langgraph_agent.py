# Create app/langgraph_agent.py

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Set
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres import PostgresSaver # Or another checkpointer (Postgres?)
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_together import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool, tool # Use the decorator for simplicity
from langchain_core.pydantic_v1 import BaseModel, Field

from app import db
from app.models import Conversation, DataUpload, ChatMessage, ProposedWrite, User
from app.services import get_embedding_model, init_pinecone # Use existing services where possible
from sqlalchemy.sql import text # For executing SQL
from sqlalchemy import create_engine, inspect as sql_inspect # To get schema
from sqlalchemy.exc import ProgrammingError

# --- Configuration ---
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" # Or your preferred model
# Consider a separate checkpointer DB if using SQLite isn't ideal for Vercel/production
# memory = SqliteSaver.from_conn_string("langgraph_checkpoints.sqlite")
# For simplicity now, use in-memory (won't persist approvals across restarts!)
memory = PostgresSaver.from_conn_string(":memory:")


# --- Agent State ---
class AgentState(TypedDict):
    conversation_id: int
    user_id: int
    selected_upload_ids: List[int] # IDs of active DataUploads for this convo
    available_data_types: Set[str] # {'tabular', 'text'} based on selected uploads
    tabular_upload_ids: List[int] # Just the tabular ones
    text_upload_ids: List[int] # Just the text ones
    chat_history: List[BaseMessage]
    user_query: str # The latest query from the user

    # Tool-related state
    retrieved_docs: List[Dict[str, Any]] | None # Context from Pinecone
    tabular_schemas: Dict[int, str] | None # Schema info {upload_id: schema_desc}
    sql_query: str | None # Generated SQL for reading
    sql_results: str | None # Stringified results from SQL query
    write_description: str | None # User request interpreted for writing
    proposed_write_sql: str | None # Generated INSERT/UPDATE/DELETE SQL
    proposed_write_description: str | None # LLM description of the SQL
    proposed_write_id: int | None # ID in ProposedWrite table

    # Control flow
    awaiting_approval: bool
    error_message: str | None
    final_response: str | None # The final response to the user for this turn


# --- Tools ---

# Tool Input Schemas (Helps LLM call tools correctly)
class PineconeQueryInput(BaseModel):
    query: str = Field(description="The user's question or topic to search for in the text documents.")

class SQLQueryInput(BaseModel):
    sql_query: str = Field(description="A valid and safe PostgreSQL SELECT query to execute against the relevant table(s). Query ONLY the 'uploaded_tabular_data' table, filtering by 'upload_id' and 'row_index' or querying the 'row_data' JSONB column.")

class ProposeWriteInput(BaseModel):
    target_upload_id: int = Field(description="The specific ID of the tabular data upload to modify.")
    write_request: str = Field(description="A detailed natural language description of the data to be inserted, updated, or deleted. Include specific values and conditions.")


# Helper to get Neon Engine (cached per request or globally if safe)
_neon_engine = None
def get_neon_engine():
    global _neon_engine
    if _neon_engine is None:
        db_url = current_app.config['DATABASE_URL']
        if not db_url:
            raise ValueError("DATABASE_URL not configured")
        _neon_engine = create_engine(db_url)
    return _neon_engine

# Tool Implementations using @tool decorator

@tool("query_text_documents", args_schema=PineconeQueryInput)
def query_pinecone_tool(query: str, state: AgentState) -> Dict[str, Any]:
    """
    Searches relevant text documents based on the user's query.
    Use this to find information or answer questions based on the uploaded text files selected for this conversation.
    Only searches text documents selected for the current conversation.
    """
    print(f"--- Tool: query_pinecone_tool --- Query: {query}")
    text_upload_ids = state.get('text_upload_ids', [])
    if not text_upload_ids:
        return {"error": "No text documents (Pinecone sources) selected for this conversation."}

    pinecone_client = init_pinecone()
    if not pinecone_client:
        return {"error": "Pinecone client not initialized."}
    pinecone_index_name = current_app.config['PINECONE_INDEX_NAME']
    try:
        pinecone_index = pinecone_client.Index(pinecone_index_name)
        model = get_embedding_model()
        query_embedding = model.encode(query).tolist()

        all_results = []
        # Query each relevant namespace (one per selected text upload)
        for upload_id in text_upload_ids:
            namespace = f"upload-{upload_id}"
            print(f"Querying Pinecone index '{pinecone_index_name}', namespace '{namespace}'")
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=3, # Get top 3 results per document
                include_metadata=True,
                namespace=namespace
            )
            all_results.extend(results.get('matches', []))

        # Sort combined results by score (descending) and take top N overall
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_results = all_results[:5] # Limit overall results

        # Format results for the agent
        context_docs = []
        for match in top_results:
            metadata = match.get('metadata', {})
            text_chunk = metadata.get('text', 'N/A')
            doc_name = metadata.get('document_name', f"Upload {metadata.get('upload_id')}")
            score = match.get('score', 0)
            context_docs.append({
                "source": doc_name,
                "content": text_chunk,
                "score": float(score) # Ensure score is float
            })

        print(f"Found {len(context_docs)} relevant text chunks.")
        return {"retrieved_docs": context_docs}

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return {"error": f"Failed to query text documents: {e}"}


@tool("get_tabular_data_schema")
def get_tabular_schema_tool(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves the structure (column names and data types) of the selected tabular datasets.
    Call this *before* trying to generate a SQL query for tabular data.
    Provides schema for tables associated with the tabular uploads selected for this conversation.
    """
    print("--- Tool: get_tabular_schema_tool ---")
    tabular_upload_ids = state.get('tabular_upload_ids', [])
    if not tabular_upload_ids:
        return {"error": "No tabular data sources selected for this conversation."}

    schemas = {}
    engine = get_neon_engine()
    inspector = sql_inspect(engine)
    target_table = 'uploaded_tabular_data' # The single table holding all rows

    if not inspector.has_table(target_table):
         return {"error": f"Target table '{target_table}' not found in the database."}

    # Get columns of the main table
    columns = inspector.get_columns(target_table)
    schema_desc = f"Table '{target_table}' structure:\n"
    schema_desc += "- id (INTEGER, Primary Key): Unique row ID.\n"
    schema_desc += "- upload_id (INTEGER, Foreign Key to data_upload.id): Identifies the original upload.\n"
    schema_desc += "- row_index (INTEGER): Original row number from the uploaded CSV.\n"
    schema_desc += "- row_data (JSONB): Contains the actual row data as a JSON object (key-value pairs).\n"
    schema_desc += "- created_at (TIMESTAMPTZ): Timestamp of insertion.\n\n"
    schema_desc += "To query data for a specific upload, filter using 'upload_id'.\n"
    schema_desc += "To access specific fields from the original CSV, query the 'row_data' JSONB column, e.g., `row_data ->> 'column_name'`.\n\n"
    schema_desc += "Available tabular uploads for this conversation (use these upload_id values):\n"

    # Add info about which uploads are available
    uploads = db.session.query(DataUpload.id, DataUpload.filename)\
                        .filter(DataUpload.id.in_(tabular_upload_ids)).all()
    for upload_id, filename in uploads:
         schema_desc += f"- Upload ID: {upload_id}, Name: '{filename}'\n"
         # Store the same schema desc for all, as they use the same target table structure
         schemas[upload_id] = schema_desc

    # We return one schema description because all data is in one table,
    # but list the relevant upload IDs within it.
    # If you had separate tables per upload, you'd fetch schemas individually.
    if not schemas:
         return {"info": "No specific schemas found for the selected tabular uploads (they share the 'uploaded_tabular_data' structure).", "available_upload_ids": tabular_upload_ids}

    # Return the schema for the *first* relevant upload ID as representative
    first_upload_id = tabular_upload_ids[0]
    print(f"Returning schema description for table '{target_table}' relevant to uploads: {tabular_upload_ids}")
    # Store all schemas in state, return one description
    return {"tabular_schemas": schemas, "schema_description": schemas[first_upload_id]}


@tool("execute_sql_read_query", args_schema=SQLQueryInput)
def execute_sql_read_query_tool(sql_query: str, state: AgentState) -> Dict[str, Any]:
    """
    Executes a *read-only* (SELECT) SQL query against the database containing tabular data.
    Use this tool *only* after getting the schema with 'get_tabular_data_schema'.
    Ensure your query targets the 'uploaded_tabular_data' table and appropriately filters by 'upload_id' based on the available tabular uploads for this conversation.
    Do NOT use this for INSERT, UPDATE, or DELETE operations.
    """
    print(f"--- Tool: execute_sql_read_query_tool --- Query: {sql_query}")
    tabular_upload_ids = state.get('tabular_upload_ids', [])
    if not tabular_upload_ids:
        return {"error": "No tabular data sources selected for this conversation to query."}

    # Basic Security Check: Allow only SELECT queries on the target table
    upper_query = sql_query.strip().upper()
    if not upper_query.startswith("SELECT"):
        return {"error": "Invalid SQL: Only SELECT queries are allowed for reading data."}
    # Very basic check against modifying statements (can be bypassed, needs proper permissions/proxy)
    if any(keyword in upper_query for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]):
        return {"error": "Invalid SQL: Modifying statements (INSERT, UPDATE, DELETE, etc.) are not allowed."}
    # Check if it queries the correct table (basic check)
    # This might be too restrictive if joining is needed, but safer initially.
    # if "'UPLOADED_TABULAR_DATA'" not in upper_query.replace(" ",""):
    #    return {"error": "Invalid SQL: Query must target the 'uploaded_tabular_data' table."}

    engine = get_neon_engine()
    try:
        with engine.connect() as connection:
            # Execute in a transaction that's immediately rolled back (read-only insurance)
            with connection.begin():
                result_proxy = connection.execute(text(sql_query))
                results = [dict(row._mapping) for row in result_proxy.fetchall()] # Fetch results as list of dicts
                # Convert results to string for LLM (limit length)
                results_str = json.dumps(results, default=str, indent=2) # Handle dates etc.
                max_len = 2000
                if len(results_str) > max_len:
                     results_str = results_str[:max_len] + "\n... (results truncated)"
                print(f"SQL query executed successfully. Result preview: {results_str[:200]}...")
                connection.rollback() # Ensure no changes are committed
            return {"sql_results": results_str}
    except ProgrammingError as e:
         print(f"SQL Programming Error: {e}")
         return {"error": f"SQL Error: {e.orig}"} # Show the underlying DB error
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return {"error": f"Failed to execute SQL query: {e}"}


@tool("propose_sql_write_operation", args_schema=ProposeWriteInput)
def propose_sql_write_tool(target_upload_id: int, write_request: str, state: AgentState) -> Dict[str, Any]:
    """
    Generates a SQL INSERT, UPDATE, or DELETE statement based on the user's request to modify a specific tabular dataset.
    This tool DOES NOT execute the SQL. It prepares it for human review and approval.
    Use this only when the user explicitly asks to add, change, or remove data in one of the selected tabular uploads.
    You MUST specify the 'target_upload_id'.
    """
    print(f"--- Tool: propose_sql_write_tool --- Request: {write_request} for Upload ID: {target_upload_id}")
    user_id = state['user_id']
    conversation_id = state['conversation_id']
    tabular_upload_ids = state.get('tabular_upload_ids', [])

    if target_upload_id not in tabular_upload_ids:
        return {"error": f"Invalid request: Upload ID {target_upload_id} is not a selected tabular data source for this conversation."}

    # Get schema for context (needed by LLM to generate correct SQL)
    schema_info = get_tabular_schema_tool.__wrapped__(state=state) # Call the underlying function directly
    schema_desc = schema_info.get("schema_description")
    if not schema_desc:
         return {"error": "Could not retrieve schema information needed to generate SQL write statement."}

    # Use an LLM to generate the SQL based on the request and schema
    # **** THIS IS A CRITICAL STEP - NEEDS STRONG PROMPTING FOR SAFETY ****
    # The prompt MUST emphasize generating SQL ONLY for 'uploaded_tabular_data',
    # using the correct `upload_id`, and correctly manipulating the `row_data` JSONB field.
    # It should also handle INSERT (needs new row_index?), UPDATE, DELETE carefully.
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=f"""You are an expert PostgreSQL assistant. Your task is to generate a SINGLE safe SQL statement (INSERT, UPDATE, or DELETE) to modify data in the 'uploaded_tabular_data' table based on the user's request.

            IMPORTANT RULES:
            1.  ONLY generate SQL for the 'uploaded_tabular_data' table.
            2.  The target data upload ID is: {target_upload_id}. Your SQL MUST filter or specify this `upload_id`.
            3.  Data for each original row is stored in the `row_data` JSONB column. Use JSONB operators (e.g., `->>`, `jsonb_set`) to access or modify specific fields within `row_data`.
            4.  For INSERT: You need to determine the next `row_index` for the given `upload_id` (e.g., `SELECT COALESCE(MAX(row_index), -1) + 1 FROM uploaded_tabular_data WHERE upload_id = {target_upload_id}`). Create the full JSONB object for `row_data`.
            5.  For UPDATE: Use a WHERE clause filtering by `upload_id` and likely some condition on the `row_data` column (e.g., `WHERE upload_id = {target_upload_id} AND row_data ->> 'some_key' = 'some_value'`). Use `jsonb_set` to modify fields in `row_data`.
            6.  For DELETE: Use a WHERE clause filtering by `upload_id` and likely some condition on `row_data`.
            7.  Generate ONLY the SQL statement itself, enclosed in triple backticks (```sql ... ```).
            8.  Do NOT include explanations outside the backticks.
            9.  If the request is ambiguous or seems unsafe, respond with "Error: Request is ambiguous or potentially unsafe." inside the backticks.

            Schema Information:
            {schema_desc}
            """
        ),
        HumanMessage(content=f"User request for Upload ID {target_upload_id}: {write_request}")
    ])
    llm = ChatTogether(model=LLM_MODEL_NAME, temperature=0.0) # Low temp for precision
    chain = sql_generation_prompt | llm
    response = chain.invoke({})
    generated_sql = response.content.strip()

    # Basic validation of the generated SQL
    if not generated_sql.startswith("```sql") or not generated_sql.endswith("```"):
         error_msg = "LLM failed to generate SQL in the expected format."
         print(error_msg)
         print(f"LLM Response: {generated_sql}")
         # Try to extract anyway? Or just fail. Let's fail for safety.
         return {"error": error_msg, "raw_output": generated_sql}

    proposed_sql = generated_sql[len("```sql"): -len("```")].strip()

    # More safety checks (redundant with prompt but good practice)
    upper_sql = proposed_sql.upper()
    allowed_keywords = ["INSERT", "UPDATE", "DELETE"]
    if not any(upper_sql.startswith(keyword) for keyword in allowed_keywords):
        return {"error": f"Generated SQL is not an INSERT, UPDATE, or DELETE statement: {proposed_sql}"}
    # Very basic check for target table (can be fooled)
    # if "UPLOADED_TABULAR_DATA" not in upper_sql:
    #     return {"error": f"Generated SQL does not appear to target 'uploaded_tabular_data': {proposed_sql}"}
    # Check if upload_id seems present (very basic)
    if f"UPLOAD_ID = {target_upload_id}" not in upper_sql.replace(" ", ""):
        return {"error": f"Generated SQL might be missing the required filter for 'upload_id = {target_upload_id}': {proposed_sql}"}

    # Generate a description of the proposed SQL (using LLM again)
    desc_prompt = ChatPromptTemplate.from_messages([
        SystemMessage("Briefly explain what the following SQL statement does in simple terms, focusing on the user's intended action."),
        HumanMessage(f"SQL Statement:\n```sql\n{proposed_sql}\n```")
    ])
    desc_chain = desc_prompt | llm
    desc_response = desc_chain.invoke({})
    sql_description = desc_response.content

    # Store the proposed write in the database
    try:
        proposal = ProposedWrite(
            conversation_id=conversation_id,
            user_id=user_id,
            target_upload_id=target_upload_id,
            proposed_sql=proposed_sql,
            description=sql_description,
            status='pending'
            # triggering_message_id might be set later if needed
        )
        db.session.add(proposal)
        db.session.commit()
        proposal_id = proposal.id
        print(f"Stored proposed write with ID: {proposal_id}")

        # Return necessary info for the agent to inform the user
        return {
            "proposed_write_id": proposal_id,
            "proposed_write_sql": proposed_sql,
            "proposed_write_description": sql_description,
            "status": "pending_approval" # Signal to agent flow
        }
    except Exception as e:
        db.session.rollback()
        print(f"Error storing proposed write: {e}")
        return {"error": f"Failed to store the proposed write operation: {e}"}


# List of tools for the agent
tools = [
    query_pinecone_tool,
    get_tabular_schema_tool,
    execute_sql_read_query_tool,
    propose_sql_write_tool,
]

# --- LangGraph Nodes ---

def get_conversation_context(state: AgentState):
    """Loads history and identifies available data types."""
    print("--- Node: get_conversation_context ---")
    conv_id = state['conversation_id']
    conversation = db.session.get(Conversation, conv_id)
    if not conversation:
        raise ValueError(f"Conversation {conv_id} not found")

    # Load history
    state['chat_history'] = conversation.get_langchain_history()

    # Get selected uploads and determine available types
    selected_uploads = conversation.data_sources # Assuming lazy='subquery' or similar works
    selected_ids = [ds.id for ds in selected_uploads]
    state['selected_upload_ids'] = selected_ids
    available_types = set(ds.data_type for ds in selected_uploads)
    state['available_data_types'] = available_types
    state['tabular_upload_ids'] = [ds.id for ds in selected_uploads if ds.data_type == 'tabular']
    state['text_upload_ids'] = [ds.id for ds in selected_uploads if ds.data_type == 'text']

    print(f"Context loaded for Conv {conv_id}. History length: {len(state['chat_history'])}. Selected Uploads: {selected_ids}. Types: {available_types}")
    # Add user query to history for the LLM call
    state['chat_history'].append(HumanMessage(content=state['user_query']))
    return state


def agent_router(state: AgentState):
    """Routes the conversation turn to the appropriate tool or generates a response."""
    print("--- Node: agent_router ---")
    messages = state['chat_history']
    available_tools = []
    tool_schemas = [] # For function calling

    # Dynamically enable tools based on selected data
    if 'text' in state['available_data_types']:
        available_tools.append(query_pinecone_tool)
    if 'tabular' in state['available_data_types']:
        available_tools.append(get_tabular_schema_tool)
        available_tools.append(execute_sql_read_query_tool)
        available_tools.append(propose_sql_write_tool) # Enable write proposal if tabular selected

    if not available_tools:
         print("No data sources selected or no tools available. Generating direct response.")
         return "generate_response"

    # Use function calling / tool calling LLM
    llm_with_tools = ChatTogether(model=LLM_MODEL_NAME, temperature=0).bind_tools(available_tools)

    # Construct prompt with available tools description if needed, or rely on bind_tools
    # system_message = f"You are a helpful assistant. Access data using the available tools based on the user query and conversation history. Available data types: {state['available_data_types']}"
    # messages_for_llm = [SystemMessage(content=system_message)] + messages

    print("Routing query using LLM with tools...")
    ai_message = llm_with_tools.invoke(messages) # Pass history including latest user query

    # Add the AI message (potentially with tool calls) to history
    state['chat_history'].append(ai_message)

    if not ai_message.tool_calls:
        print("LLM decided to respond directly.")
        return "generate_response" # No tool called, generate final response

    # Check if a write was proposed
    # We need to differentiate between proposing a write and other tool calls
    # Let's assume propose_sql_write_tool was called if its output indicates success
    if ai_message.tool_calls[0]['name'] == propose_sql_write_tool.name:
         print("LLM called propose_sql_write_tool.")
         # The tool node will execute, and its result will signal pending_approval
         # We route directly to the tool node here.
         return "tools" # Route to the ToolNode

    # Check if get_tabular_schema was called AND sql_results are NOT present yet
    # This suggests we need schema before generating/executing SQL read
    called_get_schema = any(call['name'] == get_tabular_schema_tool.name for call in ai_message.tool_calls)
    has_sql_results = state.get('sql_results') is not None
    if called_get_schema and not has_sql_results:
        print("LLM called get_tabular_schema_tool. Routing to tools.")
        # We expect the next router call (after tool execution) to call execute_sql_read_query
        return "tools"

    # Otherwise, assume it's a read query (Pinecone or SQL read)
    print(f"LLM called tool(s): {[call['name'] for call in ai_message.tool_calls]}")
    return "tools" # Route to the ToolNode to execute other tools

# ToolNode handles executing the called tools
tool_node = ToolNode(tools)


def handle_tool_result(state: AgentState):
    """Processes the output of the tool node, updates state, and decides next step."""
    print("--- Node: handle_tool_result ---")
    # Get the last message, which should be the ToolMessage
    last_message = state['chat_history'][-1]
    if not isinstance(last_message, ToolMessage):
        print("Warning: Expected last message to be ToolMessage, routing to response.")
        # Maybe the tool failed silently? Or router sent here incorrectly.
        return "generate_response" # Fallback

    tool_output = json.loads(last_message.content) # ToolNode typically returns JSON string
    tool_name = last_message.additional_kwargs.get("name", "") # Get the name of the tool that ran

    print(f"Handling result from tool: {tool_name}")
    print(f"Tool Output: {tool_output}")

    if tool_output.get("error"):
        state["error_message"] = f"Error from {tool_name}: {tool_output['error']}"
        print(f"Error detected from tool {tool_name}. Routing to generate_response.")
        # Add error info to history for the final response generation?
        # state['chat_history'].append(SystemMessage(content=f"Error encountered: {state['error_message']}"))
        return "generate_response" # Let the LLM explain the error

    # Update state based on tool output
    if tool_name == query_pinecone_tool.name:
        state["retrieved_docs"] = tool_output.get("retrieved_docs")
        print("Updated state with retrieved_docs.")
    elif tool_name == get_tabular_schema_tool.name:
        state["tabular_schemas"] = tool_output.get("tabular_schemas")
        # Add schema description to history for context for next LLM call (SQL generation)
        state["chat_history"].append(SystemMessage(content=f"Schema Information:\n{tool_output.get('schema_description')}"))
        print("Updated state with tabular_schemas. Returning to router.")
        # Go back to router to potentially call execute_sql_read_query_tool next
        return "__continue__" # Use special value to loop back to router immediately
    elif tool_name == execute_sql_read_query_tool.name:
        state["sql_results"] = tool_output.get("sql_results")
        print("Updated state with sql_results.")
    elif tool_name == propose_sql_write_tool.name:
        state["proposed_write_id"] = tool_output.get("proposed_write_id")
        state["proposed_write_sql"] = tool_output.get("proposed_write_sql")
        state["proposed_write_description"] = tool_output.get("proposed_write_description")
        if tool_output.get("status") == "pending_approval":
             print("Write proposed and pending approval. Routing to handle_approval.")
             return "handle_approval" # Go to the approval node
        else:
             # Should not happen if tool returns correctly
             state["error_message"] = "Write proposal tool finished unexpectedly."
             print(state["error_message"])

    # After successful tool execution (except get_schema), proceed to generate response
    print("Routing to generate_response after handling tool result.")
    return "generate_response"


def generate_final_response(state: AgentState):
    """Generates the final response to the user using the LLM."""
    print("--- Node: generate_final_response ---")
    context = ""
    if state.get("retrieved_docs"):
        context += "Relevant information from text documents:\n"
        for doc in state["retrieved_docs"][:3]: # Limit context length
             context += f"- Source: {doc.get('source')}\n  Content: {doc.get('content')[:200]}...\n" # Truncate content
        context += "\n"
    if state.get("sql_results"):
         context += f"Results from tabular data query:\n{state['sql_results']}\n\n"

    # Handle case where an error occurred
    if state.get("error_message"):
        context += f"Note: An error occurred during processing: {state['error_message']}\n\n"

    # Construct prompt for final response generation
    system_message = "You are a helpful assistant. Synthesize the information from the conversation history and any retrieved data context (provided below) to answer the user's last query. If a write operation was proposed, inform the user it requires approval. If an error occurred, explain it clearly."
    if context:
         final_prompt = ChatPromptTemplate.from_messages([
             SystemMessage(content=system_message),
             # History already includes latest user query and previous AI turns/tool calls
             # Add retrieved context explicitly if needed, or assume LLM sees history
             SystemMessage(content=f"Retrieved Context:\n{context}"),
         ])
    else:
         final_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_message)])


    llm = ChatTogether(model=LLM_MODEL_NAME, temperature=0.7)
    # Combine context prompt with existing history
    messages_for_llm = final_prompt.format_messages() + state['chat_history']

    print("Generating final response...")
    final_response_msg = llm.invoke(messages_for_llm)
    state["final_response"] = final_response_msg.content
    state['chat_history'].append(final_response_msg) # Add final AI response to history
    print(f"Final response generated: {state['final_response'][:200]}...")
    return state


def handle_approval_request(state: AgentState):
    """Handles the state when a write requires approval."""
    print("--- Node: handle_approval_request ---")
    state['awaiting_approval'] = True
    proposal_id = state['proposed_write_id']
    sql_desc = state['proposed_write_description']
    # Generate a message informing the user about the approval needed
    approval_message = f"I have generated the following SQL statement to fulfill your request:\n```sql\n{state['proposed_write_sql']}\n```\nDescription: {sql_desc}\n\nThis requires your approval before execution. Please review proposed write action ID {proposal_id}."
    state['final_response'] = approval_message
    state['chat_history'].append(AIMessage(content=approval_message))
    print(f"State set to awaiting_approval=True for Proposal ID {proposal_id}. Graph run will pause.")
    # The graph run effectively ends here, waiting for external approval trigger.
    return state

def should_continue(state: AgentState):
    """Determines the next step after the router or tool execution."""
    # This logic is largely replaced by the direct routing return values
    # from agent_router and handle_tool_result ("generate_response", "tools", "handle_approval", "__continue__")
    # However, we can use it as a final condition check.
    if state.get("error_message"):
        print("Conditional Edge: Error detected, ending.")
        return END # Or route to a specific error handling node?
    if state.get("awaiting_approval"):
        print("Conditional Edge: Awaiting approval, ending.")
        return END # Stop the flow here
    # If the router decided on tools, we go to the tool node
    # If the router decided on response, or tool finished, we go to generate response
    # Logic moved into router/handler return values.
    # Fallback case - should ideally not be reached if router/handler work.
    last_message = state['chat_history'][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
         return "tools" # Default to calling tools if AI proposed them
    else:
         return "generate_response" # Default to generating response


# --- Build the Graph ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("get_context", get_conversation_context)
workflow.add_node("agent_router", agent_router)
workflow.add_node("tool_node", tool_node) # Renaming 'tools' node
workflow.add_node("handle_tool_result", handle_tool_result)
workflow.add_node("generate_response", generate_final_response)
workflow.add_node("handle_approval", handle_approval_request)

# Define edges
workflow.set_entry_point("get_context")
workflow.add_edge("get_context", "agent_router")

# Conditional routing after the agent decides
workflow.add_conditional_edges(
    "agent_router",
    lambda x: x['chat_history'][-1].tool_calls is not None, # Simple check if tools were called
    {
        True: "tool_node", # If tools called, execute them
        False: "generate_response", # Otherwise, generate response directly
    }
)

# Routing after tools are executed
workflow.add_conditional_edges(
    "tool_node",
    handle_tool_result, # This node now returns the next step string
    {
        "handle_approval": "handle_approval",
        "generate_response": "generate_response",
        "__continue__": "agent_router", # Loop back to router (e.g., after get_schema)
        # Add error handling route?
    }
)

# Edges leading to the end
workflow.add_edge("generate_response", END)
workflow.add_edge("handle_approval", END) # Stops here, waiting for external trigger


# Compile the graph
# Use the checkpointer defined earlier
# Note: Checkpointing is CRUCIAL if you want the 'awaiting_approval' state to persist
app_graph = workflow.compile(checkpointer=memory)

# --- Function to Run Graph (called by Flask route/Celery task) ---
def run_agent_turn(conversation_id: int, user_query: str):
    """Runs one turn of the conversation through the LangGraph agent."""
    user = current_user # Assuming Flask-Login context is available
    initial_state = AgentState(
        conversation_id=conversation_id,
        user_id=user.id,
        user_query=user_query,
        # Other fields will be populated by the graph nodes
        selected_upload_ids=[],
        available_data_types=set(),
        tabular_upload_ids=[],
        text_upload_ids=[],
        chat_history=[],
        awaiting_approval=False
    )

    # Use conversation_id as the thread_id for checkpointing state per conversation
    config = {"configurable": {"thread_id": f"conv_{conversation_id}"}}

    # Add user message to DB first
    user_msg = ChatMessage(conversation_id=conversation_id, role='user', content=user_query)
    db.session.add(user_msg)
    db.session.commit() # Commit user message

    final_state_response = None
    try:
        # Stream or Invoke
        # Use stream to potentially see intermediate steps
        print(f"\n--- Running Agent for Conv {conversation_id} ---")
        events = app_graph.stream(initial_state, config=config)
        for event in events:
            # You can inspect events here for debugging
            # print(f"Event Type: {event['event']}, Node: {event.get('name')}, Data Keys: {list(event.get('data', {}).get('output', {}).keys())}")
            # Capture the *output* of the final nodes that produce the response
             if event['event'] == 'on_chain_end':
                 node_name = event.get("name")
                 output = event.get('data', {}).get('output')
                 if isinstance(output, dict):
                     if node_name == "generate_response" and output.get("final_response"):
                         final_state_response = output.get("final_response")
                         print("Captured final_response from generate_response node.")
                         break # Got the response
                     elif node_name == "handle_approval" and output.get("final_response"):
                          final_state_response = output.get("final_response")
                          print("Captured final_response from handle_approval node.")
                          break # Got the approval message


        print(f"--- Agent Run Finished for Conv {conversation_id} ---")

        # Retrieve the final state from the checkpointer if needed, though response is captured above
        # final_state = app_graph.get_state(config)
        # final_response = final_state.values.get("final_response", "Agent finished, but no response captured.")

        if not final_state_response:
             # Fallback if streaming didn't capture it correctly
             final_state = app_graph.get_state(config)
             final_state_response = final_state.values.get("final_response", "Agent finished, but no response captured.")
             print(f"Fallback: Captured final_response from final state: {final_state_response[:100]}...")


        # Add agent response message to DB
        agent_msg = ChatMessage(
            conversation_id=conversation_id,
            role='agent',
            content=final_state_response
        )
        # Link message to proposal if one was generated in this turn
        final_state = app_graph.get_state(config) # Get latest state
        proposed_write_id = final_state.values.get("proposed_write_id")
        if proposed_write_id:
             # Find the proposal we just created
             proposal = db.session.get(ProposedWrite, proposed_write_id)
             if proposal:
                 agent_msg.proposed_write_id = proposed_write_id
                 # Also link the proposal back to this agent message
                 proposal.proposing_message_id = agent_msg.id # Need to commit msg first to get ID?
                 # Commit agent message first, then update proposal
                 db.session.add(agent_msg)
                 db.session.commit() # Commit agent message to get its ID
                 proposal.proposing_message_id = agent_msg.id
                 db.session.commit() # Commit updated proposal
             else: # Proposal not found? Should not happen.
                 db.session.add(agent_msg)
                 db.session.commit()
        else:
            db.session.add(agent_msg)
            db.session.commit() # Commit agent message

        return final_state_response

    except Exception as e:
        print(f"!!! LangGraph execution failed for Conv {conversation_id} !!!")
        import traceback
        traceback.print_exc()
        error_response = f"An error occurred while processing your request: {e}"
        # Add error message to chat
        agent_msg = ChatMessage(conversation_id=conversation_id, role='agent', content=error_response)
        db.session.add(agent_msg)
        db.session.commit()
        return error_response