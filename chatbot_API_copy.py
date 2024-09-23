import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from langchain_community.document_loaders import JSONLoader
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()

# Setup FastAPI

class QueryRequest(BaseModel):
    query: str
    session_id: str

embedding_model = OpenAIEmbeddings()

# når du kører første gang til persist:
#vectorstore = Chroma.from_documents(documents=splits,
#                                    embedding=OpenAIEmbeddings(),
#                                    persist_directory = "vector_storage")

#vectorstore.persist()

vectorstore = Chroma(persist_directory='vector_storage3',
                     embedding_function=embedding_model)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

memory = MemorySaver()
doc_prompt = PromptTemplate.from_template(
    "<context>\n{page_content}\n\n<meta>\nSequence nummer: {seq_num}\nDokument titel: {title}\nURL: {url}\n</meta>\n</context>"
)
tool = create_retriever_tool(
    retriever,
    "vidensportal_retriever",
    "Henter viden og materialer om el og vvs-branchen",
    document_prompt=doc_prompt
    )
tools = [tool]

prompt = """Du er en assistent, som hjælper med at svare på spørgsmål om el og vvs-branchen.

Du har et værktøj til rådighed, som du skal benytte dig af ved nye spørgsmål. Ved opfølgende spørgsmål,
som du allerede har i din kontekst, behøver du ikke genkalde værktøjet.

Når du benytter dig af værktøjet, så inkluder dokument titel og url til de dele som du bruger til at svare.

Brug de relevante oplysninger fra de hentede dokumenter til at besvare spørgsmålet.
Hvis du ikke kender svaret, sig at du ikke kender svaret.
Hvis du ikke forstår spørgsmålet, så spørg uddybende spørgsmål for at finde ud af, hvad brugeren gerne vil vide.

Formater dit svar ved hjælp af Markdown syntax:
- Brug '**' for fed skrift.
- Brug '*' for *kursiv* tekst.
- Brug '#', '##', '###' osv. for overskrifter.
- Brug '-' eller '*' for punktopstillinger.
- Brug '1.', '2.', '3.' osv. for nummererede lister.
- Brug '`' for inline kode og '```' for kodeblokke.

Sørg for at strukturere dit svar med god brug af afsnit, overskrifter og lister for at gøre det let at læse.

Du skal ikke skrive spørgsmålet øverst i dit svar - du kan bare begynde at svare direkte. 

Svar:"""

# Create the agent executor
agent_executor = create_react_agent(llm, tools, checkpointer=memory, state_modifier=prompt)

@app.post("/process_query")
async def process_query(request: QueryRequest):
    # Prepare the query for the agent
    query = request.query
    config = {"configurable": {"thread_id": request.session_id}}

    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    print(response)

    return {
        "message": "Query processed",
        "response": response["messages"][-1].content,
        "session_id": request.session_id
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)