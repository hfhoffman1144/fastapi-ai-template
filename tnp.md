LangGraph is a versatile Python library designed for stateful, cyclic, and multi-actor Large Language Model (LLM) applications. LangGraph builds upon its parent library, [LangChain](https://realpython.com/build-llm-rag-chatbot-with-langchain/), and allows you to build sophisticated workflows that are capable of handling the complexities of real-world LLM applications.

**In this tutorial, you'll learn:**

- **What LangGraph is and how it expands upon LangChain's capabilities**
- **How to build LLM workflows in LangGraph**
- **How to build LLM agents in LangGraph**

You'll learn all of this hand-on by building a LangGraph agent that can autonomously parse emails, send emails, and interact with API services. 

While you'll get a brief primer on `LangChain` in this tutorial, you'll benefit from having prior knowledge of LangChain fundamentals. You'll also want to ensure you have intermediate Python knowledge - specifically in [object-oriented programming](https://realpython.com/python3-object-oriented-programming/) concepts like classes and methods.

## Install LangGraph

LangGraph is available on [PyPI](https://pypi.org/), and you can install it with [pip](https://realpython.com/what-is-pip/). Open a terminal or command prompt, create a new virtual environment, and then run the following command to install LangGraph:

```console
(venv) $ python -m pip install langgraph
```

This command will install the latest version of LangGraph from PyPI onto your machine. To verify that the installation was successful, start a [Python REPL](https://realpython.com/python-repl/) and import LangGraph:

```pycon
>>> import langgraph
```

If the import runs without error, then you've successfully installed LangGraph. You'll also need a few more libraries for this tutorial:

```console
(venv) $ python -m pip install langchain-openai "pydantic[email]"
```

You'll use `langchain-openai` to interact with OpenAI LLMs, but keep in mind you can use any LLM provider you like with LangGraph and LangChain. You'll use [`pydantic`](https://realpython.com/python-pydantic/) to validate the information your agent parses from emails.

Before moving forward, if you choose to use OpenAI, make sure you're signed up for an OpenAI account and you have a valid [API key](https://openai.com/api/). You'll need to set the following [environment variable](https://en.wikipedia.org/wiki/Environment_variable) before running any examples in this tutorial:

```dotenv
OPENAI_API_KEY=<YOUR-OPENAI-API-KEY>
```

Note that while LangGraph was made by the creators of LangChain, and the two libraries are highly compatible, it is possible to use LangGraph without LangChain. However, it's more common to use LangChain and LangGraph together, and you'll see throughout this tutorial how they compliment each other. 

With that, you've installed all the dependencies you'll need for this tutorial, and you're ready to create your LangGraph email processor. You'll start by going through an overview of LangChain chains and exploring LangGraph's core concept - the state graph.

## State Graphs

As you might have inferred from the name, LangGraph is all about implementing LLM applications as directed graphs. You can think of a directed graph as a sequence of instructions, composed of nodes and edges, that tell you how to complete a task. In LangGraph, nodes represent actions that your graph can take, such as calling a function, and edges tell you which node to go to next. 

To understand this better, take a look at this directed graph:

{% img 'example-directed-graph-2' centered=True caption="A directed graph example"%}

This graph models what you might do when you eat a meal in a cafeteria. It consists of two actions represented by nodes - **Buy Food** and **Eat Food**. Once you've eaten your food you ask yourself **Am I still hungry or am I full?** You can think of the dotted arrows, which are edges, as the answer to this question. If you're hungry, you buy more food and eat it. This cycle continues until you're full at which point you leave the cafeteria. 

This simple example illustrates the essence of how LangGraph represents and implements LLM applications. In this tutorial, you're going to step into the shoes of an AI engineer at a large real estate development firm and build a graph to process emails from regulatory agencies. Your graph will:

- Extract structured fields like dates, names, phone numbers, and locations from email messages
- Notify internal stakeholders if an email requires immediate escalation
- Create tickets with your company's legal team using the information extracted from the email
- Forward and reply to emails that were sent to the wrong address

To understand why LangGraph is a great choice for building such an application, you'll begin this tutorial by reviewing and building a LangChain chain, and you'll see why chains can't accomplish the tasks listed above.

### LangChain Chains and Their Limitations

Suppose you work for a large real estate development company. Your company receives hundreds of emails a day from regulatory entities and other organizations regarding active construction sites. For instance, your company might receive a notice from an inspector saying that a construction site doesn't comply with safety regulations. 

Your job is to build a tool that can read these emails, extract critical information from them, and notify the correct internal team who will take action. The first step you'll take to accomplish this is to build a LangChain chain that uses an LLM to extract structured fields from a regulatory notice email. You start by defining a Pydantic `BaseModel` that describes all the fields you want to extract from an email:

```python filename="chains/notice_extraction.py"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, EmailStr, Field
from datetime import date

class NoticeEmailExtract(BaseModel):
    date_of_notice: date | None = Field(
        default=None,
        description="""The date of the notice (if any) reformatted
        to match YYYY-mm-dd""",
    )
    entity_name: str | None = Field(
        default=None,
        description="""The name of the entity sending the notice (if present
        in the message)""",
    )
    entity_phone: str | None = Field(
        default=None,
        description="""The phone number of the entity sending the notice
        (if present in the message)""",
    )
    entity_email: EmailStr | None = Field(
        default=None,
        description="""The email of the entity sending the notice
        (if present in the message)""",
    )
    project_id: int | None = Field(
        default=None,
        description="""The project ID (if present in the message) -
        must be an integer""",
    )
    site_location: str | None = Field(
        default=None,
        description="""The site location of the project (if present
        in the message)""",
    )
    violation_type: str | None = Field(
        default=None,
        description="""The type of violation (if present in the
        message)""",
    )
    required_changes: str | None = Field(
        default=None,
        description="""The required changes specified by the entity
        (if present in the message)""",
    )
    compliance_deadline: date | None = Field(
        default=None,
        description="""The date that the company must comply (if any)
        reformatted to match YYYY-mm-dd""",
    )
    max_potential_fine: float | None = Field(
        default=None, description="""The maximum potential fine
        (if any)"""
    )
```

You first import all of the dependencies you'll need to create your chain, and you define `NoticeEmailExtract` which is a Pydantic `BaseModel` that provides type definitions and descriptions of each field you want to extract. Downstream, LangChain will pass the information in the `NoticeEmailExtract` definition to an LLM as raw text. The LLM will try to extract these fields from an email based on the type hints and description parameters in `Field()`.

As an example, the LLM will try to identify and extract the project ID corresponding to the construction site discussed in the email. If successfully extracted, the LLM will return the project ID in a [JSON](https://realpython.com/python-json/) object with an integer `project_id` entry. If it's unable to extract a project ID, the `project_id` entry will be `None`. 

Next, you create a chain to parse notice emails using `NoticeEmailExtract`:

```python filename="chains/notice_extraction.py"
# ...

info_parse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Parse the date of notice, sending entity name, sending entity
            phone, sending entity email, project id, site location,
            violation type, required changes, compliance deadline, and
            maximum potential fine from the message. If any of the fields
            aren't present, don't populate them. Try to cast dates into
            the YYYY-mm-dd format. Don't populate fields if they're not
            present in the message.
            
            Here's the notice message:
            
            {message}
            """,
        )
    ]
)

notice_parser_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

NOTICE_PARSER_CHAIN = (
    info_parse_prompt
    | notice_parser_model.with_structured_output(NoticeEmailExtract)
)
```

You define `info_parse_prompt` to tell the LLM how it should behave and which tasks it should accomplish. In this case, you're instructing it to extract structured fields from an email notice. The `{message}` parameter is a placeholder that will store the email you pass through the chain. 

You then instantiate a `ChatOpenAI` model using `gpt-4o-mini` as the foundation model - you can experiment with different LLM providers and models here. Lastly, you instantiate `NOTICE_PARSER_CHAIN` using the [LangChain Expression Language (LCEL)](https://realpython.com/build-llm-rag-chatbot-with-langchain/#chains-and-langchain-expression-language-lcel). 

This statement creates a chain that injects an email into the `message` parameter of `info_parse_prompt`. It then passes the output of `info_parse_prompt` to `notice_parser_model`. By calling `.with_structured_output(NoticeEmailExtract)`, LangChain converts your `NoticeEmailExtract` base model to a prompt that tells the LLM to adhere it's output to the schema defined by `NoticeEmailExtract`.

{% alert %}

**Note:** You may have noticed that `NOTICE_PARSER_CHAIN` is in all caps. This is intentional because `NOTICE_PARSER_CHAIN` will act as a [global variable](https://realpython.com/python-use-global-variable-in-function/) that you'll use in other functions later in this tutorial, and it's considered a best practice to define global variables in all caps. 

{% endalert %}

To bring this all together, open a Python interpreter and test `NOTICE_PARSER_CHAIN` on an example email notice:


```pycon
>>> from chains.notice_extraction import NOTICE_PARSER_CHAIN
 
>>> example_notice = """
# ... Date: October 15, 2024
# ... From: Occupational Safety and Health Administration (OSHA)
# ... To: Blue Ridge Construction, project 111232345 - Downtown Office Complex
# ...
# ... During a recent inspection of your construction site at 123 Main Street,
# ... the following safety violations were identified:
# ...
# ... Lack of fall protection: Workers on scaffolding above 10 feet were
# ... observed without required harnesses or other fall protection equipment.
# ... Unsafe scaffolding setup: Several scaffolding structures were noted as
# ... lacking secure base plates and bracing, creating potential collapse risks.
# ... Inadequate personal protective equipment (PPE): Multiple workers were
# ... found without proper PPE, including hard hats and safety glasses.
# ... Required Corrective Actions:
# ...
# ... Install guardrails and fall arrest systems on all scaffolding over 10
# ... feet. Conduct an inspection of all scaffolding structures and reinforce
# ... unstable sections. Ensure all workers on-site are provided with
# ... necessary PPE and conduct safety training on proper usage. Deadline
# ... for Compliance: All violations must be rectified by November 10, 2024.
# ... Failure to comply may result in fines of up to $25,000 per violation.
# ...
# ... Contact: For questions or to confirm compliance, please reach out to the
# ... OSHA regional office at (555) 123-4567 or email compliance.osha@osha.gov.
# ... """

>>> NOTICE_PARSER_CHAIN.invoke({"message":example_notice})
NoticeEmailExtract(
    date_of_notice=datetime.date(2024, 10, 15),
    entity_name='Occupational Safety and Health Administration (OSHA)',
    entity_phone='(555) 123-4567',
    entity_email='compliance.osha@osha.gov',
    project_id=111232345,
    site_location='123 Main Street',
    violation_type='Safety violations',
    required_changes='Install guardrails and fall arrest systems on
      all scaffolding over 10 feet. Conduct an inspection of all
      scaffolding structures and reinforce unstable sections.
      Ensure all workers on-site are provided with necessary
      PPE and conduct safety training on proper usage.',
    compliance_deadline=datetime.date(2024, 11, 10),
    max_potential_fine=25000.0
)
```

Here, you import `NOTICE_PARSER_CHAIN` and pass `example_notice` to `NOTICE_PARSER_CHAIN.invoke()`. `NOTICE_PARSER_CHAIN` successfully parses the email and returns a `NoticeEmailExtract`. It's pretty awesome that `NOTICE_PARSER_CHAIN`, and specifically `gpt-4o-mini`, pulled these fields out of raw text. Think about how difficult it would be to write logic to do this without an LLM!

Not only did `NOTICE_PARSER_CHAIN` extract these fields, it did so with high accuracy. For example, `NOTICE_PARSER_CHAIN` extracted the date the notice was received, the entity that sent it, and even the maximum potential fine for non-compliance. Notice how all of the extracted fields are the correct data type that you specified when defining `NoticeEmailExtract` - all of this without having to write a single line of type conversion logic.

Next, you'll build another chain that you'll use throughout this tutorial. This chain will check whether the email notice requires escalation within the company based on a text description of what constitutes escalation. For instance, you might want to escalate a message if employees are in danger or if the notice warns about a fine above a specified threshold. Here's what the escalation chain looks like:

```python filename="chains/escalation_check.py"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class EscalationCheck(BaseModel):
    needs_escalation: bool = Field(
        description="""Whether the notice requires escalation
        according to specified criteria"""
    )


escalation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Determine whether the following notice received
            from a regulatory body requires immediate escalation.
            Immediate escalation is required when {escalation_criteria}.
            
            Here's the notice message:
            
            {message}
            """,
        )
    ]
)

escalation_check_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ESCALATION_CHECK_CHAIN = (
    escalation_prompt
    | escalation_check_model.with_structured_output(EscalationCheck)
)
```

In this block, `ESCALATION_CHECK_CHAIN` accepts a `message` and `escalation criteria` as parameters and returns a `BaseModel` with a single boolean attribute called `needs_escalation`. Here's what `ESCALATION_CHECK_CHAIN` looks like in action:

```pycon
>>> from chains.escalation_check import ESCALATION_CHECK_CHAIN

>>> escalation_criteria = """There is currently water damage
# ... or potential water damage reported"""

>>> message = """Several cracks in the foundation have
been identified along with water leaks"""
 
>>> ESCALATION_CHECK_CHAIN.invoke(
# ... {"message":message, "escalation_criteria":escalation_criteria}
# ... )
EscalationCheck(needs_escalation=True)

>>> message = "The wheel chair ramps are too steep"
 
>>> ESCALATION_CHECK_CHAIN.invoke(
# ... {"message":message, "escalation_criteria":escalation_criteria}
# ... )
EscalationCheck(needs_escalation=False)
```

You first import `ESCALATION_CHECK_CHAIN` and define the escalation criteria. Any messages that mention water damage require escalation. The first message mentions water leaks, and `ESCALATION_CHECK_CHAIN` correctly identifies that this meets the criteria and returns `EscalationCheck(needs_escalation=True)`. The second message doesn't mention water damage, and `ESCALATION_CHECK_CHAIN` returns `EscalationCheck(needs_escalation=False)`.

You now have the first two components of your email parsing system built, but in isolation, `NOTICE_PARSER_CHAIN` and `ESCALATION_CHECK_CHAIN` don't exactly solve your problems. You want your system to take different actions depending on whether the email requires escalation. You also might want to check if the email even comes from a regulatory body and forward it to the correct department if it doesn't. 

This is where chains reach their limit. Chains aren't suited for handling state and making conditional decisions like which action to take depending on whether an email requires escalation. To accomplish all of the tasks discussed previously, you'll need something more than a stateless chain that passes data linearly from one step to another. This is where LangGraph's core object, the State Graph, comes in to help.

### Build Your First State Graph

Now that you've built the notice parsing and escalation check chains, you need to orchestrate them and add additional functionality that your company requires to process notice emails. To do this, you'll use LangGraph's `StateGraph` to create a graph that builds upon `NOTICE_PARSER_CHAIN` and `ESCALATION_CHECK_CHAIN`. Start by initializing your graph:

```python filename="graphs/notice_extraction.py"
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import EmailStr
from chains.notice_extraction import NoticeEmailExtract, NOTICE_PARSER_CHAIN
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from chains.binary_questions import BINARY_QUESTION_CHAIN
from utils.graph_utils import send_escalation_email, create_legal_ticket

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


class GraphState(TypedDict):
    notice_message: str
    notice_email_extract: NoticeEmailExtract | None
    escalation_text_criteria: str
    escalation_dollar_criteria: float
    requires_escalation: bool
    escalation_emails: list[EmailStr] | None
    follow_ups: dict[str, bool] | None
    current_follow_up: str | None

workflow = StateGraph(GraphState)
```

You first import dependencies and configure a logger - notice that you import the chains you built previously. You then define `GraphState` -  a [typed dict](https://peps.python.org/pep-0589/) that defines the information each node in your graph updates and passes to the next node. Here's what each field in `GraphState` stores:

- **notice_message**: The notice email that you want to parse and process.
- **notice_email_extract**: A `NoticeEmailExtract` instance which is the output of running `notice_message` through `NOTICE_PARSER_CHAIN`. When you initialize the graph, `notice_email_extract` is `None`.
- **escalation_text_critera**: A text description of how to determine whether an email notice requires immediate escalation. 
- **escalation_dollar_amount**: The smallest maximum potential fine used to determine whether escalation is needed.
- **requires_escalation**: Whether the notice requires escalation.
- **escalation_emails**: A list of emails to notify if escalation is required.
- **follow_ups**: A dictionary that stores follow-up questions that your graph needs to answer about the notice message before creating a legal ticket. You'll learn more about this in the next section.
- **current_follow_up**: The current follow-up question your graph needs to answer.

After defining `GraphState` you initialize a `StateGraph` instance and assign it to the variable `workflow`. At this point, `workflow` is an empty graph that can't do anything. To make `workflow` functional, you need to add nodes and edges. In LangGraph, a node represents an action that your graph can take, and every action is defined by a function. For example, you can use `NOTICE_PARSER_CHAIN` and `ESCALATION_CHECK_CHAIN` as the first nodes in your graph:


```python filename="graphs/notice_extraction.py"
# ...

def parse_notice_message_node(state: GraphState) -> GraphState:
    """Use the notice parser chain to extract fields from the notice"""

    LOGGER.info("Parsing notice# ...")

    notice_email_extract = NOTICE_PARSER_CHAIN.invoke(
        {"message": state["notice_message"]}
    )

    state["notice_email_extract"] = notice_email_extract

    return state


def check_escalation_status_node(state: GraphState) -> GraphState:
    """Determine whether a notice needs escalation"""

    LOGGER.info("Determining escalation status# ...")

    text_check = ESCALATION_CHECK_CHAIN.invoke(
        {
            "escalation_criteria": state["escalation_text_criteria"],
            "message": state["notice_message"],
        }
    ).needs_escalation

    if (
        text_check
        or state["notice_email_extract"].max_potential_fine
        >= state["escalation_dollar_criteria"]
    ):
        state["requires_escalation"] = True
    else:
        state["requires_escalation"] = False

    return state

workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
```

Here, you define `parse_notice_message_node()` -  a function that accepts your `GraphState` instance, runs the `notice_message` attribute of `state` through `NOTICE_PARSER_CHAIN.invoke()`, stores the results in the `state`, and returns the `state`. In general, all node functions accept the graph state, perform some action, update the graph state, and return the graph state.

Similarly, `check_escalation_status_node()` passes the `escalation_text_criteria` and `notice_message` from `state` through `ESCALATION_CHECK_CHAIN.invoke()`. If the chain determines that escalation is required, or the extracted `max_potential_fine` is greater than `state["escalation_dollar_criteria"]`, the `requires_escalation` attribute is set to `True`.

You then add the nodes to your graph with `workflow.add_node()` - a method that accepts the name of your node and the function that determines what your node does. For example, `workflow.add_node("parse_notice_message", parse_notice_message_node)` assigns `parse_notice_message_node()` to a graph node called `parse_notice_message`. The graph passes state to `parse_notice_message_node()` and stores the output in an updated state.

The next thing you need to do is add edges to your graph. Edges control the flow of data between nodes in your graph. Said differently, after a node performs an action and updates your graph's state, the edge flowing out of the node tells it which node to pass state to next. Here's how you add edges to your graph:

```python filename="graphs/notice_extraction.py"
# ...

workflow.add_edge(START, "parse_notice_message")
workflow.add_edge("parse_notice_message", "check_escalation_status")
workflow.add_edge("check_escalation_status", END)

NOTICE_EXTRACTION_GRAPH = workflow.compile()
```

You call `workflow.add_edge()` which accepts the names of the start and end nodes of the edge, respectively.`START` is a predefined node representing the entry point of the graph, and `END` is the node that terminates graph. 

You call `workflow.add_edge(START, "parse_notice_message")` to add an edge from `START` to the `parse_notice_message`. Then, `workflow.add_edge("parse_notice_message", "check_escalation_status")` adds an edge from `parse_notice_message` to `check_escalation_status`. Lastly, `workflow.add_edge("check_escalation_status", END)` to terminate the graph after running `check_escalation_status`. 

You can now compile your graph by running `workflow.complie()` which creates a `Runnable` interface that can be invoked, streamed, batched, and run asynchronously just like chains.

To visualize your graph, open Python REPL and run the following:

```pycon
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH

>>> image_data = NOTICE_EXTRACTION_GRAPH.get_graph().draw_mermaid_png()
>>> with open("notice_extraction_graph.png", "wb") as f:
# ...     f.write(image_data)
# ...
8088
```

Here, you import `NOTICE_EXTRACTION_GRAPH` and use `.get_grah().draw_mermaid_png()` to create an image of your graph. You then save the image to a file called `notice_extraction_graph.png`. Here's what the image should look like:

{% img 'notice-extraction-graph-2' centered=True caption="Your first notice extraction graph"%}

This visual shows you that state flows from `parse_notice_message` to `check_escalation_status`, and it confirms that you've built your graph correctly. Here's how you use your graph:

```pycon
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH

>>> example_notice = """
# ...      Date: October 15, 2024
# ...      From: Occupational Safety and Health Administration (OSHA)
# ...      To: Blue Ridge Construction, project 111232345 - Downtown
# ...      Office Complex
# ...
# ...      During a recent inspection of your construction site at 123
# ...      Main Street, the following safety violations were identified:
# ...
# ...      Lack of fall protection: Workers on scaffolding above 10 feet
# ...      were observed without required harnesses or other fall protection
# ...      equipment. Unsafe scaffolding setup: Several scaffolding structures
# ...      were noted as lacking secure base plates and bracing, creating
# ...      potential collapse risks. Inadequate personal protective equipment
# ...      (PPE): Multiple workers were found without proper PPE, including
# ...      hard hats and safety glasses.
# ...
# ...      Required Corrective Actions:
# ...
# ...      Install guardrails and fall arrest systems on all scaffolding over
# ...      10 feet. Conduct an inspection of all scaffolding structures and
# ...      reinforce unstable sections. Ensure all workers on-site are provided
# ...      with necessary PPE and conduct safety training on proper usage.
# ...      Deadline for Compliance: All violations must be rectified by
# ...      November 10, 2024. Failure to comply may result in fines of up
# ...      to $25,000 per violation.
# ...
# ...      Contact: For questions or to confirm compliance, please reach out
# ...      to the OSHA regional office at (555) 123-4567 or email
# ...      compliance.osha@osha.gov.
# ... """

>>> initial_state = {
# ...     "notice_message": example_notice,
# ...     "notice_email_extract": None,
# ...     "escalation_text_criteria": """There's a risk of fire or water damage
# ...     at the site""",
# ...     "escalation_dollar_criteria": 100_000,
# ...     "requires_escalation": False,
# ...     "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
# ... }
 
>>> final_state = NOTICE_EXTRACTION_GRAPH.invoke(initial_state)
2025-01-02 08:34:34,042 - INFO - Parsing notice# ...
2025-01-02 08:34:46,456 - INFO - Determining escalation status# ...
 
>>> final_state["notice_email_extract"]
NoticeEmailExtract(
    date_of_notice=datetime.date(2024, 10, 15),
    entity_name='Occupational Safety and Health Administration (OSHA)',
    entity_phone='(555) 123-4567',
    entity_email='compliance.osha@osha.gov',
    project_id=111232345,
    site_location='123 Main Street',
    violation_type='Lack of fall protection, Unsafe scaffolding
    setup, Inadequate personal protective equipment (PPE)',
    required_changes='Install guardrails and fall arrest systems
    on all scaffolding over 10 feet. Conduct an inspection of all
    scaffolding structures and reinforce unstable sections. Ensure
    all workers on-site are provided with necessary PPE and conduct
    safety training on proper usage.',
    compliance_deadline=datetime.date(2024, 11, 10),
    max_potential_fine=25000.0
)

>>> final_state["requires_escalation"]
False
```

In this example, you import `NOTICE_EXTRACTION_GRAPH` and define `example_notice`. You then define your graph's initial state with `example_notice` as the `notice_message`, escalation criteria, and a few other fields that you'll use later in this tutorial. After calling `NOTICE_EXTRACTION_GRAPH.invoke(initial_state)`, your `final_state` stores the extracted notice fields along with the `requires_escalation` flag.

Notice that in `final_state`, `notice_email_extract` is now a `NoticeEmailExtract` object. Also, `requires_escalation` is set to `False` because `example_notice` doesn't say anything about fire or water damage, and the maximum potential fine is less than $100,000.  

With that, you've built and successfully run your first state graph! If you're thinking that this graph isn't any more useful than a chain at this point, you're correct. In this next section, you'll address this by learning about and implementing LangGraph's differentiating feature - conditional edges.

## Conditional Edges

LangGraph is all about modeling LLM workflows as graphs with nodes and edges. Nodes represent actions that your graph can take like calling functions or invoking chains, and edges tell your graph how to navigate between nodes. 

So far, you've built a graph with a couple of nodes and edges between them that can't do much more than a chain. In this section, you'll learn about conditional edges which you can use to move beyond chain-like structures to create intricate, conditional, and even cyclic workflows.

### Create a Conditional Edge

Up to this point, your graph can extract notice fields using `NOTICE_PARSER_CHAIN` and determine whether the notice message requires immediate escalation using `ESCALATION_CHECK_CHAIN`.

Next, you'll see how to change the path your graph takes depending on whether a notice message requires escalation. If a notice message requires escalation, your graph will immediately send an email informing the correct team. If escalation isn't required, your graph will create a legal ticket using your company's ticketing system API.

You'll start by defining a function that sends emails regarding the details of a notice message when the notice requires escalation. Here's what that looks like:

```python filename="utils/graph_utils.py"
import time
import random
import logging
from pydantic import EmailStr
from chains.notice_extraction import NoticeEmailExtract

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def send_escalation_email(
    notice_email_extract: NoticeEmailExtract,
    escalation_emails: list[EmailStr]
) -> None:
    """Simulate sending escalation emails"""

    LOGGER.info("Sending escalation emails# ...")
    for email in escalation_emails:
        time.sleep(1)
        LOGGER.info(f"Escalation email sent to {email}")


def create_legal_ticket(
    current_follow_ups: dict[str, bool] | None,
    notice_email_extract: NoticeEmailExtract,
) -> str | None:
    """Simulate creating a legal ticket using your company's API."""
    LOGGER.info("Creating legal ticket for notice# ...")
    time.sleep(2)

    follow_ups = [
        None,
        """Does this message mention the states of Texas,
        Georgia, or New Jersey?""",
        """Did this notice involve an issue with FakeAirCo's
        HVAC system?""",
    ]

    if current_follow_ups:

        follow_ups = [
            f for f in follow_ups if f not in current_follow_ups.keys()
        ]

    follow_up = random.choice(follow_ups)

    if not follow_up:

        LOGGER.info("Legal ticket created!")
        return follow_up

    LOGGER.info("Follow-up is required before creating this ticket")

    return follow_up
```

Here, you import dependencies and create a function called `send_escalation_email()` which accepts a `NoticeEmailExtract` and a list of addresses to send emails to. The purpose of `send_escalation_email()` is to simulate sending emails since actually sending emails is beyond the scope of this tutorial. For now, imagine that `send_escalation_email()` sends an email to each address in `escalation_emails` regarding the details stored in `NoticeEmailExtract`.

You then define `create_legal_ticket()` to simulate creating a ticket for your company's legal team to investigate. Notice how the first argument to `create_legal_ticket()` is a dictionary with string keys and boolean values called `current_follow_ups`.

One feature of your legal team's ticketing system API is that it occasionally requires you to answer yes/no follow-up questions. These questions can change at any time depending on what the legal team is interested in knowing, so it's difficult to know what the follow-up questions might be ahead of time.

To simulate this logic, `create_legal_ticket()` checks the follow-up questions that you've already answered, and it randomly picks a new follow-up question from the remaining questions. If follow-up isn't required, `create_legal_ticket()` creates a legal ticket and returns `None`. If follow-up is required, `create_legal_ticket()` returns the follow-up question, and you'll see how to handle this in your graph later on.

Next, you'll wrap `send_escalation_email()` and `create_legal_ticket()` in nodes that can interact with your graph's state:

```python filename="graphs/notice_extraction.py" hl_lines="7"
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import EmailStr
from chains.notice_extraction import NoticeEmailExtract, NOTICE_PARSER_CHAIN
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from utils.graph_utils import send_escalation_email, create_legal_ticket

# ...

def send_escalation_email_node(state: GraphState) -> GraphState:
    """Send an escalation email"""

    send_escalation_email(
        notice_email_extract=state["notice_email_extract"],
        escalation_emails=state["escalation_emails"],
    )

    return state


def create_legal_ticket_node(state: GraphState) -> GraphState:
    """Node to create a legal ticket"""

    follow_up = create_legal_ticket(
        current_follow_ups=state.get("follow_ups"),
        notice_email_extract=state["notice_email_extract"],
    )

    state["current_follow_up"] = follow_up

    return state

# ...
```

After importing your util functions, you define two new node functions. In `send_escalation_email_node()`, you call `send_escalation_email()` without modifying `state`. Conversely, in `create_legal_ticket_node()`, you call `create_legal_ticket()` and store the `follow_up` question, if there is one, in `state`. 

Now onto the critical part of this section. You need to create a conditional edge based on whether a notice email requires escalation. Specifically, if a notice email requires escalation, your graph needs to pass state to `send_escalation_email_node()` before creating a legal ticket. If no escalation is necessary, your graph can move directly to `create_legal_ticket_node()`. 

Here's how you can create this behavior in your graph:

```python filename="graphs/notice_extraction.py" hl_lines="17 18 22-31"
# ...

def route_escalation_status_edge(state: GraphState) -> str:
    """Determine whether to send an escalation email or
    create a legal ticket"""

    if state["requires_escalation"]:
        LOGGER.info("Escalation needed!")
        return "send_escalation_email"

    LOGGER.info("No escalation needed")

    return "create_legal_ticket"


workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
workflow.add_node("send_escalation_email", send_escalation_email_node)
workflow.add_node("create_legal_ticket", create_legal_ticket_node)

workflow.add_edge(START, "parse_notice_message")
workflow.add_edge("parse_notice_message", "check_escalation_status")
workflow.add_conditional_edges(
    "check_escalation_status",
    route_escalation_status_edge,
    {
        "send_escalation_email": "send_escalation_email",
        "create_legal_ticket": "create_legal_ticket",
    },
)
workflow.add_edge("send_escalation_email", "create_legal_ticket")
workflow.add_edge("create_legal_ticket", END)

NOTICE_EXTRACTION_GRAPH = workflow.compile()
```

You first define `route_escalation_status_edge()` which is a function that governs the behavior of your conditional edge. Depending on whether the notice requires escalation, `route_escalation_status_edge()` returns a string that indicates which node to navigate to next. After registering the `send_escalation_email` and `create_legal_ticket` nodes, you register a conditional edge using `workflow.add_conditional_edge()`. 

In `workflow.add_conditional_edge()`, the first argument tells your graph which node the edge starts from - `check_escalation_status` in this case. The second argument is the function that governs the behavior of the conditional edge, and the third argument is a mapping that tells your graph which node to go to based on the output of the conditional edge function. 

For instance, if `route_escalation_status_edge()` returns `send_escalation_email`, your graph will invoke the `send_escalation_email` node.
Lastly, you add edges from `send_escalation_email` to `create_legal_ticket` and from `create_legal_ticket` to `END`, and you compile your graph. Here's what your graph looks like now:

{% img 'langgraph-graph-2' centered=True caption="Your current notice extraction graph with a conditional edge"%}

You now see dotted lines representing the conditional edge flowing out of `check_escalation_status` into `send_escalation_email` and `create_legal_ticket`. As expected, if escalation is needed, your graph goes to `send_escalation_email` before `create_legal_ticket`. Otherwise, your graph flows directly to `create_legal_ticket`.

To see your conditional edge in action, try this example:

```pycon
>>> from graphs.notice_extraction_dev import NOTICE_EXTRACTION_GRAPH
 
>>> example_notice = """
# ...     Date: October 15, 2024
# ...     From: Occupational Safety and Health Administration (OSHA)
# ...     To: Blue Ridge Construction, project 111232345 - Downtown
# ...     Office Complex
# ...
# ...     During a recent inspection of your construction site at 123
# ...     Main Street, the following safety violations were identified:
# ...
# ...     Lack of fall protection: Workers on scaffolding above 10 feet
# ...     were observed without required harnesses or other fall protection
# ...     equipment. Unsafe scaffolding setup: Several scaffolding structures
# ...     were noted as lacking secure base plates and bracing, creating
# ...     potential collapse risks. Inadequate personal protective equipment
# ...     (PPE): Multiple workers were found without proper PPE, including
# ...     hard hats and safety glasses.
# ...
# ...     Required Corrective Actions:
# ...
# ...     Install guardrails and fall arrest systems on all scaffolding over
# ...     10 feet. Conduct an inspection of all scaffolding structures and
# ...     reinforce unstable sections. Ensure all workers on-site are provided
# ...     with necessary PPE and conduct safety training on proper usage.
# ...     Deadline for Compliance: All violations must be rectified by November
# ...     10, 2024. Failure to comply may result in fines of up to $25,000 per
# ...     violation.
# ...
# ...     Contact: For questions or to confirm compliance, please reach out to
# ...     the OSHA regional office at (555) 123-4567 or email
# ...     compliance.osha@osha.gov.
# ... """
 
>>> initial_state_no_escalation = {
# ...      "notice_message": example_notice,
# ...      "notice_email_extract": None,
# ...      "escalation_text_criteria": """There's a risk of fire or water
# ...      damage at the site""",
# ...      "escalation_dollar_criteria": 100_000,
# ...      "requires_escalation": False,
# ...      "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
# ... }

>>> initial_state_escalation = {
# ...      "notice_message": example_notice,
# ...      "notice_email_extract": None,
# ...      "escalation_text_criteria": """Workers explicitly violating
# ...      safety protocols""",
# ...      "escalation_dollar_criteria": 100_000,
# ...      "requires_escalation": False,
# ...      "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
# ... }
 
>>> _ = NOTICE_EXTRACTION_GRAPH.invoke(initial_state_no_escalation)
2025-01-19 23:37:35,627 - INFO - Parsing notice# ...
2025-01-19 23:37:38,584 - INFO - Determining escalation status# ...
2025-01-19 23:37:39,270 - INFO - No escalation needed
2025-01-19 23:37:39,271 - INFO - Creating legal ticket for notice# ...
2025-01-19 23:37:41,277 - INFO - Legal ticket created!


>>> _ = NOTICE_EXTRACTION_GRAPH.invoke(initial_state_escalation)
2025-01-19 23:37:57,977 - INFO - Parsing notice# ...
2025-01-19 23:38:01,391 - INFO - Determining escalation status# ...
2025-01-19 23:38:01,903 - INFO - Escalation needed!
2025-01-19 23:38:01,903 - INFO - Sending escalation emails# ...
2025-01-19 23:38:02,908 - INFO - Escalation email sent to brog@abc.com
2025-01-19 23:38:03,913 - INFO - Escalation email sent to bigceo@company.com
2025-01-19 23:38:03,915 - INFO - Creating legal ticket for notice# ...
2025-01-19 23:38:05,920 - INFO - Legal ticket created!
```

In this block, you use the same `example_notice` from the previous section and create two initial states - `initial_state_no_escalation` and `initial_state_escalation`. You expect `initial_state_no_escalation` to not require escalation because `example_notice` doesn't mention anything about water damage. On the other hand, `initial_state_escalation` should require escalation because the workers are violating safety protocols by not wearing PPE. 

Exactly as expected, you can see from the logs that `NOTICE_EXTRACTION_GRAPH.invoke(initial_state_escalation)` requires escalation, and your graph sends emails to the addresses listed in `state["escalation_emails"]` before creating a legal ticket. Compare this to `NOTICE_EXTRACTION_GRAPH.invoke(initial_state_no_escalation)` which moves directly to creating a legal ticket without sending escalation emails. 

Your graph now moves beyond the limitations of chains by handling conditional workflows. Take a moment to think about why the conditional edge abstraction is so powerful. Most meaningful real-world tasks involve several decisions that change the trajectory of steps you take. Trying to replicate this behavior with chains would require a lot of boilerplate conditional and iterative logic, and it would quickly get out of hand as your graph grows.   

Now what about those follow-up questions? If the legal ticketing API returns a follow-up question, how can you answer it and attempt to create a legal ticket again? The answer lies in cycles, and that's what you'll explore next.

### Conditional Edges for Cycles

The last capability you'll explore for your notice extraction graph is cycle. A graph cycle is effectively a loop between two nodes that continues until a task is complete. In this section, you'll build a cycle between the node that creates legal tickets and a new node that answers the follow-up questions.

If you recall from the last section, sometimes the legal ticketing API returns follow-up questions that you're required to answer before creating the ticket. You want to build a cycle that receives and answers the follow-up questions from the legal ticket node until no other follow-up questions are required. To keep things simple for this tutorial, these questions will always have binary yes or no answers.

First, create a chain that you'll use to answer binary questions:

```python filename="chains/binary_questions.py"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class BinaryAnswer(BaseModel):
    is_true: bool = Field(
        description="""Whether the answer to the question is yes or no.
        True if yes otherwise false."""
    )


binary_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer this question as True for "yes" and False for "no".
            No other answers are allowed:
            
            {question}
            """,
        )
    ]
)

binary_question_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

BINARY_QUESTION_CHAIN = (
    binary_question_prompt
    | binary_question_model.with_structured_output(BinaryAnswer)
)
```

This `BINARY_QUESTION_CHAIN` is similar to the chains you built previously. It accepts a `question` as input and outputs `True` if the answer to the question is `yes` and `False` if the answer is `no`. Of course, this chain could give strange results for questions that don't have yes/no answers, but you could modify it to return `None` or to answer arbitrary questions with a text response. 

Next, you'll create a node in your graph that uses `BINARY_QUESTION_CHAIN` to answer follow-up questions:

```python filename="graphs/notice_extraction.py" hl_lines="7"
import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import EmailStr
from chains.notice_extraction import NoticeEmailExtract, NOTICE_PARSER_CHAIN
from chains.escalation_check import ESCALATION_CHECK_CHAIN
from chains.binary_questions import BINARY_QUESTION_CHAIN
from utils.graph_utils import send_escalation_email, create_legal_ticket

# ...

def answer_follow_up_question_node(state: GraphState) -> GraphState:
    """Answer follow-up questions about the notice using
    BINARY_QUESTION_CHAIN"""

    if state["current_follow_up"]:

        question = state["current_follow_up"] + " " + state["notice_message"]

        answer = BINARY_QUESTION_CHAIN.invoke({"question": question})

        if state.get("follow_ups"):

            state["follow_ups"][state["current_follow_up"]] = answer

        else:

            state["follow_ups"] = {state["current_follow_up"]: answer}

    return state

# ...
```

In `answer_follow_up_question_node()`, you check if there's a follow-up question in `state`. If there is, you pass the question, along with this notice message, through `BINARY_QUESTION_CHAIN.invoke()`. You then store the answer as an entry to the `state["follow_ups"]` dictionary.

Now you need to create a function to define the conditional edge between the create legal ticket and answer follow-up question nodes:

```python filename="graphs/notice_extraction.py"
# ...

def route_follow_up_edge(state: GraphState) -> str:
    """Determine whether a follow-up question is required"""

    if state.get("current_follow_up"):

        return "answer_follow_up_question"

    return END

# ...
```

In `route_follow_up_edge()`, if there is a follow-up question in `state`, your navigate to `answer_follow_up_question`. If not, you exit the graph. Now you can register your new node and create the conditional edge:

```python filename="graphs/notice_extraction.py" hl_lines="7 19-26 28 29"
# ...

workflow.add_node("parse_notice_message", parse_notice_message_node)
workflow.add_node("check_escalation_status", check_escalation_status_node)
workflow.add_node("send_escalation_email", send_escalation_email_node)
workflow.add_node("create_legal_ticket", create_legal_ticket_node)
workflow.add_node("answer_follow_up_question", answer_follow_up_question_node)

workflow.add_edge(START, "parse_notice_message")
workflow.add_edge("parse_notice_message", "check_escalation_status")
workflow.add_conditional_edges(
    "check_escalation_status",
    route_escalation_status_edge,
    {
        "send_escalation_email": "send_escalation_email",
        "create_legal_ticket": "create_legal_ticket",
    },
)
workflow.add_conditional_edges(
    "create_legal_ticket",
    route_follow_up_edge,
    {
        "answer_follow_up_question": "answer_follow_up_question",
        END: END,
    },
)

workflow.add_edge("send_escalation_email", "create_legal_ticket")
workflow.add_edge("answer_follow_up_question", "create_legal_ticket")

NOTICE_EXTRACTION_GRAPH = workflow.compile()
```

Here, you add the `answer_follow_up_question` node and define the conditional edge that starts from `create_legal_ticket` and flows to either `answer_follow_up_question` or `END`. Notice that you also add an edge from `answer_follow_up_question` back to `create_legal_ticket` which completes the cycle. To make more sense of this, take a look at the updated visualization of your graph:

{% img 'langgraph-graph-3' centered=True caption="Your updated graph with a cycle between `create_legal_ticket` and `answer_follow_up_question`" %}

Your graph has come a long way from where it started! Notice the dotted arrows, representing conditional edges, coming out of `create_legal_ticket`, and the edge flowing back into `create_legal_ticket` from `answer_follow_up_question`. This creates a cycle because `create_legal_ticket` will continue to pass state to `answer_follow_up_question` until all follow-ups are answered. When there are no remaining follow-ups, the graph exits.

Give your updated graph a test run:

```pycon
>>> from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH
 
>>> example_notice = """
# ... Date: October 15, 2024
# ... From: Occupational Safety and Health Administration (OSHA)
# ... To: Blue Ridge Construction, project 111232345 - Downtown Office Complex
# ... Location: Dallas, TX
# ...
# ... During a recent inspection of your construction site at 123 Main Street,
the following safety violations were identified:
# ...
# ... Lack of fall protection: Workers on scaffolding above 10 feet were
# ... without required harnesses or other fall protection equipment.
# ... Unsafe scaffolding setup: Several scaffolding structures were noted as
# ... lacking secure base plates and bracing, creating potential collapse risks.
# ... Inadequate personal protective equipment (PPE): Multiple workers were found
# ... without proper PPE, including hard hats and safety glasses.
# ... Required Corrective Actions:
# ...
# ... Install guardrails and fall arrest systems on all scaffolding over 10 feet.
# ... Conduct an inspection of all scaffolding structures and reinforce unstable
# ... sections. Ensure all workers on-site are provided with necessary PPE and conduct
# ... safety training on proper usage.
# ... Deadline for Compliance: All violations must be rectified by November 10, 2024.
# ... Failure to comply may result in fines of up to $25,000 per violation.
# ...
# ... Contact: For questions or to confirm compliance, please reach out to the OSHA
# ... regional office at (555) 123-4567 or email compliance.osha@osha.gov.
# ... """
 
>>> initial_state_escalation = {
# ...       "notice_message": example_notice,
# ...       "notice_email_extract": None,
# ...       "escalation_text_criteria": """Workers explicitly violating safety
# ...                                   protocols""",
# ...       "escalation_dollar_criteria": 100_000,
# ...       "requires_escalation": False,
# ...       "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
# ...  }

>>> results = NOTICE_EXTRACTION_GRAPH.invoke(initial_state_escalation)
2025-01-21 22:06:23,507 - INFO - Parsing notice# ...
2025-01-21 22:06:27,046 - INFO - Determining escalation status# ...
2025-01-21 22:06:27,868 - INFO - Escalation needed!
2025-01-21 22:06:27,869 - INFO - Sending escalation emails# ...
2025-01-21 22:06:28,873 - INFO - Escalation email sent to brog@abc.com
2025-01-21 22:06:29,879 - INFO - Escalation email sent to bigceo@company.com
2025-01-21 22:06:29,882 - INFO - Creating legal ticket for notice# ...
2025-01-21 22:06:31,887 - INFO - Follow-up is required before creating this ticket
2025-01-21 22:06:32,375 - INFO - Creating legal ticket for notice# ...
2025-01-21 22:06:34,377 - INFO - Follow-up is required before creating this ticket
2025-01-21 22:06:34,935 - INFO - Creating legal ticket for notice# ...
2025-01-21 22:06:39,397 - INFO - Legal ticket created!

>>> results["follow_ups"]
{"Did this notice involve an issue with FakeAirCo's HVAC system?":
 BinaryAnswer(is_true=False),
 'Does this message mention the states of Texas, Georgia, or New Jersey?':
 BinaryAnswer(is_true=True)}
```

Here, you import `NOTICE_EXTRACTION_GRAPH` and modify the previous example to include the text `Location: Dallas, TX`. Recall that one of the follow-up questions in `create_legal_ticket()` is `Does this message mention the states of Texas, Georgia, or New Jersey?` Hence, if `create_legal_ticket()` requires you to answer that question, the answer should be `True` since `example_notice` mentions Dallas, Texas.  

In the above example when you run `example_notice` through your graph, you see from the logs that `create_legal_ticket()` requires your graph to answer both follow-up questions before creating the ticket. Keep in mind that it might take you a few tries to replicate this example since `create_legal_ticket()` randomly selects between two questions and `None`. From `results["follow_ups"]`, you see that `BINARY_QUESTION_CHAIN` correctly answered the questions.

With that, you've completed your notice email processing graph - nice work! Hopefully you're seeing the power of LangGraph in creating sophisticated, conditional, and cyclic LLM workflows. From here, you can continue adding any functionality that you can imagine by simply creating nodes to accomplish tasks and adding edges to navigate between them. 

In the next and final section, you'll add some nice finishing touches to your notice email processing graph. You'll use LangGraph to create an agent that can handle any kind of email and use `NOTICE_EXTRACTION_GRAPH` as one of it's tools.

## Graph Agents

In the final section of this tutorial, you'll complete your email-parsing system by creating an agent with LangGraph. If you're unfamiliar with agents, you can read about them in a previous [LangChain tutorial](https://realpython.com/build-llm-rag-chatbot-with-langchain/#agents). For the sake of a brief definition, an AI agent is a system of AI models, usually LLMs, capable of performing tasks and making decisions autonomously. 

The two main components of an agent are the models that make decisions and the tools the models use to perform actions. LangGraph was designed with agents in mind because agent architectures tend to be conditional and cyclic in nature. The agent you'll create in this section will govern email processing, and it will have access to tools that can send emails and call `NOTICE_EXTRACTION_GRAPH` when it determines that an email is a regulatory notice.

### Agents as Graphs

To get started building your email processing agent, first import the following dependencies and configure a logger:

```python filename="graphs/email_agent.py"
import time
import random
import logging
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from chains.notice_extraction import NoticeEmailExtract
from graphs.notice_extraction import NOTICE_EXTRACTION_GRAPH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)
```

Most of these imports should look familiar, but notice that you've imported `MessagesState` instead of `GraphState`. `MessagesState` is a predefined `GraphState` that has only one attribute called `messages`. Each node in your agent graph simply appends it's output to `messages` in `MessagesState`, and `MessagesState` comes with some nice features that makes creating agents easier. You'll see these in a moment.

Notice also that you've imported the `ToolNode` class from `langgraph.prebuilt`. `ToolNode` allows you to create nodes in your graph explicitly designed for providing tools to agent nodes. In both LangChain and LangGraph, a tool is nothing more than a function your agent models can call to perform actions. Here are the tools you'll need for your email agent:

```python filename="graphs/email_agent.py"
# ...

@tool
def forward_email(email_message: str, send_to_email: str) -> bool:
    """
    Forward an email_message to the address of sent_to_email. Returns
    true if the email was successful otherwise it wil return false. Note
    that this tool only forwards the email to an internal department -
    it does not reply to the sender.
    """

    LOGGER.info(f"Forwarding the email to {send_to_email}# ...")
    time.sleep(2)
    LOGGER.info("Email forwarded!")

    return True


@tool
def send_wrong_email_notification_to_sender(
    sender_email: str, correct_department: str
):
    """
    Send an email back to the sender informing them that
    they have the wrong address. The email should be sent
    to the correct_department.
    """

    LOGGER.info(f"Sending wrong email notification to {sender_email}# ...")
    time.sleep(2)
    LOGGER.info("Email sent!")

    return True


@tool
def extract_notice_data(
    email: str, escalation_criteria: str
) -> NoticeEmailExtract:
    """
    Extract structured fields from a regulatory notice.
    This should be used when the email message comes from
    a regulatory body or auditor regarding a property or
    construction site that the company works on.

    escalation_criteria is a description of which kinds of
    notices require immediate escalation.

    After calling this tool, you don't need to call any others.
    """

    LOGGER.info("Calling the email notice extraction graph# ...")

    initial_state = {
        "notice_message": email,
        "notice_email_extract": None,
        "critical_fields_missing": False,
        "escalation_text_criteria": escalation_criteria,
        "escalation_dollar_criteria": 100_000,
        "requires_escalation": False,
        "escalation_emails": ["brog@abc.com", "bigceo@company.com"],
    }

    results = NOTICE_EXTRACTION_GRAPH.invoke(initial_state)
    return results["notice_email_extract"]


@tool
def determine_email_action(email: str) -> str:
    """
    Call to determine which action should be taken
    for an email. Only use when the other tools don't seem
    relevant for the email task. Do not call this tool if
    you've already called extract_notice_data.
    """

    instructions = """
    If the email appears to be an invoice of any kind or related to
    billing, forward the email to the billing and invoices team:
    billing@company.com and send a wrong email notice back to the
    sender. The correct department is billing@company.com.
    
    If the email appears to be from a customer, forward to
    support@company.com, cdetuma@company.com, and ctu@abc.com.
    Be sure to forward it to all three emails listed.
    Send a wrong email notice back to the
    customer and let them know the correct department is
    support@company.com.
    
    For any other emails, please send a wrong email notification
    and try to infer the correct department from one of
    billing@company.com, support@company.com,
    humanresources@company.com, and it@company.com.
    """

    return instructions
```

The functions you defined above are the tools your agent will use, and each function is [decorated](https://realpython.com/primer-on-python-decorators/) by `@tool`. One crucial functionality of `@tool` is that it gives the function's docstring to your agent's LLM to help it determine whether that particular tool is relevant to the task at hand. Because of this, **it's important that you write informative docstrings so that you maximize the chances that your agent uses the appropriate tools**.

As an example of how your agent will use these tools, if the agent determines that it needs to forward an email, the agent's underlying LLM will return a JSON string with the appropriate `email_message` and `send_to_email` parameters. Your graph will convert this JSON to a dictionary and pass it to `forward_email()`. 

Perhaps the most powerful tool available to your agent is `extract_notice_data()`. Your agent should call `extract_notice_data()` when it thinks the email is a regulatory notice, and it will invoke your `NOTICE_EXTRACTION_GRAPH`. This is an amazing abstraction - because tools are nothing more than functions, you can use them to call other chains, graphs, and agents, and perform just about any task that you can encapsulate in a function. 

The last tool you define, `determine_email_action()`, should be used when your agent receives an email that it doesn't know how to process. In `determine_email_action()`, you can see a set of instructions are predefined, but this tool could be another LLM trained to make decisions on what to do with your company's emails. 

Next, you need to define the nodes and edges of your agent graph:

```python filename="graphs/email_agent.py"
# ...

tools = [
    determine_email_action,
    forward_email,
    send_wrong_email_notification_to_sender,
    extract_notice_data,
]
tool_node = ToolNode(tools)

EMAIL_AGENT_MODEL = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0).bind_tools(tools)

def call_agent_model_node(
    state: MessagesState
) -> dict[str, list[AIMessage]]:
    """Node to call the email agent model"""
    messages = state["messages"]
    response = EMAIL_AGENT_MODEL.invoke(messages)
    return {"messages": [response]}


def route_agent_graph_edge(state: MessagesState) -> str:
    """Determine whether to call more tools or exit the graph"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "email_tools"
    return END
```

You first instantiate a `ToolNode` that can call the four tool functions you defined above. Keep in mind that `tool_node` only executes your tool functions, and knowing which tool to use is the agent node's responsibility. You then instantiate `EMAIL_AGENT_MODEL` which is the LLM that acts as your agent's brain. 

By calling `.bind_tools()` when instantiating `EMAIL_AGENT_MODEL`, you're giving the agent LLM descriptions of each tool from their docstring and input arguments. If your agent LLM determines that its input requires a tool call, it'll return a JSON tool message with the name of the tool it wants to use along with the input arguments. 

In `call_agent_model_node()`, you define the logic for invoking `EMAIL_AGENT_MODEL`. In this case, your email agent node simply takes the messages in `state` and passes them though `EMAIL_AGENT_MODEL.invoke()`. You might notice here that you're not directly modifying the state in `call_agent_model_node()`, but under the hood, LangGraph will automatically append `response` to your graph's state. 

Lastly, you define `route_agent_graph_edge` which governs the conditional edge between your agent and tool nodes. If the most recent message in `state` is a tool, your graph moves to `tool_node`. Otherwise, you terminate the graph. This will make more sense once you instantiate and visualize your graph. Here's the last few lines you need for your graph agent:

```python filename="graphs/email_agent.py"
# ...

workflow = StateGraph(MessagesState)

workflow.add_node("email_agent", call_agent_model_node)
workflow.add_node("email_tools", tool_node)

workflow.add_edge(START, "email_agent")
workflow.add_conditional_edges(
    "email_agent", route_agent_graph_edge, ["email_tools", END]
)
workflow.add_edge("email_tools", "email_agent")

email_agent_graph = workflow.compile()
```

You first instantiate `workflow` - a `StateGraph` that passes around a `MessageState` object. You then add  the `email_agent` and `email_tool` nodes to your graph and create the edges between them. The first edge moves from `START` to `email_agent` where `email_agent` will process the first message in your graph. In this case, the first message will be an email. You then add the conditional edge between `email_agent`, `email_tools`, and `END`. 

As specified in `route_agent_graph_edge()`, if the `email_agent` node returns a tool message, your graph moves to the `email_tools` node to call the respective tool. The last edge you define connects `email_tools` back to `email_agent`, thus creating a cycle between the two. This means that the `email_agent` and `email_tools` nodes will continuously pass information back and forth until `email_agent` returns a non-tool message.

Zooming out, here's what your email agent looks like:

{% img 'langraph-agent' centered=True caption="Your final LangGraph agent." %}

This architecture, while straightforward, is very common and a great place to start when building agents. The general idea is that your `email_agent` will accept an email message and continually call `email_tools` until it believes that it has successfully processed the email. From here, you can continue expanding your agent's capabilities. The last thing to do now is give your agent a try and see how it  performs on a few examples.

### Test Your Graph Agent

Your email graph agent is complete and ready for you to test! In practice, it's a good idea to test your agents on several examples where you know what the desired behavior should be. This way, you can measure your agent's performance and adjust its architecture and prompts to improve it. However, for this tutorial, you'll just empirically inspect how your agent responds to three example emails.

Here are the emails you'll test your agent on:


```python filename="example_emails.py"
EMAILS = [
    # Email 1
    
    """
    From: debby@stack.com
    
    Hey Betsy,
    
    Here's your invoice for $1000 for the cookies you ordered.
    """,
    
    # Email 2
    
    """
    From: tdavid@companyxyz.com
    
    Hi Paul,
    
    We have an issue with the HVAC system your team installed in
    apartment 1235. We'd like to request maintenance or a refund.
    
    Thanks,
    Terrance
    """,
    
    # Email 3
    
    """
    Date: January 10, 2025
    From: City of Los Angeles Building and Safety Department
    To: West Coast Development, project 345678123 - Sunset Luxury
    Condominiums
    Location: Los Angeles, CA

    Following an inspection of your site at 456 Sunset Boulevard, we have
    identified the following building code violations:

    Electrical Wiring: Exposed wiring was found in the underground parking
    garage, posing a safety hazard. Fire Safety: Insufficient fire
    extinguishers were available across multiple floors of the structure
    under construction.
    Structural Integrity: The temporary support beams in the eastern wing
    do not meet the load-bearing standards specified in local building codes.
    Required Corrective Actions:

    Replace or properly secure exposed wiring to meet electrical safety
    standards. Install additional fire extinguishers in compliance with
    fire code requirements. Reinforce or replace temporary support beams
    to ensure structural stability. Deadline for Compliance: Violations
    must be addressed no later than February 5,
    2025. Failure to comply may result in
    a stop-work order and additional fines.

    Contact: For questions or to schedule a re-inspection, please contact
    the Building and Safety Department at
    (555) 456-7890 or email inspections@lacity.gov.
    """,
]
```

Given these three emails, here how you might expect your agent to respond to each:

- **Email 1**: This email looks like it's an invoice that doesn't come from any kind of regulatory body or inspector. Your email agent should forward this to the billing department and send an email to `deby@stack.com` informing her that she has the wrong email address.

- **Email 2**: This email comes from a customer that's having issues with equipment your company installed. Your agent should forward it to customer support and a few other emails, and it should send an email back to `tdavid@companyxyz.com` telling him that he has the wrong address.

- **Email 3**: This email is clearly a regulatory notice, and your agent should call your notice extraction graph. 

To see how your agent does, open a Python REPL and try it out on the first example email:


```pycon
>>> from graphs.email_agent import email_agent_graph
>>> from example_emails import EMAILS

>>> message_1 = {"messages": [("human", EMAILS[0])]}
 
>>> for chunk in email_agent_graph.stream(message_1, stream_mode="values"):
# ...     chunk["messages"][-1].pretty_print()
# ...
================================ Human Message ============================


    From: debby@stack.com
    
    Hey Betsy,
    
    Here's your invoice for $1000 for the cookies you ordered.
    
================================== Ai Message =============================
Tool Calls:
  determine_email_action (call_MWiiVRHsEHKMX6lftRwJEZyO)
 Call ID: call_MWiiVRHsEHKMX6lftRwJEZyO
  Args:
    email: From: debby@stack.com

Hey Betsy,

Here's your invoice for $1000 for the cookies you ordered.
================================= Tool Message ============================
Name: determine_email_action


    If the email appears to be an invoice of any kind or related to
    billing, forward the email to the billing and invoices team:
    billing@company.com and send a wrong email notice back to the sender.
    The correct department is
    billing@company.com.
    
    If the email appears to be from a customer, forward to
    support@company.com,
    cdetuma@company.com, and ctu@abc.com. Send a wrong email
    notice back to the customer and let them know the correct
    department is support@company.com.
    
    For any other emails, please send a wrong email notification
    and try to infer the correct department from one of
    billing@company.com, support@company.com,
    humanresources@company.com, and it@company.com.
    
================================== Ai Message =============================
Tool Calls:
  forward_email (call_PTXylQoEEdKfLS680bPCd9WO)
 Call ID: call_PTXylQoEEdKfLS680bPCd9WO
  Args:
    email_message: From: debby@stack.com

Hey Betsy,

Here's your invoice for $1000 for the cookies you ordered.
    send_to_email: billing@company.com
  send_wrong_email_notification_to_sender (call_NGpIerzPsToKSK3u2JZo5RwW)
 Call ID: call_NGpIerzPsToKSK3u2JZo5RwW
  Args:
    sender_email: debby@stack.com
    correct_department: billing@company.com
2025-01-26 12:00:46,317 - INFO - Forwarding the email to
billing@company.com# ...
2025-01-26 12:00:46,318 - INFO - Sending wrong email notification
to debby@stack.com# ...
2025-01-26 12:00:48,323 - INFO - Email forwarded!
2025-01-26 12:00:48,324 - INFO - Email sent!
================================= Tool Message ============================
Name: send_wrong_email_notification_to_sender

true
================================== Ai Message =============================

The email has been successfully forwarded to the billing and invoices team,
and a notification
has been sent to the sender, informing them that the correct department is
billing@company.com.
```

Here, you import `email_agent_graph` and the example emails. You then run the first email through `email_agent_graph.stream()` which returns a message from your agent as received. The thing your agent does is call `determine_email_action()` to get instructions. From these instructions, your agent forwards the email to `billing@comapny.com` and sends a wrong email notification back to `debby@stack.com`. Your agent behaved exactly like you wanted it to!

Now try your agent on the next email:

```pycon
>>> message_2 = {"messages": [("human", EMAILS[1])]}

>>> for chunk in email_agent_graph.stream(message_2, stream_mode="values"):
# ...     chunk["messages"][-1].pretty_print()
# ...
================================ Human Message ===========================


    From: tdavid@companyxyz.com
    
    Hi Paul,
    
    We have an issue with the HVAC system your team installed in
    apartment 1235. We'd like to request maintenance or a refund.
    
    Thanks,
    Terrance
    
================================== Ai Message ============================
Tool Calls:
  determine_email_action (call_QQwWoaqG3dKc6AjJ92mc6iXI)
 Call ID: call_QQwWoaqG3dKc6AjJ92mc6iXI
  Args:
    email: From: tdavid@companyxyz.com

Hi Paul,

We have an issue with the HVAC system your team installed in
apartment 1235. We'd like to request maintenance or a refund.

Thanks,
Terrance
================================= Tool Message ===========================
Name: determine_email_action


    If the email appears to be an invoice of any kind or related to
    billing, forward the email to the billing and invoices team:
    billing@company.com
    and send a wrong email notice back to the sender. The correct
    department is
    billing@company.com.
    
    If the email appears to be from a customer, forward to
    support@company.com,
    cdetuma@company.com, and ctu@abc.com. Be sure to forward it to
    all three emails listed.
    Send a wrong email notice back to the
    customer and let them know the correct department is
    support@company.com.
    
    For any other emails, please send a wrong email notification
    and try to
    infer the correct department from one of billing@company.com,
    support@company.com,
    humanresources@company.com, and it@company.com.
    
================================== Ai Message ============================
Tool Calls:
  forward_email (call_vvlAiZrF6CyNlD6pOOSzfCIr)
 Call ID: call_vvlAiZrF6CyNlD6pOOSzfCIr
  Args:
    email_message: From: tdavid@companyxyz.com

Hi Paul,

We have an issue with the HVAC system your team installed in
apartment 1235. We'd like to request maintenance or a refund.

Thanks,
Terrance
    send_to_email: support@company.com
  forward_email (call_Pk9b89g1gUty9mMqvo9luZSD)
 Call ID: call_Pk9b89g1gUty9mMqvo9luZSD
  Args:
    email_message: From: tdavid@companyxyz.com

Hi Paul,

We have an issue with the HVAC system your team installed in
apartment 1235. We'd like to request maintenance or a refund.

Thanks,
Terrance
    send_to_email: cdetuma@company.com
  forward_email (call_4JLEOKpb8qOHgdmbDohH071p)
 Call ID: call_4JLEOKpb8qOHgdmbDohH071p
  Args:
    email_message: From: tdavid@companyxyz.com

Hi Paul,

We have an issue with the HVAC system your team installed in
apartment 1235. We'd like to request maintenance or a refund.

Thanks,
Terrance
    send_to_email: ctu@abc.com
  send_wrong_email_notification_to_sender (call_tRimFJO4ElR8g8P96Eqi8EPC)
 Call ID: call_tRimFJO4ElR8g8P96Eqi8EPC
  Args:
    sender_email: tdavid@companyxyz.com
    correct_department: support@company.com
2025-01-26 12:10:21,796 - INFO - Forwarding the email to
support@company.com# ...
2025-01-26 12:10:21,797 - INFO - Forwarding the email to
cdetuma@company.com# ...
2025-01-26 12:10:21,797 - INFO - Sending wrong email notification
to tdavid@companyxyz.com# ...
2025-01-26 12:10:21,798 - INFO - Forwarding the email to ctu@abc.com# ...
2025-01-26 12:10:23,801 - INFO - Email sent!
2025-01-26 12:10:23,802 - INFO - Email forwarded!
2025-01-26 12:10:23,802 - INFO - Email forwarded!
2025-01-26 12:10:23,804 - INFO - Email forwarded!
================================= Tool Message ===========================
Name: send_wrong_email_notification_to_sender

true
================================== Ai Message ===========================

The email from Terrance has been successfully forwarded to the support team,
as well as to the
additional contacts. A notification has also been sent to Terrance informing
him that the correct
the department for his request is support@company.com.
```

For the second email, your agent recognizes that it comes from a customer and it successfully forwards it to the three customer support emails listed in the instructions. It then sends a wrong email notification back to `tdavid@companyxyz.com`. This means your agent made a total of 5 tool calls before terminating the graph - determining the email action, forwarding the email to three different addresses, and sending a wrong email notification. 

Now for the last example email:

```pycon
>>> escalation_criteria = """"There's an immediate risk of electrical,
# ... water, or fire damage""""

>>> message_with_criteria = f"""
# ... The escalation criteria is: {escalation_criteria}
# ...
# ... Here's the email:
# ...
# ... {EMAILS[2]}
# ... """
>>> message_3 = {"messages": [("human", message_with_criteria)]}
 
>>> for chunk in email_agent_graph.stream(message_3, stream_mode="values"):
# ...     chunk["messages"][-1].pretty_print()
# ...
================================ Human Message ===========================


The escalation criteria is: There's an immediate risk of electrical, water,
or fire damage

Here's the email:

 
    Date: January 10, 2025
    From: City of Los Angeles Building and Safety Department
    To: West Coast Development, project 345678123 - Sunset Luxury Condominiums
    Location: Los Angeles, CA

    Following an inspection of your site at 456 Sunset Boulevard, we have
    identified
    the following building code
    violations:

    Electrical Wiring: Exposed wiring was found in the underground parking
    garage, posing
    a safety hazard.
    Fire Safety: Insufficient fire extinguishers were available across
    multiple floors of
    the structure under
    construction.
    Structural Integrity: The temporary support beams in the eastern wing
    do not meet the
    load-bearing standards
    specified in local building codes.
    Required Corrective Actions:

    Replace or properly secure exposed wiring to meet electrical safety
    standards.
    Install additional fire extinguishers in compliance with fire code
    requirements.
# ...
    Contact: For questions or to schedule a re-inspection, please contact
    the Building and
    Safety Department at
    (555) 456-7890 or email inspections@lacity.gov.
    

2025-01-26 13:19:14,234 - INFO - Calling the email notice
extraction graph# ...
2025-01-26 13:19:14,237 - INFO - Parsing notice# ...
================================== Ai Message ==========================
Tool Calls:
  extract_notice_data (call_xd8mvZQmuhUOsoOden6Mrl1z)
 Call ID: call_xd8mvZQmuhUOsoOden6Mrl1z
  Args:
    email:
    Date: January 10, 2025
    From: City of Los Angeles Building and Safety Department
    To: West Coast Development, project 345678123 - Sunset Luxury
    Condominiums
    Location: Los Angeles, CA

    Following an inspection of your site at 456 Sunset Boulevard, we have
    identified the following building code
    violations:

    Electrical Wiring: Exposed wiring was found in the underground parking
    garage, posing a safety hazard.
    Fire Safety: Insufficient fire extinguishers were available across
    multiple floors of the structure under
    construction.
    Structural Integrity: The temporary support beams in the eastern wing
    do not meet the load-bearing standards
    specified in local building codes.
    Required Corrective Actions:

    Replace or properly secure exposed wiring to meet electrical safety
    standards.
    Install additional fire extinguishers in compliance with fire code
    requirements.
    Reinforce or replace temporary support beams to ensure structural
    stability.
    Deadline for Compliance: Violations must be addressed no later than
    February 5, 2025. Failure to comply may result in
# ...

    Contact: For questions or to schedule a re-inspection, please
    ontact the Building and Safety Department at
    (555) 456-7890 or email inspections@lacity.gov.
    escalation_criteria: There's an immediate risk of electrical,
    water, or fire damage
2025-01-26 13:19:16,368 - INFO - Determining escalation status# ...
2025-01-26 13:19:16,982 - INFO - Escalation needed!
2025-01-26 13:19:16,987 - INFO - Sending escalation emails# ...
2025-01-26 13:19:17,988 - INFO - Escalation email sent to
brog@abc.com
2025-01-26 13:19:18,995 - INFO - Escalation email sent to
bigceo@company.com
2025-01-26 13:19:18,998 - INFO - Creating legal ticket for notice# ...
2025-01-26 13:19:21,004 - INFO - Follow-up is required before
creating this ticket
2025-01-26 13:19:21,682 - INFO - Creating legal ticket for notice# ...
2025-01-26 13:19:23,686 - INFO - Legal ticket created!
================================= Tool Message ==========================
Name: extract_notice_data

date_of_notice=datetime.date(2025, 1, 10) entity_name='City of Los Angeles
Building and Safety Department'
entity_phone='(555) 456-7890' entity_email='inspections@lacity.gov'
project_id=345678123
site_location='Los Angeles, CA' violation_type='Electrical Wiring,
Fire Safety, Structural Integrity'
equired_changes='Replace or properly secure exposed wiring to meet
electrical safety standards.
Install additional fire extinguishers in compliance with fire code
requirements. Reinforce or
replace temporary support beams to ensure structural stability.'
compliance_deadline=datetime.date(2025, 2, 5) max_potential_fine=None
================================== Ai Message =======================

The email from the City of Los Angeles Building and Safety Department
contains serious
violations that pose an immediate risk of electrical and fire damage.
Here are the details
extracted from the notice:

- **Date of Notice:** January 10, 2025
- **Entity Name:** City of Los Angeles Building and Safety Department
- **Entity Phone:** (555) 456-7890
- **Entity Email:** inspections@lacity.gov
- **Project ID:** 345678123
- **Site Location:** Los Angeles, CA
- **Violation Types:** Electrical Wiring, Fire Safety, Structural
Integrity
- **Required Changes:**
  - Replace or properly secure exposed wiring to meet electrical
  safety standards.
  - Install additional fire extinguishers in compliance with fire
  code requirements.
  - Reinforce or replace temporary support beams to ensure structural
  stability.
- **Compliance Deadline:** February 5, 2025

Given the nature of the violations, it is crucial to address these
issues immediately to avoid
potential stop-work orders and fines. Would you like to take any
specific actions regarding
this notice?
```

You first define `escalation_criteria` and create a message that includes `escalation_criteria` with the email message. Your agent successfully recognizes that this is a notice email, and it calls the `extract_notice_data` tool. This invokes your notice extraction graph which successfully parses the notice and identifies that it requires escalation. Your agent worked end-to-end on all three examples, and the results are awesome!

Keep in mind that you might not get the exact same results as above given that LLM outputs aren't deterministic, but your results should be very similar. You now have the tools (no pun intended) needed to build your own LangGraph agents!

## Conclusion

Congratulations on completing this in-depth tutorial!

You successfully built an LLM agent in LangGraph, and you have a solid foundation that you can leverage to build your own applications.

**In this tutorial, you've learned how to:**

- Think about **LangGraph** as a complement to **LangChain** for building sophisticated LLM workflows
- Build LangGraph **StateGraph** workflows with **conditional edges** and **cycles**
- Build LangGraph LLM **agents**

You can find the complete source code and data for this project, in the supporting materials, which you can download using the link below: