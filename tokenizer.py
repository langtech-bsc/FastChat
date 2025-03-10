import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "BSC-LT/salamandra-7b",
    cache_dir="",
    model_max_length=4096,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
    add_prefix_space=False
)

template = """{%- set tools = tools if tools is defined else None -%}
{%- set date_string = date_string if date_string is defined else "1 Sep 2024" -%}

{%- set system_message = messages[0].content if messages[0].role == "system" else "" -%}
{%- if messages[0].role == "system" -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}

{%- if not tool_prompt -%}
    {%- set tool_prompt = "For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-dict>}\n</tool_call>" -%}
{%- endif -%}

{%- if system_message or tools -%}
  {{- '<|im_start|>system\n'}}
{%- endif -%}

{%- if system_message %}
  {{- system_message + "\n"}}
{%- endif -%}

{%- if tools  -%}
  {{- "You have function-calling capabilities. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n" }}
  {{- "<tools>\n" }}
  {{- tools }}
  {{- "\n</tools>\n" }}
  {{- tool_prompt -}}
{%- endif -%}

{%- if system_message or tools -%}
  {{- '<|im_end|>\n'}}
{%- endif -%}

{# Main message loop #}
{%- for message in messages -%}
    {%- if message.role == "user" or message.role == "assistant" or message.role == "tool" -%}
        {%- if loop.first and message.role != "user" -%}
            {{ raise_exception("Invalid sequence: The first message role must be 'user' after 'system' if provided .") }}
        {%- endif -%}

        {%- if not loop.first and message.role in ["user", "assistant"] and message.role == loop.previtem.role -%}
            {{ raise_exception("Invalid sequence: Consecutive messages cannot have the same role ('user' or 'assistant').") }}
        {%- endif -%}

        {%- if message.role == "user" and not loop.first and loop.previtem.role != "assistant" -%}
            {{ raise_exception("Invalid sequence: A 'user' message must be preceded by an 'assistant' message.") }}
        {%- endif -%}

        {%- if message.role == "tool" and not loop.first and loop.previtem.role not in ["assistant", "tool"] -%}
            {{ raise_exception("Invalid sequence: A 'tool' message must be preceded by 'assistant' or 'tool'.") }}
        {%- endif -%}
    {%- else -%}
        {{- raise_exception("Invalid role detected: only 'user', 'assistant', or 'tool' roles are accepted.") }}
    {%- endif -%}
    {%- if message.role == "user" or (message.role == "assistant" and message.tool_calls is not defined) -%}
        {{- '<|im_start|>' + message.role + '\n' + message.content | trim + '<|im_end|>\n'}}
    {%- elif message.role == "assistant" -%}
        {{- '<|im_start|>' + message.role }}
        {%- for tool_call in message.tool_calls -%}
            {{ '\n<tool_call>\n' }}
              {%- if tool_call.function -%}
                {"name": "{{ tool_call.function.name }}", "arguments": {{ tool_call.function.arguments | tojson }} }
              {%- else -%}
                {"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }} }
              {%- endif -%}
            {{ '\n</tool_call>' }}
        {%- endfor -%}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" -%}
        {%- if loop.previtem and loop.previtem.role != "tool" -%}
            {{- '<|im_start|>tool\n' }}
        {%- endif -%}
        {{- '<tool_response>\n' }} 
            {{- message.content }}
        {{- '\n</tool_response>\n' }}
        {%- if loop.last or loop.nextitem.role != "tool" -%}
            {{- '<|im_end|>\n'}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}

{# Prompt for assistant generation if needed #}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' }}
{%- endif -%}"""

msg = [
    {'role': 'system', 'content': 'You are an assistant specialized in providing product information and facilitating online shopping tasks. Your role is to assist with queries related to product availability, details, and purchase processes. Your primary task is to provide support in adding products to the cart, checking shipping costs, and offering information on payment methods, shipping options, available currencies, languages, and return policies. You are tasked with ensuring the conversation remains focused on these topics, while addressing off-topic queries respectfully and guiding the conversation back to the main objectives.'},
 {'role': 'user', 'content': "I'm looking for a specific product. Can you check if it's available?"},
 {'role': 'assistant', 'content': 'Of course! Please provide me with the name of the product.'},
 {'role': 'user', 'content': "I'm interested in the 'Smartwatch X1'."},
 {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'check_product_availability', 'arguments': {'product_name': 'Smartwatch X1'}}]},
 {'role': 'tool', 'content': '{"check_product_availability": true}'},
 {'role': 'assistant', 'content': "The 'Smartwatch X1' is currently available."},
 {'role': 'user', 'content': 'Can you provide me with more details about this product?'},
 {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'get_product_details', 'arguments': {'product_name': 'Smartwatch X1'}}]},
 {'role': 'tool', 'content': '{"get_product_details": {"name": "Smartwatch X1", "description": "A sleek and powerful smartwatch with advanced health and fitness tracking features.", "price": "$249.99", "rating": "4.5/5"}}'},
 {'role': 'assistant', 'content': "The 'Smartwatch X1' is a sleek and powerful smartwatch with advanced health and fitness tracking features. It has a rating of 4.5/5 and is priced at $249.99."},
 {'role': 'user', 'content': "I'd like to add this product to my cart. Can you help me with that?"},
 {'role': 'assistant', 'content': 'Certainly! Please confirm the quantity you would like to add.'},
 {'role': 'user', 'content': "I'd like to add 1 Smartwatch X1 to my cart."},
 {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'add_product_to_cart', 'arguments': {'product_name': 'Smartwatch X1', 'quantity': 1}}]},
 {'role': 'tool', 'content': '{"add_product_to_cart": true}'},
 {'role': 'assistant', 'content': "The 'Smartwatch X1' has been successfully added to your cart."},
 {'role': 'user', 'content': "I'm planning to ship this to the UK. Can you tell me the shipping cost?"},
 {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'get_shipping_cost', 'arguments': {'product_name': 'Smartwatch X1', 'destination_country': 'UK'}}]},
 {'role': 'tool', 'content': '{"get_shipping_cost": "$20.00"}'},
 {'role': 'assistant', 'content': "The shipping cost for the 'Smartwatch X1' to the UK is $20.00."},
 {'role': 'user', 'content': 'What are the available payment methods?'},
 {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'get_available_payment_methods'}]},
 ]

tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False, chat_template=template)
