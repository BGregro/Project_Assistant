import random

async def get_joke() -> dict:
    """Return a random joke from a hardcoded list of 5 jokes."""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the scarecrow win an award? He was outstanding in his field!",
        "What do you call a fake noodle? An impasta!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What did the ocean say to the beach? Nothing, it just waved!"
    ]
    
    selected_joke = random.choice(jokes)
    return {
        "success": True,
        "joke": selected_joke
    }

def register_get_joke_tools():
    """Register the get_joke tool."""
    from agent_tools import register_tool
    
    register_tool(
        name="get_joke",
        description="Returns a random joke from a hardcoded list of 5 jokes.",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=get_joke,
        is_destructive=False,
    )
