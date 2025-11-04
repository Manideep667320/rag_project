"""Conversation memory manager using LangChain memory primitives.
 
Exposes a ConversationBufferWindowMemory wrapper and helpers to clear or
retrieve the recent conversation window.
"""
try:
    from langchain.memory import ConversationBufferWindowMemory
except Exception:
    ConversationBufferWindowMemory = None


class LocalConversationMemory:
    def __init__(self, k: int = 3, return_messages: bool = True):
        self.k = k
        self.return_messages = return_messages
        self.buffer = []

    def load_memory_variables(self, inputs: dict):
        history = self.buffer[-(self.k * 2):] if self.k else self.buffer
        return {"history": history}

    def save_context(self, inputs: dict, outputs: dict):
        q = inputs.get("question") or inputs.get("input")
        a = outputs.get("answer") or outputs.get("output") or outputs.get("output_text") or outputs.get("result")
        if q is not None:
            self.buffer.append({"role": "human", "content": str(q)})
        if a is not None:
            self.buffer.append({"role": "ai", "content": str(a)})


def create_memory(k: int = 3) -> "object":
    """Create a ConversationBufferWindowMemory that retains the last k turns.

    return_messages=True so we can render them in the UI.
    """
    if ConversationBufferWindowMemory is not None:
        memory = ConversationBufferWindowMemory(k=k, return_messages=True)
        return memory
    return LocalConversationMemory(k=k, return_messages=True)


def clear_memory(memory: "object"):
    """Clear the conversation buffer."""
    try:
        memory.buffer.clear()
    except Exception:
        try:
            memory.clear()
        except Exception:
            pass
