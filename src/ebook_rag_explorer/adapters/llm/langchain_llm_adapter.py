"""LangChain LLM adapter."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ebook_rag_explorer.ports.llm_port import LLM


class LangChainLLMAdapter(LLM):
    """LangChain-based implementation of the LLM port."""

    def __init__(
        self,
        provider: Literal["openai", "lmstudio", "azure"],
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the LLM adapter.

        Args:
            provider: The LLM provider (openai, lmstudio, azure).
            model: The model name or deployment name.
            api_key: API key for authentication.
            base_url: Optional base URL for the API (e.g., for LM Studio).
        """
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._llm: ChatOpenAI | None = None

    def _get_llm(self) -> ChatOpenAI:
        """Lazy load the LangChain LLM.

        Returns:
            The ChatOpenAI instance.
        """
        if self._llm is None:
            kwargs: dict = {
                "model": self._model,
                "temperature": 0.3,
            }

            if self._api_key:
                kwargs["api_key"] = self._api_key

            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._llm = ChatOpenAI(**kwargs)

        return self._llm

    def _build_prompt(self, query: str, context: str) -> list[SystemMessage | HumanMessage]:
        """Build the prompt messages for the LLM.

        Args:
            query: The user's question.
            context: The retrieved context.

        Returns:
            List of messages for the chat model.
        """
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite your sources by referencing the relevant parts of the context."""

        human_prompt = f"""Context:
{context}

Question: {query}

Please provide a clear and concise answer based on the context above."""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

    def generate(self, prompt: str, context: str) -> str:
        """Generate a response based on the prompt and context.

        Args:
            prompt: The user's query/question.
            context: The retrieved context documents combined into a string.

        Returns:
            The generated answer from the LLM.

        Raises:
            RuntimeError: If generation fails.
        """
        llm = self._get_llm()
        messages = self._build_prompt(prompt, context)

        try:
            response = llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e

    async def agenerate(self, prompt: str, context: str) -> str:
        """Asynchronously generate a response.

        Args:
            prompt: The user's query/question.
            context: The retrieved context documents combined into a string.

        Returns:
            The generated answer from the LLM.
        """
        llm = self._get_llm()
        messages = self._build_prompt(prompt, context)

        try:
            response = await llm.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            raise RuntimeError(f"LLM async generation failed: {e}") from e

    @property
    def model_name(self) -> str:
        """Return the name of the LLM model.

        Returns:
            The model identifier string.
        """
        return f"{self._provider}/{self._model}"

    @property
    def is_available(self) -> bool:
        """Check if the LLM is available and configured.

        Returns:
            True if the LLM can be used for generation.
        """
        # For OpenAI/Azure, check if API key is provided
        if self._provider in ("openai", "azure"):
            return bool(self._api_key)

        # For LM Studio (local), assume available if configured
        return True
