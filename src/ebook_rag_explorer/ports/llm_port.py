"""LLM port - abstract interface for language model generation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Protocol for LLM implementations.

    Implementations should provide text generation capabilities
    using various LLM providers (OpenAI, Azure, local models, etc.).
    """

    def generate(self, prompt: str, context: str) -> str:
        """Generate a response based on the prompt and context.

        Args:
            prompt: The user's query/question.
            context: The retrieved context documents combined into a string.

        Returns:
            The generated answer from the LLM.

        Raises:
            RuntimeError: If generation fails or the LLM is unavailable.
        """
        ...

    async def agenerate(self, prompt: str, context: str) -> str:
        """Asynchronously generate a response.

        Args:
            prompt: The user's query/question.
            context: The retrieved context documents combined into a string.

        Returns:
            The generated answer from the LLM.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the name of the LLM model.

        Returns:
            The model identifier string.
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if the LLM is available and configured.

        Returns:
            True if the LLM can be used for generation.
        """
        ...
