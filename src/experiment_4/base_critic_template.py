from typing import ClassVar, List

from autointent.schemas import Intent
from autointent.generation.chat_templates._evolution_templates_schemas import Message, Role


class BaseCriticTemplate:
    """Base chat template for using LLM-based critic."""

    _GENERATE_INSTRUCTION: ClassVar[str]

    def __init__(self) -> None:
        """Initialize the BaseCriticTemplate."""
        return

    def __call__(self, intent_data: Intent, prompt: str) -> List[Message]:
        """
        Generate a message with feedback on provided data.

        Args:
            intent_data: Intent data for which the prompt is being constructed.
            prompt: Prompt for which to generate feedback.

        Returns:
            List of messages (consisting only of 1 message) for the chat template.
        """
        content = self._GENERATE_INSTRUCTION.format(
            intent_name=intent_data.name, prompt=prompt
        )

        return [Message(role=Role.USER, content=content)]
