"""Continuous Working Memory - Generative implementation."""

from typing import List, Any


class ContinuousWorkingMemoryGenerative:
    """Continuous Working Memory - Generative (CWM-G).

    CWM-G manages generative memory processes, enabling the system to
    generate and manipulate information dynamically.
    """

    def __init__(self) -> None:
        """Initialize the generative working memory."""
        self._buffer: List[Any] = []

    def add(self, item: Any) -> None:
        """Add an item to the generative memory buffer.

        Args:
            item: The item to add
        """
        self._buffer.append(item)

    def get_buffer(self) -> List[Any]:
        """Get the current memory buffer.

        Returns:
            Copy of the current buffer
        """
        return self._buffer.copy()

    def clear(self) -> None:
        """Clear the generative memory buffer."""
        self._buffer.clear()

    def size(self) -> int:
        """Get the number of items in the buffer.

        Returns:
            Number of items in buffer
        """
        return len(self._buffer)
