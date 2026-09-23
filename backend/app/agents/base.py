from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.tools.base import Tool


class BaseAgent(ABC):
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self.tools = {tool.name: tool for tool in tools}

    @abstractmethod
    async def run(self, message: str) -> str:
        raise NotImplementedError
