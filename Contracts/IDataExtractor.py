from abc import ABC, abstractmethod
from typing import Optional

class IDataExtractor(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> Optional[str]:
        pass