from typing import Dict, Any, Optional
from src.models.schemas import QAChecklistConfig


class ChecklistManager:
    """Manages configurable QA rubrics and rule thresholds."""

    def __init__(self, config: Optional[QAChecklistConfig] = None):
        self.config = config or QAChecklistConfig()

    def get_config(self) -> QAChecklistConfig:
        return self.config

    def update_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        if item_id in self.config.items:
            self.config.items[item_id].update(updates)
            return True
        return False

    def reset_to_default(self):
        self.config = QAChecklistConfig()
