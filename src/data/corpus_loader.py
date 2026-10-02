import os
import re
import pandas as pd
from typing import List, Dict, Optional, Any
from src.models.schemas import Turn, Speaker, TranscriptInput


class CorpusLoader:
    """Loads and formats conversations from the telecom dataset."""

    def __init__(self, data_path: Optional[str] = None):
        if not data_path:
            # Default to supplemental corpus or full 200k
            suppl_path = os.path.join("telecom-conversation-corpus", "telecom_corpus_supplimental.csv")
            if os.path.exists(suppl_path):
                self.data_path = suppl_path
            else:
                self.data_path = os.path.join("telecom-conversation-corpus", "telecom_200k.csv")
        else:
            self.data_path = data_path

        self._cache: Dict[str, TranscriptInput] = {}
        self._loaded_ids: List[str] = []

    def extract_agent_name(self, text: str) -> str:
        """Extracts agent name from greeting e.g. 'My name is Julia'."""
        match = re.search(r"my name is ([A-Za-z]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        match2 = re.search(r"this is ([A-Za-z]+)", text, re.IGNORECASE)
        if match2:
            return match2.group(1).capitalize()
        return "Agent_Unknown"

    def load_sample_conversations(self, limit_convs: int = 20) -> List[TranscriptInput]:
        """Loads a slice of distinct conversations from the CSV."""
        if not os.path.exists(self.data_path):
            return []

        # Read enough rows to get limit_convs conversations
        df = pd.read_csv(self.data_path, nrows=limit_convs * 40)
        grouped = df.groupby("conversation_id", sort=False)

        transcripts: List[TranscriptInput] = []
        count = 0

        for conv_id, group in grouped:
            turns: List[Turn] = []
            agent_name = "Agent_Unknown"

            for idx, (_, row) in enumerate(group.iterrows()):
                speaker_str = str(row["speaker"]).strip().lower()
                speaker = Speaker.AGENT if speaker_str == "agent" else Speaker.CLIENT
                text = str(row["text"]).strip()
                dt = str(row.get("date_time", ""))

                if speaker == Speaker.AGENT and agent_name == "Agent_Unknown":
                    agent_name = self.extract_agent_name(text)

                turns.append(Turn(
                    turn_id=idx + 1,
                    speaker=speaker,
                    text=text,
                    timestamp=dt if dt else None
                ))

            # Team allocation based on agent for realistic contact center simulation
            team_map = {
                "Julia": "Retention_Team_Alpha",
                "Justin": "Retention_Team_Alpha",
                "Gertrude": "Compliance_Specialists",
                "Ray": "Tech_Support_Tier1",
                "Alexandra": "Billing_Retention_Team_Beta",
                "Francesco": "Billing_Retention_Team_Beta"
            }
            team_id = team_map.get(agent_name, "General_Telecom_Support")

            transcript = TranscriptInput(
                conversation_id=str(conv_id),
                agent_id=agent_name,
                team_id=team_id,
                turns=turns
            )

            self._cache[str(conv_id)] = transcript
            transcripts.append(transcript)
            count += 1
            if count >= limit_convs:
                break

        self._loaded_ids = list(self._cache.keys())
        return transcripts

    def get_conversation_by_id(self, conv_id: str) -> Optional[TranscriptInput]:
        if conv_id in self._cache:
            return self._cache[conv_id]
        # Otherwise load first batch
        self.load_sample_conversations(limit_convs=30)
        return self._cache.get(conv_id)
