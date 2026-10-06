import os
import re
import random
import pandas as pd
from typing import List, Dict, Optional, Any
from src.models.schemas import Turn, Speaker, TranscriptInput


class CorpusLoader:
    """Loads and formats conversations from the telecom dataset with dynamic random sampling."""

    def __init__(self, data_path: Optional[str] = None):
        if not data_path:
            suppl_path = os.path.join("telecom-conversation-corpus", "telecom_corpus_supplimental.csv")
            if os.path.exists(suppl_path):
                self.data_path = suppl_path
            else:
                self.data_path = os.path.join("telecom-conversation-corpus", "telecom_200k.csv")
        else:
            self.data_path = data_path

        self._cache: Dict[str, TranscriptInput] = {}
        self._loaded_ids: List[str] = []
        self._df: Optional[pd.DataFrame] = None
        self._all_conv_ids: List[str] = []

    def _ensure_data_loaded(self):
        """Loads and caches the dataframe in memory for sub-second sampling."""
        if self._df is None and os.path.exists(self.data_path):
            self._df = pd.read_csv(self.data_path)
            self._all_conv_ids = self._df["conversation_id"].dropna().unique().tolist()

    def extract_agent_name(self, text: str) -> str:
        """Extracts agent name from greeting e.g. 'My name is Julia'."""
        match = re.search(r"my name is ([A-Za-z]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        match2 = re.search(r"this is ([A-Za-z]+)", text, re.IGNORECASE)
        if match2:
            return match2.group(1).capitalize()
        return "Agent_Unknown"

    def _build_transcript(self, conv_id: str, group: pd.DataFrame) -> TranscriptInput:
        turns: List[Turn] = []
        agent_name = "Agent_Unknown"

        for idx, (_, row) in enumerate(group.iterrows()):
            speaker_str = str(row.get("speaker", "")).strip().lower()
            speaker = Speaker.AGENT if speaker_str == "agent" else Speaker.CLIENT
            text = str(row.get("text", "")).strip()
            dt = str(row.get("date_time", ""))

            if speaker == Speaker.AGENT and agent_name == "Agent_Unknown":
                agent_name = self.extract_agent_name(text)

            turns.append(Turn(
                turn_id=idx + 1,
                speaker=speaker,
                text=text,
                timestamp=dt if dt else None
            ))

        TEAMS = [
            "Retention_Team_Alpha",
            "Billing_Retention_Team_Beta",
            "Tech_Support_Tier1",
            "Compliance_Specialists",
            "General_Telecom_Support"
        ]

        team_map = {
            "Julia": "Retention_Team_Alpha",
            "Justin": "Retention_Team_Alpha",
            "Gertrude": "Compliance_Specialists",
            "Ray": "Tech_Support_Tier1",
            "Alexandra": "Billing_Retention_Team_Beta",
            "Francesco": "Billing_Retention_Team_Beta",
            "Devin": "Retention_Team_Alpha",
            "Vada": "Tech_Support_Tier1"
        }
        if agent_name in team_map:
            team_id = team_map[agent_name]
        elif agent_name != "Agent_Unknown":
            idx = sum(ord(c) for c in agent_name) % len(TEAMS)
            team_id = TEAMS[idx]
        else:
            team_id = "General_Telecom_Support"

        return TranscriptInput(
            conversation_id=str(conv_id),
            agent_id=agent_name,
            team_id=team_id,
            turns=turns
        )

    def load_sample_conversations(self, limit_convs: int = 15, shuffle: bool = True) -> List[TranscriptInput]:
        """Loads a fresh, diverse slice of conversations from across all 8,300+ corpus records."""
        self._ensure_data_loaded()
        if self._df is None or not self._all_conv_ids:
            return []

        if shuffle and len(self._all_conv_ids) > limit_convs:
            selected_ids = random.sample(self._all_conv_ids, limit_convs)
        else:
            selected_ids = self._all_conv_ids[:limit_convs]

        sub_df = self._df[self._df["conversation_id"].isin(selected_ids)]
        grouped = sub_df.groupby("conversation_id", sort=False)

        transcripts: List[TranscriptInput] = []
        for conv_id, group in grouped:
            transcript = self._build_transcript(conv_id, group)
            self._cache[str(conv_id)] = transcript
            transcripts.append(transcript)

        self._loaded_ids = list(self._cache.keys())
        return transcripts

    def get_conversation_by_id(self, conv_id: str) -> Optional[TranscriptInput]:
        if conv_id in self._cache:
            return self._cache[conv_id]

        self._ensure_data_loaded()
        if self._df is not None:
            sub = self._df[self._df["conversation_id"] == conv_id]
            if not sub.empty:
                transcript = self._build_transcript(conv_id, sub)
                self._cache[str(conv_id)] = transcript
                return transcript

        return None
