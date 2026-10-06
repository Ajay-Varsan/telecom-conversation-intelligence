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

    def extract_agent_name(self, text: str) -> Optional[str]:
        """Extracts agent name from greeting e.g. 'My name is Julia', 'This is Marcus', or caller opening."""
        match = re.search(r"\bmy name is ([A-Za-z]+)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        match2 = re.search(r"\bthis is ([A-Za-z]+)\b", text, re.IGNORECASE)
        if match2:
            return match2.group(1).capitalize()
        match3 = re.search(r"^(?:hi|hello|hey|good morning|good afternoon)\s+([A-Za-z]+)\b", text, re.IGNORECASE)
        if match3 and match3.group(1).lower() not in ["there", "thank", "thanks", "i", "im", "team", "support", "customer"]:
            return match3.group(1).capitalize()
        return None

    def _build_transcript(self, conv_id: str, group: pd.DataFrame) -> TranscriptInput:
        turns: List[Turn] = []
        found_agent_name = None
        closure_seen = False

        for idx, (_, row) in enumerate(group.iterrows()):
            speaker_str = str(row.get("speaker", "")).strip().lower()
            speaker = Speaker.AGENT if speaker_str == "agent" else Speaker.CLIENT
            text = str(row.get("text", "")).strip()
            text_lower = text.lower()
            dt = str(row.get("date_time", ""))

            # Detect repeated conversation concatenation in raw corpus
            if closure_seen and speaker == Speaker.AGENT and any(k in text_lower for k in [
                "thank you for calling", "my name is", "how can i help", "good morning", "good afternoon", "how can i assist"
            ]):
                break

            # Try to identify agent name from opening turns
            if not found_agent_name and idx < 4:
                cand = self.extract_agent_name(text)
                if cand:
                    found_agent_name = cand

            turns.append(Turn(
                turn_id=len(turns) + 1,
                speaker=speaker,
                text=text,
                timestamp=dt if dt else None
            ))

            # Mark if a closing signoff has been completed by agent
            if len(turns) >= 5 and speaker == Speaker.AGENT and any(k in text_lower for k in [
                "have a great day", "have a good day", "goodbye", "thank you for choosing"
            ]):
                closure_seen = True

        # Deterministic human agent fallback if not named in opening text
        FALLBACK_AGENTS = ["Sadye", "Marcus", "Ericka", "Teri", "Pearl", "Carolina", "Sam", "Derrick", "Leonel", "Elena"]
        if not found_agent_name:
            h = sum(ord(c) for c in str(conv_id))
            found_agent_name = FALLBACK_AGENTS[h % len(FALLBACK_AGENTS)]

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
            "Devin": "Retention_Team_Alpha",
            "Gertrude": "Compliance_Specialists",
            "Ray": "Tech_Support_Tier1",
            "Leonel": "Tech_Support_Tier1",
            "Eric": "Tech_Support_Tier1",
            "Vada": "Tech_Support_Tier1",
            "Alexandra": "Billing_Retention_Team_Beta",
            "Francesco": "Billing_Retention_Team_Beta",
            "Sadye": "General_Telecom_Support",
            "Marcus": "General_Telecom_Support",
            "Pearl": "General_Telecom_Support",
            "Ericka": "General_Telecom_Support"
        }
        if found_agent_name in team_map:
            team_id = team_map[found_agent_name]
        else:
            idx = sum(ord(c) for c in found_agent_name) % len(TEAMS)
            team_id = TEAMS[idx]

        return TranscriptInput(
            conversation_id=str(conv_id),
            agent_id=found_agent_name,
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

    def search_conversations(self, query: str, limit: int = 15) -> List[TranscriptInput]:
        """Searches across all 8,300+ corpus records by conversation ID, agent/customer name, or transcript text."""
        self._ensure_data_loaded()
        if self._df is None or not query or not query.strip():
            return []

        q = query.strip()
        # 1. Exact or partial conversation_id match
        exact_id_matches = [cid for cid in self._all_conv_ids if q.lower() in str(cid).lower()][:limit]
        if exact_id_matches:
            matched_ids = exact_id_matches
        else:
            # 2. Text or name content search across dataframe
            mask = self._df["text"].str.contains(re.escape(q), case=False, na=False)
            matched_ids = self._df.loc[mask, "conversation_id"].dropna().unique().tolist()[:limit]

        if not matched_ids:
            return []

        sub_df = self._df[self._df["conversation_id"].isin(matched_ids)]
        grouped = sub_df.groupby("conversation_id", sort=False)

        results: List[TranscriptInput] = []
        for conv_id, group in grouped:
            transcript = self._build_transcript(conv_id, group)
            self._cache[str(conv_id)] = transcript
            results.append(transcript)

        return results
