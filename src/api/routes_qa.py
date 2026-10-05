from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from src.models.schemas import (
    QAChecklistConfig,
    AgentScorecard,
    TeamRollup
)
from src.api.routes_analyze import checklist_manager, rollup_manager, corpus_loader

router = APIRouter(prefix="/qa", tags=["Quality Assurance & Rollups"])


@router.get("/config", response_model=QAChecklistConfig)
async def get_qa_config():
    """Retrieve the current configurable QA checklist rubric."""
    return checklist_manager.get_config()


@router.post("/config/item/{item_id}")
async def update_qa_item(item_id: str, updates: Dict[str, Any]):
    """Update a specific checklist item's weight, description, or critical status."""
    success = checklist_manager.update_item(item_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail=f"Checklist item '{item_id}' not found.")
    return {"status": "success", "updated_item": item_id, "config": checklist_manager.get_config()}


@router.get("/rollups/agent/{agent_id}", response_model=AgentScorecard)
async def get_agent_scorecard(agent_id: str):
    """Retrieve rolled-up quality performance, pass rate, and coaching tips for an agent."""
    return rollup_manager.get_agent_scorecard(agent_id)


@router.get("/rollups/team/{team_id}", response_model=TeamRollup)
async def get_team_rollup(team_id: str):
    """Retrieve team-level quality rollup, agent ranking leaderboard, and QA item breakdown."""
    return rollup_manager.get_team_rollup(team_id)


@router.get("/rollups/teams", response_model=List[str])
async def list_monitored_teams():
    """List all teams that have active recorded conversations."""
    teams = list(rollup_manager.team_records.keys())
    if not teams:
        return ["Retention_Team_Alpha", "Billing_Retention_Team_Beta", "Tech_Support_Tier1"]
    return teams


@router.get("/sample-conversations")
async def get_sample_conversations(limit: int = 15, shuffle: bool = True):
    """Fetch sample conversations from the telecom dataset for interactive testing."""
    samples = corpus_loader.load_sample_conversations(limit_convs=limit, shuffle=shuffle)
    return [
        {
            "conversation_id": s.conversation_id,
            "agent_id": s.agent_id,
            "team_id": s.team_id,
            "turn_count": len(s.turns),
            "preview": s.turns[1].text if len(s.turns) > 1 else s.turns[0].text
        }
        for s in samples
    ]


@router.get("/sample-conversation/{conv_id}")
async def get_conversation_detail(conv_id: str):
    """Fetch complete turns for a given conversation ID."""
    conv = corpus_loader.get_conversation_by_id(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation {conv_id} not found.")
    return conv
