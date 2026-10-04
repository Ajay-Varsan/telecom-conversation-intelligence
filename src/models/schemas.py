from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Speaker(str, Enum):
    AGENT = "agent"
    CLIENT = "client"
    SYSTEM = "system"


class Turn(BaseModel):
    turn_id: int
    speaker: Speaker
    text: str
    timestamp: Optional[str] = None
    sentiment: Optional[float] = None
    sentiment_label: Optional[str] = None


class TranscriptInput(BaseModel):
    conversation_id: str
    agent_id: Optional[str] = "agent_unknown"
    team_id: Optional[str] = "general_support"
    turns: List[Turn]


class LiveTurnInput(BaseModel):
    conversation_id: str
    agent_id: Optional[str] = "agent_unknown"
    team_id: Optional[str] = "general_support"
    current_turn: Turn
    history: List[Turn] = Field(default_factory=list)


class LiveActionRecommendation(BaseModel):
    action_type: str
    title: str
    recommended_script: str
    urgency: str = "medium"  # low, medium, high, critical
    trigger_reason: str


class LiveAssistResponse(BaseModel):
    conversation_id: str
    turn_id: int
    turn_speaker: Speaker = Speaker.CLIENT
    turn_sentiment: float
    sentiment_label: str
    customer_sentiment: float = 0.0
    customer_sentiment_label: str = "neutral"
    running_sentiment_trend: str
    compliance_alerts: List[str] = Field(default_factory=list)
    recommended_actions: List[LiveActionRecommendation] = Field(default_factory=list)
    latency_ms: float = 0.0


class CallReason(BaseModel):
    label: str
    confidence: float
    evidence_turns: List[int] = Field(default_factory=list)
    explanation: str


class SentimentArc(BaseModel):
    start_sentiment: float
    middle_sentiment: float
    end_sentiment: float
    start_label: str
    end_label: str
    trajectory: str  # positive_recovery, negative_escalation, consistently_negative, consistently_positive, neutral
    turn_sentiments: List[float] = Field(default_factory=list)


class ResolutionStatus(BaseModel):
    status: str  # RESOLVED, UNRESOLVED, ESCALATED, PENDING_CUSTOMER_ACTION
    confidence: float
    explanation: str


class ChurnRisk(BaseModel):
    is_risk: bool
    risk_score: float  # 0.0 to 1.0
    risk_level: str    # LOW, MEDIUM, HIGH, CRITICAL
    drivers: List[str] = Field(default_factory=list)
    competitor_mentioned: Optional[str] = None


class QAChecklistConfig(BaseModel):
    checklist_id: str = "default_telecom_v1"
    name: str = "Standard Telecom Contact Center QA Rubric"
    items: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "greeting": {
                "name": "Professional Greeting & Name Introduction",
                "category": "SERVICE_QUALITY",
                "weight": 10.0,
                "is_critical": False,
                "description": "Agent greeted caller politely and introduced themselves and company name."
            },
            "identity_verification": {
                "name": "CPNI & Customer Identity Verification",
                "category": "COMPLIANCE",
                "weight": 25.0,
                "is_critical": True,
                "description": "Agent verified customer identity (PIN, last 4 digits, account number) before account actions."
            },
            "empathy": {
                "name": "Empathy & Frustration Acknowledgment",
                "category": "SERVICE_QUALITY",
                "weight": 15.0,
                "is_critical": False,
                "description": "Agent acknowledged customer frustration, apologized for service issues, or validated concerns."
            },
            "correct_disclosure": {
                "name": "Mandatory Terms, Fees & Disclosures",
                "category": "COMPLIANCE",
                "weight": 20.0,
                "is_critical": True,
                "description": "Agent explicitly disclosed cancellation fees, return requirements, activation costs, or plan terms."
            },
            "no_prohibited_promises": {
                "name": "Zero Prohibited / Deceptive Promises",
                "category": "COMPLIANCE",
                "weight": 20.0,
                "is_critical": True,
                "description": "Agent did NOT make unauthorized guarantees, deceptive 'free' device offers without terms, or impossible SLA promises."
            },
            "proper_closure": {
                "name": "Comprehensive Closure & Survey / Branding",
                "category": "SERVICE_QUALITY",
                "weight": 10.0,
                "is_critical": False,
                "description": "Agent offered additional assistance, thanked caller for choosing company, and closed professionally."
            }
        }
    )


class QAScoreResult(BaseModel):
    item_id: str
    name: str
    category: str
    passed: bool
    score: float
    weight: float
    quoted_evidence: List[str] = Field(default_factory=list)
    evidence_turn_indices: List[int] = Field(default_factory=list)
    is_grounded: bool = True
    explanation: str
    is_violation: bool = False
    severity: str = "NONE"


class ConversationAnalysisResponse(BaseModel):
    conversation_id: str
    agent_id: str
    team_id: str
    concise_summary: str
    call_reasons: List[CallReason]
    sentiment_arc: SentimentArc
    resolution: ResolutionStatus
    churn_risk: ChurnRisk
    follow_up_actions: List[str]
    qa_score: float
    qa_passed: bool
    critical_compliance_violation: bool
    qa_details: List[QAScoreResult]
    audit_metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentScorecard(BaseModel):
    agent_id: str
    team_id: str
    total_calls_analyzed: int
    average_qa_score: float
    pass_rate: float
    critical_violation_count: int
    top_failure_reasons: List[str]
    coaching_tips: List[str]


class TeamRollup(BaseModel):
    team_id: str
    total_calls: int
    average_score: float
    compliance_pass_rate: float
    churn_containment_rate: float
    agent_rankings: List[AgentScorecard]
    qa_item_breakdown: Dict[str, float]


class SystemHealthResponse(BaseModel):
    status: str
    service_version: str
    uptime_seconds: float
    total_calls_processed: int
    total_stream_turns_processed: int
    average_batch_latency_ms: float
    average_stream_latency_ms: float
    groundedness_rate_pct: float
    compliance_violation_rate_pct: float
