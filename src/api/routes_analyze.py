import time
from fastapi import APIRouter, HTTPException
from src.models.schemas import (
    TranscriptInput,
    ConversationAnalysisResponse,
    LiveTurnInput,
    LiveAssistResponse
)
from src.analytics.sentiment_analyzer import SentimentAnalyzer
from src.analytics.reason_classifier import ReasonClassifier
from src.analytics.churn_detector import ChurnAndResolutionDetector
from src.analytics.summarizer import ConversationSummarizer
from src.analytics.live_assist import LiveAssistEngine
from src.qa.checklist import ChecklistManager
from src.qa.evaluator import QAEvaluator
from src.qa.rollups import RollupManager
from src.data.corpus_loader import CorpusLoader

router = APIRouter(prefix="/analyze", tags=["Conversation Analytics"])

# Singletons for microservice state
sentiment_analyzer = SentimentAnalyzer()
reason_classifier = ReasonClassifier()
churn_detector = ChurnAndResolutionDetector()
summarizer = ConversationSummarizer()
live_assist_engine = LiveAssistEngine()
checklist_manager = ChecklistManager()
rollup_manager = RollupManager()
corpus_loader = CorpusLoader()

# Metrics state
metrics_state = {
    "total_batch_calls": 0,
    "total_stream_turns": 0,
    "batch_latencies_ms": [],
    "stream_latencies_ms": [],
    "total_quotes_verified": 0,
    "grounded_quotes_verified": 0,
    "compliance_violations_count": 0
}


@router.post("/batch", response_model=ConversationAnalysisResponse)
async def analyze_batch_conversation(transcript: TranscriptInput):
    start_time = time.perf_counter()

    if not transcript.turns:
        raise HTTPException(status_code=400, detail="Transcript contains no conversation turns.")

    # 1. Enrich turns with turn-level sentiment
    enriched_turns = sentiment_analyzer.analyze_turns(transcript.turns)

    # 2. Sentiment Arc Trajectory
    sentiment_arc = sentiment_analyzer.compute_sentiment_arc(enriched_turns)

    # 3. Multi-label Call Reasons
    call_reasons = reason_classifier.classify(enriched_turns)
    reason_labels = [r.label for r in call_reasons]

    # 4. Churn Risk & Resolution Status
    churn_risk = churn_detector.evaluate_churn_risk(enriched_turns)
    resolution = churn_detector.evaluate_resolution(enriched_turns)

    # 5. Concise Summary & Follow-up Actions
    summary, follow_ups = summarizer.summarize(
        transcript.conversation_id, enriched_turns, reason_labels, churn_risk, resolution
    )

    # 6. QA Checklist Evaluation with Grounded Quotes
    qa_evaluator = QAEvaluator(checklist_manager.get_config())
    qa_score, qa_passed, critical_violation, qa_details = qa_evaluator.evaluate(enriched_turns)

    # Calculate audit & groundedness stats
    for d in qa_details:
        if d.quoted_evidence:
            metrics_state["total_quotes_verified"] += len(d.quoted_evidence)
            if d.is_grounded:
                metrics_state["grounded_quotes_verified"] += len(d.quoted_evidence)

    if critical_violation:
        metrics_state["compliance_violations_count"] += 1

    proc_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    metrics_state["total_batch_calls"] += 1
    metrics_state["batch_latencies_ms"].append(proc_time_ms)
    if len(metrics_state["batch_latencies_ms"]) > 1000:
        metrics_state["batch_latencies_ms"].pop(0)

    response = ConversationAnalysisResponse(
        conversation_id=transcript.conversation_id,
        agent_id=transcript.agent_id or "Agent_Unknown",
        team_id=transcript.team_id or "General_Support",
        concise_summary=summary,
        call_reasons=call_reasons,
        sentiment_arc=sentiment_arc,
        resolution=resolution,
        churn_risk=churn_risk,
        follow_up_actions=follow_ups,
        qa_score=qa_score,
        qa_passed=qa_passed,
        critical_compliance_violation=critical_violation,
        qa_details=qa_details,
        audit_metadata={
            "processing_time_ms": proc_time_ms,
            "turns_count": len(enriched_turns),
            "grounding_check": "deterministic_substring_verified"
        }
    )

    # Record in rollup manager for agent and team scorecards
    rollup_manager.record_analysis(response)

    return response


@router.post("/stream-turn", response_model=LiveAssistResponse)
async def process_stream_turn(turn_input: LiveTurnInput):
    start_time = time.perf_counter()

    result = live_assist_engine.process_turn(turn_input)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    result.latency_ms = latency_ms

    metrics_state["total_stream_turns"] += 1
    metrics_state["stream_latencies_ms"].append(latency_ms)
    if len(metrics_state["stream_latencies_ms"]) > 1000:
        metrics_state["stream_latencies_ms"].pop(0)

    return result
