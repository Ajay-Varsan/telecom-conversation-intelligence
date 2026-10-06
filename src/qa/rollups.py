from collections import defaultdict
from typing import Dict, List, Any
from src.models.schemas import (
    ConversationAnalysisResponse,
    AgentScorecard,
    TeamRollup
)


class RollupManager:
    """Computes and tracks agent-level scorecards and supervisor team-level rollups."""

    def __init__(self):
        # Key: conv_id -> ConversationAnalysisResponse
        self.analyses_by_id: Dict[str, ConversationAnalysisResponse] = {}
        # Key: agent_id -> list of ConversationAnalysisResponse
        self.agent_records: Dict[str, List[ConversationAnalysisResponse]] = defaultdict(list)
        # Key: team_id -> list of ConversationAnalysisResponse
        self.team_records: Dict[str, List[ConversationAnalysisResponse]] = defaultdict(list)

    def record_analysis(self, analysis: ConversationAnalysisResponse):
        # Deduplicate by conversation_id to avoid inflating counts on UI reruns
        self.analyses_by_id[analysis.conversation_id] = analysis
        self._rebuild_indices()

    def _rebuild_indices(self):
        self.agent_records = defaultdict(list)
        self.team_records = defaultdict(list)
        for analysis in self.analyses_by_id.values():
            self.agent_records[analysis.agent_id].append(analysis)
            self.team_records[analysis.team_id].append(analysis)

    def get_agent_scorecard(self, agent_id: str) -> AgentScorecard:
        records = self.agent_records.get(agent_id, [])
        if not records:
            return AgentScorecard(
                agent_id=agent_id,
                team_id="unassigned",
                total_calls_analyzed=0,
                average_qa_score=0.0,
                pass_rate=0.0,
                critical_violation_count=0,
                top_failure_reasons=[],
                coaching_tips=["No calls recorded yet for this agent."]
            )

        team_id = records[0].team_id
        total_calls = len(records)
        avg_score = round(sum(r.qa_score for r in records) / total_calls, 1)
        passed_calls = sum(1 for r in records if r.qa_passed)
        pass_rate = round((passed_calls / total_calls) * 100.0, 1)
        critical_violations = sum(1 for r in records if r.critical_compliance_violation)

        # Tally failure reasons across QA checklist
        failure_counts = defaultdict(int)
        for r in records:
            for detail in r.qa_details:
                if not detail.passed or detail.is_violation:
                    failure_counts[detail.name] += 1

        top_failures = sorted(failure_counts.keys(), key=lambda k: failure_counts[k], reverse=True)[:3]

        # Generate targeted coaching tips
        coaching_tips = []
        if "CPNI & Customer Identity Verification" in top_failures:
            coaching_tips.append("Mandatory Training: Re-certify on CPNI verification rules before discussing any cancellation or account alteration.")
        if "Zero Prohibited / Deceptive Promises" in top_failures:
            coaching_tips.append("Compliance Warning: Do not offer 'free' devices without reading full 24-month installment contract disclosures.")
        if "Mandatory Terms, Fees & Disclosures" in top_failures:
            coaching_tips.append("Process Adherence: Always state the $10-$25 cancellation/activation fee policy and 14-day equipment return timeline.")
        if "Empathy & Frustration Acknowledgment" in top_failures:
            coaching_tips.append("Soft Skills Coaching: Acknowledge caller frustration immediately upon hearing dropped calls or billing complaints.")
        if not coaching_tips:
            coaching_tips.append("Excellent performance! Continue maintaining compliance rigor and strong customer rapport.")

        return AgentScorecard(
            agent_id=agent_id,
            team_id=team_id,
            total_calls_analyzed=total_calls,
            average_qa_score=avg_score,
            pass_rate=pass_rate,
            critical_violation_count=critical_violations,
            top_failure_reasons=top_failures,
            coaching_tips=coaching_tips
        )

    def get_team_rollup(self, team_id: str) -> TeamRollup:
        records = self.team_records.get(team_id, [])
        if not records:
            return TeamRollup(
                team_id=team_id,
                total_calls=0,
                average_score=0.0,
                compliance_pass_rate=0.0,
                churn_containment_rate=0.0,
                agent_rankings=[],
                qa_item_breakdown={}
            )

        total_calls = len(records)
        avg_score = round(sum(r.qa_score for r in records) / total_calls, 1)
        passed_calls = sum(1 for r in records if not r.critical_compliance_violation)
        compliance_pass_rate = round((passed_calls / total_calls) * 100.0, 1)

        # Churn containment: calls with churn risk that did not end in cancellation / account termination
        churn_risk_calls = [
            r for r in records
            if r.churn_risk.is_risk or any("cancel" in d.lower() or "competitor" in d.lower() for d in r.churn_risk.drivers)
        ]
        if churn_risk_calls:
            contained_calls = [
                r for r in churn_risk_calls
                if (
                    "Customer tentatively accepted retention offer / plan adjustment" in r.churn_risk.drivers
                    or (
                        r.resolution.status == "RESOLVED"
                        and "Service cancellation processed / confirmed" not in r.churn_risk.drivers
                        and "service termination workflow" not in str(r.concise_summary).lower()
                    )
                    or (r.churn_risk.risk_score < 0.50 and r.resolution.status != "UNRESOLVED")
                )
            ]
            containment_rate = round((len(contained_calls) / len(churn_risk_calls)) * 100.0, 1)
        else:
            containment_rate = 100.0

        # Item-by-item pass rate breakdown
        item_pass_counts = defaultdict(int)
        item_total_counts = defaultdict(int)
        for r in records:
            for d in r.qa_details:
                item_total_counts[d.item_id] += 1
                if d.passed:
                    item_pass_counts[d.item_id] += 1

        qa_item_breakdown = {
            item_id: round((item_pass_counts[item_id] / item_total_counts[item_id]) * 100.0, 1)
            for item_id in item_total_counts
        }

        # Unique agents on this team
        agent_ids = set(r.agent_id for r in records)
        agent_scorecards = [self.get_agent_scorecard(aid) for aid in agent_ids]
        # Rank by average score descending, then least violations
        agent_scorecards.sort(key=lambda s: (s.average_qa_score, -s.critical_violation_count), reverse=True)

        return TeamRollup(
            team_id=team_id,
            total_calls=total_calls,
            average_score=avg_score,
            compliance_pass_rate=compliance_pass_rate,
            churn_containment_rate=containment_rate,
            agent_rankings=agent_scorecards,
            qa_item_breakdown=qa_item_breakdown
        )
