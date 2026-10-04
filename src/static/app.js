// Contact Center Conversation Intelligence State
let currentConversation = null;
let currentTurnIndex = 0;
let simulatedHistory = [];
let allSampleConversations = [];

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  await loadSampleList();
  await loadTeams();
  await refreshHealthMetrics();
});

// Navigation Tabs
function setupNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      document.getElementById(tabId).classList.add("active");

      if (tabId === "tab-rollups") {
        loadTeamRollup();
      } else if (tabId === "tab-health") {
        refreshHealthMetrics();
      }
    });
  });
}

// 1. Load Sample Conversations from Corpus
async function loadSampleList() {
  try {
    const res = await fetch("/qa/sample-conversations?limit=15");
    allSampleConversations = await res.json();
    const select = document.getElementById("convSelect");
    select.innerHTML = "";

    allSampleConversations.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.conversation_id;
      opt.textContent = `[${c.team_id}] Agent: ${c.agent_id} (${c.turn_count} turns) - ${c.preview.substring(0, 45)}...`;
      select.appendChild(opt);
    });

    if (allSampleConversations.length > 0) {
      select.value = allSampleConversations[0].conversation_id;
      await selectConversation();
    }
  } catch (err) {
    console.error("Error loading sample conversations:", err);
  }
}

async function selectConversation() {
  const convId = document.getElementById("convSelect").value;
  try {
    const res = await fetch(`/qa/sample-conversation/${convId}`);
    currentConversation = await res.json();
    resetLiveSimulator();
    await runBatchAnalysis();
  } catch (err) {
    console.error("Error selecting conversation:", err);
  }
}

// 2. Live Assist Simulator
function resetLiveSimulator() {
  currentTurnIndex = 0;
  simulatedHistory = [];
  document.getElementById("liveChatBox").innerHTML = "";
  document.getElementById("liveAlertsBox").innerHTML = "";
  document.getElementById("liveActionsBox").innerHTML = `
    <div style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 2rem;">
      Ready. Click "Step Next Turn" or "Play Live Call" to stream interaction.
    </div>
  `;
  document.getElementById("liveSentimentLabel").textContent = "Neutral (0.0)";
  document.getElementById("liveSentimentTrend").textContent = "Stable";
  document.getElementById("stepTurnBtn").disabled = false;
}

async function stepNextTurn() {
  if (!currentConversation || currentTurnIndex >= currentConversation.turns.length) {
    return;
  }

  const currentTurn = currentConversation.turns[currentTurnIndex];
  
  // Render chat bubble
  renderChatBubble("liveChatBox", currentTurn);

  // Call /analyze/stream-turn
  try {
    const payload = {
      conversation_id: currentConversation.conversation_id,
      agent_id: currentConversation.agent_id,
      team_id: currentConversation.team_id,
      current_turn: currentTurn,
      history: simulatedHistory
    };

    const res = await fetch("/analyze/stream-turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const liveData = await res.json();

    // Update sentiment meters with speaker & cumulative relationship state awareness
    const custScore = liveData.customer_sentiment !== undefined ? liveData.customer_sentiment : liveData.turn_sentiment;
    const custLabel = liveData.customer_sentiment_label || liveData.sentiment_label;
    const custScoreText = custScore > 0 ? `+${custScore}` : `${custScore}`;
    const custBadgeClass = custLabel === 'positive' ? 'badge-emerald' : custLabel === 'negative' ? 'badge-rose' : 'badge-indigo';

    const speakerLabel = liveData.turn_speaker === 'agent' ? 'Agent' : 'Customer';
    const turnScoreText = liveData.turn_sentiment > 0 ? `+${liveData.turn_sentiment}` : `${liveData.turn_sentiment}`;
    const stateText = liveData.customer_state || "In Progress";

    document.getElementById("liveSentimentLabel").innerHTML = `
      <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.25rem;">
        <span class="badge ${custBadgeClass}" style="font-size: 0.85rem; padding: 0.3rem 0.7rem;">
          Customer: ${custLabel.toUpperCase()} (${custScoreText}) • ${stateText}
        </span>
        <span style="font-size: 0.72rem; color: var(--text-muted);">
          Latest Turn #${liveData.turn_id} by ${speakerLabel}: ${liveData.sentiment_label} (${turnScoreText})
        </span>
      </div>
    `;

    // Trend badge
    const trendBadge = document.getElementById("liveSentimentTrend");
    trendBadge.textContent = `TREND: ${liveData.running_sentiment_trend.toUpperCase()}`;
    trendBadge.className = `badge ${liveData.running_sentiment_trend === 'improving' ? 'badge-emerald' : liveData.running_sentiment_trend === 'deteriorating' ? 'badge-rose' : 'badge-cyan'}`;

    // Render compliance alerts
    const alertsBox = document.getElementById("liveAlertsBox");
    alertsBox.innerHTML = "";
    if (liveData.compliance_alerts && liveData.compliance_alerts.length > 0) {
      liveData.compliance_alerts.forEach(alert => {
        const alertEl = document.createElement("div");
        alertEl.className = "alert-banner";
        alertEl.innerHTML = `⚠️ <span>${alert}</span>`;
        alertsBox.appendChild(alertEl);
      });
    }

    // Render recommended next best actions
    const actionsBox = document.getElementById("liveActionsBox");
    actionsBox.innerHTML = "";
    if (liveData.recommended_actions && liveData.recommended_actions.length > 0) {
      const urgencyLabels = {
        critical: "URGENCY: CRITICAL (LEGAL COMPLIANCE)",
        high: "URGENCY: HIGH (ACTION REQUIRED)",
        medium: "URGENCY: MEDIUM (PROACTIVE)",
        low: "PRIORITY: LOW (STANDARD CONVERSATION)"
      };

      liveData.recommended_actions.forEach(action => {
        const urgencyText = urgencyLabels[action.urgency] || `URGENCY: ${action.urgency.toUpperCase()}`;
        const card = document.createElement("div");
        card.className = `action-card ${action.urgency}`;
        card.innerHTML = `
          <div class="action-header">
            <span class="action-title">${action.title}</span>
            <span class="badge ${action.urgency === 'critical' ? 'badge-rose' : action.urgency === 'high' ? 'badge-amber' : 'badge-indigo'}">
              ${urgencyText}
            </span>
          </div>
          <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.2rem;">${action.trigger_reason}</div>
          <div class="script-box" style="margin-top: 0.4rem;">"${action.recommended_script}"</div>
        `;
        actionsBox.appendChild(card);
      });
    }

    simulatedHistory.push(currentTurn);
    currentTurnIndex++;

    if (currentTurnIndex >= currentConversation.turns.length) {
      document.getElementById("stepTurnBtn").disabled = true;
    }
  } catch (err) {
    console.error("Error stepping turn:", err);
  }
}

let isPlaying = false;
async function playLiveCall() {
  if (isPlaying) return;
  isPlaying = true;
  document.getElementById("playCallBtn").disabled = true;

  while (currentTurnIndex < currentConversation.turns.length && isPlaying) {
    await stepNextTurn();
    await new Promise(r => setTimeout(r, 1200));
  }

  isPlaying = false;
  document.getElementById("playCallBtn").disabled = false;
}

// 3. Batch Call Analytics & Grounded QA
async function runBatchAnalysis() {
  if (!currentConversation) return;

  try {
    const res = await fetch("/analyze/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentConversation)
    });
    const analysis = await res.json();

    // Render batch transcript
    const batchChatBox = document.getElementById("batchChatBox");
    batchChatBox.innerHTML = "";
    currentConversation.turns.forEach(t => renderChatBubble("batchChatBox", t));

    // Summary & KPIs
    document.getElementById("kpiSummary").textContent = analysis.concise_summary;
    document.getElementById("kpiScore").textContent = `${analysis.qa_score}/100`;
    document.getElementById("kpiScore").style.color = analysis.qa_passed ? "var(--accent-emerald)" : "var(--accent-rose)";

    // Resolution & Churn
    const resBadge = document.getElementById("kpiResolution");
    resBadge.textContent = analysis.resolution.status;
    resBadge.className = `badge ${analysis.resolution.status === 'RESOLVED' ? 'badge-emerald' : analysis.resolution.status === 'ESCALATED' ? 'badge-amber' : 'badge-rose'}`;

    const churnBadge = document.getElementById("kpiChurn");
    churnBadge.textContent = `${analysis.churn_risk.risk_level} (${Math.round(analysis.churn_risk.risk_score * 100)}%)`;
    churnBadge.className = `badge ${analysis.churn_risk.risk_level === 'CRITICAL' || analysis.churn_risk.risk_level === 'HIGH' ? 'badge-rose' : analysis.churn_risk.risk_level === 'MEDIUM' ? 'badge-amber' : 'badge-emerald'}`;

    // Sentiment Arc
    const arc = analysis.sentiment_arc;
    document.getElementById("kpiArc").textContent = `${arc.start_label.toUpperCase()} (${arc.start_sentiment}) → ${arc.end_label.toUpperCase()} (${arc.end_sentiment}) [${arc.trajectory}]`;

    // Multi-Label Call Reasons
    const reasonsContainer = document.getElementById("reasonsContainer");
    reasonsContainer.innerHTML = "";
    analysis.call_reasons.forEach(r => {
      const tag = document.createElement("div");
      tag.style.marginBottom = "0.5rem";
      tag.innerHTML = `
        <span class="badge badge-indigo">${r.label}</span>
        <span style="font-size: 0.78rem; color: var(--text-muted); margin-left: 0.4rem;">${Math.round(r.confidence * 100)}% conf - ${r.explanation}</span>
      `;
      reasonsContainer.appendChild(tag);
    });

    // Follow-Up Actions
    const followUpsContainer = document.getElementById("followUpsContainer");
    followUpsContainer.innerHTML = "";
    analysis.follow_up_actions.forEach(act => {
      const item = document.createElement("li");
      item.style.fontSize = "0.85rem";
      item.style.marginBottom = "0.4rem";
      item.textContent = act;
      followUpsContainer.appendChild(item);
    });

    // QA Checklist Scorecard with Grounded Evidence
    const qaContainer = document.getElementById("qaDetailsContainer");
    qaContainer.innerHTML = "";

    analysis.qa_details.forEach(item => {
      const row = document.createElement("div");
      row.className = "qa-item";

      const hasQuotes = item.quoted_evidence && item.quoted_evidence.length > 0;
      let quoteHtml = "";
      if (hasQuotes) {
        quoteHtml = `
          <div class="qa-quote-box" title="Click to view highlighted quote in transcript">
            <strong>Grounded Quote:</strong> "${item.quoted_evidence[0]}"
            <div style="font-size: 0.7rem; color: #a5b4fc; margin-top: 0.2rem;">
              Turns: #${item.evidence_turn_indices.join(", #")} | Grounding: 100% Verbatim Verified
            </div>
          </div>
        `;
      }

      row.innerHTML = `
        <div class="qa-item-header">
          <div class="qa-item-name">
            <span>${item.name}</span>
            <span class="badge ${item.passed ? 'badge-emerald' : 'badge-rose'}">
              ${item.passed ? 'PASS (' + item.score + ')' : 'FAIL (' + item.score + ')'}
            </span>
            <span class="badge badge-indigo">${item.category}</span>
          </div>
          <span style="font-size: 0.8rem; color: var(--text-muted);">Weight: ${item.weight}%</span>
        </div>
        <div style="font-size: 0.82rem; color: var(--text-secondary);">${item.explanation}</div>
        ${quoteHtml}
      `;

      // Highlight turn on click
      row.addEventListener("click", () => {
        highlightTranscriptTurns("batchChatBox", item.evidence_turn_indices);
      });

      qaContainer.appendChild(row);
    });

  } catch (err) {
    console.error("Error running batch analysis:", err);
  }
}

// 4. Highlight Transcript Turns
function highlightTranscriptTurns(boxId, turnIds) {
  const box = document.getElementById(boxId);
  const bubbles = box.querySelectorAll(".chat-bubble");
  bubbles.forEach(b => b.classList.remove("highlighted"));

  if (!turnIds || turnIds.length === 0) return;

  turnIds.forEach(tid => {
    const target = box.querySelector(`[data-turn-id='${tid}']`);
    if (target) {
      target.classList.add("highlighted");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
}

function renderChatBubble(boxId, turn) {
  const box = document.getElementById(boxId);
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${turn.speaker}`;
  bubble.setAttribute("data-turn-id", turn.turn_id);

  const speakerName = turn.speaker === "agent" ? (currentConversation.agent_id || "Agent") : "Customer";
  bubble.innerHTML = `
    <div class="bubble-meta">
      <strong>${speakerName} (Turn #${turn.turn_id})</strong>
      <span>${turn.timestamp ? turn.timestamp.split("T")[1].substring(0, 8) : ""}</span>
    </div>
    <div>${turn.text}</div>
  `;
  box.appendChild(bubble);
  box.scrollTop = box.scrollHeight;
}

// 5. Supervisor Team Rollups
async function loadTeams() {
  try {
    const res = await fetch("/qa/rollups/teams");
    const teams = await res.json();
    const select = document.getElementById("teamSelect");
    select.innerHTML = "";

    teams.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t.replace(/_/g, " ");
      select.appendChild(opt);
    });

    if (teams.length > 0) {
      await loadTeamRollup();
    }
  } catch (err) {
    console.error("Error loading teams:", err);
  }
}

async function loadTeamRollup() {
  const teamId = document.getElementById("teamSelect").value;
  if (!teamId) return;

  try {
    const res = await fetch(`/qa/rollups/team/${teamId}`);
    const data = await res.json();

    document.getElementById("teamCallsCount").textContent = data.total_calls;
    document.getElementById("teamAvgScore").textContent = `${data.average_score}%`;
    document.getElementById("teamComplianceRate").textContent = `${data.compliance_pass_rate}%`;
    document.getElementById("teamContainmentRate").textContent = `${data.churn_containment_rate}%`;

    // Leaderboard
    const tbody = document.getElementById("leaderboardBody");
    tbody.innerHTML = "";

    data.agent_rankings.forEach((ag, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>#${idx + 1}</strong></td>
        <td><strong>${ag.agent_id}</strong></td>
        <td>${ag.total_calls_analyzed}</td>
        <td><strong style="color: ${ag.average_qa_score >= 80 ? 'var(--accent-emerald)' : 'var(--accent-amber)'}">${ag.average_qa_score}</strong></td>
        <td><span class="badge ${ag.pass_rate >= 80 ? 'badge-emerald' : 'badge-rose'}">${ag.pass_rate}%</span></td>
        <td><span class="badge ${ag.critical_violation_count === 0 ? 'badge-emerald' : 'badge-rose'}">${ag.critical_violation_count}</span></td>
        <td style="font-size: 0.78rem; color: #cbd5e1;">${ag.coaching_tips[0] || 'Good standing'}</td>
      `;
      tbody.appendChild(tr);
    });

    // Breakdown bars
    const breakdownBox = document.getElementById("qaBreakdownBars");
    breakdownBox.innerHTML = "";
    for (const [key, pct] of Object.entries(data.qa_item_breakdown)) {
      const barItem = document.createElement("div");
      barItem.style.marginBottom = "0.8rem";
      barItem.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
          <span>${key.replace(/_/g, ' ').toUpperCase()}</span>
          <strong>${pct}% Pass</strong>
        </div>
        <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
          <div style="background: ${pct >= 85 ? 'var(--accent-emerald)' : pct >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)'}; width: ${pct}%; height: 100%;"></div>
        </div>
      `;
      breakdownBox.appendChild(barItem);
    }
  } catch (err) {
    console.error("Error loading team rollup:", err);
  }
}

// 6. System Health & Evals
async function refreshHealthMetrics() {
  try {
    const res = await fetch("/health");
    const h = await res.json();

    document.getElementById("healthStatus").textContent = h.status;
    document.getElementById("healthUptime").textContent = `${h.uptime_seconds}s`;
    document.getElementById("healthBatchCalls").textContent = h.total_calls_processed;
    document.getElementById("healthStreamTurns").textContent = h.total_stream_turns_processed;
    document.getElementById("healthBatchLat").textContent = `${h.average_batch_latency_ms} ms`;
    document.getElementById("healthStreamLat").textContent = `${h.average_stream_latency_ms} ms`;
    document.getElementById("healthGroundedness").textContent = `${h.groundedness_rate_pct}%`;
    document.getElementById("healthViolations").textContent = `${h.compliance_violation_rate_pct}%`;

    // Prometheus raw preview
    const promRes = await fetch("/metrics");
    const promText = await promRes.text();
    document.getElementById("prometheusPreview").textContent = promText;
  } catch (err) {
    console.error("Error refreshing health metrics:", err);
  }
}
