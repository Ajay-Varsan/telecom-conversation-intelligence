from src.models.schemas import Turn, Speaker, LiveTurnInput
from src.analytics.live_assist import LiveAssistEngine

engine = LiveAssistEngine()

turns = [
    Turn(turn_id=1, speaker=Speaker.AGENT, text="Thank you for calling Union Mobile. My name is Julia. How may I assist you today?"),
    Turn(turn_id=2, speaker=Speaker.CLIENT, text="Hi Julia, I'm calling to cancel my mobile service."),
    Turn(turn_id=3, speaker=Speaker.AGENT, text="I can help with that, but first could you please verify your account PIN or the last 4 digits of your card?"),
    Turn(turn_id=4, speaker=Speaker.CLIENT, text="Sure, my PIN is 4821."),
    Turn(turn_id=5, speaker=Speaker.AGENT, text="Thanks! I've located your account. You're currently on our 5GB Unlimited Plan for $45/month."),
    Turn(turn_id=6, speaker=Speaker.CLIENT, text="Right. But Mint Mobile is offering me a better deal and a free phone, so I really just want to switch."),
    Turn(turn_id=7, speaker=Speaker.AGENT, text="I understand. What if I offer you our loyalty discount of $10 off your bill every month?"),
    Turn(turn_id=8, speaker=Speaker.CLIENT, text="No thanks, I've made up my mind. Just cancel it."),
]

print("=== EVALUATING TURN 2 (Cancel request before auth) ===")
res2 = engine.process_turn(LiveTurnInput(conversation_id="conv-1", current_turn=turns[1], history=turns[:1]))
print("Action Type:", res2.recommended_actions[0].action_type)
print("Urgency:", res2.recommended_actions[0].urgency)
print("Customer State:", res2.customer_state)

print("\n=== EVALUATING TURN 6 (Competitor switch after auth) ===")
res6 = engine.process_turn(LiveTurnInput(conversation_id="conv-1", current_turn=turns[5], history=turns[:5]))
print("Action Type:", res6.recommended_actions[0].action_type)
print("Action Title:", res6.recommended_actions[0].title)
print("Urgency:", res6.recommended_actions[0].urgency)
print("Customer State:", res6.customer_state)
print("Turn Sentiment:", res6.sentiment_label)
print("Customer Sentiment:", res6.customer_sentiment)

print("\n=== EVALUATING TURN 8 (Customer insists on cancel) ===")
res8 = engine.process_turn(LiveTurnInput(conversation_id="conv-1", current_turn=turns[7], history=turns[:7]))
print("Action Type:", res8.recommended_actions[0].action_type)
print("Action Title:", res8.recommended_actions[0].title)
print("Urgency:", res8.recommended_actions[0].urgency)
print("Customer State:", res8.customer_state)
