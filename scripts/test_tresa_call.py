from src.models.schemas import Turn, Speaker, LiveTurnInput
from src.analytics.live_assist import LiveAssistEngine

engine = LiveAssistEngine()

turns = [
    Turn(turn_id=1, speaker=Speaker.CLIENT, text="Hello, I'm calling to cancel my mobile service with Union Mobile."),
    Turn(turn_id=2, speaker=Speaker.AGENT, text="Hi Tresa, sorry to hear that you're considering canceling your service. Can you tell me a little bit more about why you're looking to cancel?"),
    Turn(turn_id=3, speaker=Speaker.CLIENT, text="Well, I just don't have good coverage in my area. I've been having trouble getting signal and it's really frustrating."),
    Turn(turn_id=4, speaker=Speaker.AGENT, text="I understand how frustrating that must be. Unfortunately, we may not be able to offer you a different plan or solution that will improve your coverage. However, I can certainly assist you with the cancellation process."),
    Turn(turn_id=5, speaker=Speaker.CLIENT, text="That's fine. Can you just cancel my service now?"),
    Turn(turn_id=6, speaker=Speaker.AGENT, text="Of course, Tresa. Before we proceed, I just want to make sure that you're aware that canceling your service will mean that you'll no longer be able to use your phone number or access any data or minutes associated with your plan. Is that understood?"),
]

for i in range(len(turns)):
    current = turns[i]
    history = turns[:i]
    res = engine.process_turn(LiveTurnInput(conversation_id="conv_tresa", current_turn=current, history=history))
    print(f"--- Turn #{current.turn_id} ({current.speaker.value.upper()}) ---")
    print(f"Text: {current.text[:60]}...")
    if res.compliance_alerts:
        print(f"ALERT: {res.compliance_alerts[0]}")
    if res.recommended_actions:
        print(f"NBA: [{res.recommended_actions[0].urgency.upper()}] {res.recommended_actions[0].title}")
    else:
        print("NBA: None")
    print()
