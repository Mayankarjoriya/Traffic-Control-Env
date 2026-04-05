def create_prompt(state):
    prompt = f"""Current traffic state:
North lane: {} cars, waited {???} seconds
South lane: {???} cars, waited {???} seconds
East lane: {???} cars, waited {???} seconds
West lane: {???} cars, waited {???} seconds

Choose one action from: NS_GREEN, EW_GREEN, NE_GREEN, SW_GREEN
Reply with only the action name."""
    
    return prompt