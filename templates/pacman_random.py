legal = state.get_legal_actions()
if 'Stop' in legal and len(legal) > 1:
    legal.remove('Stop')
return random.choice(legal)
