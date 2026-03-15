legal = state.get_legal_actions()
if 'Stop' in legal:
    legal.remove('Stop')
food = state.get_food().as_list()
pos = state.get_pacman_position()

if not food:
    return legal[0] if legal else 'Stop'

closest_food = min(food, key=lambda f: abs(f[0] - pos[0]) + abs(f[1] - pos[1]))

best_action = legal[0]
best_dist = float('inf')
for action in legal:
    successor = state.generate_pacman_successor(action)
    new_pos = successor.get_pacman_position()
    dist = abs(new_pos[0] - closest_food[0]) + abs(new_pos[1] - closest_food[1])
    if dist < best_dist:
        best_dist = dist
        best_action = action

return best_action
