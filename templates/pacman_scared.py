legal = state.get_legal_actions()
if 'Stop' in legal:
    legal.remove('Stop')
pos = state.get_pacman_position()
ghost_positions = state.get_ghost_positions()

best_action = legal[0]
best_score = float('-inf')

for action in legal:
    successor = state.generate_pacman_successor(action)
    new_pos = successor.get_pacman_position()

    min_ghost_dist = min(
        abs(new_pos[0] - g[0]) + abs(new_pos[1] - g[1])
        for g in ghost_positions
    ) if ghost_positions else 100

    food_left = successor.get_num_food()
    score = successor.get_score() - state.get_score()

    if min_ghost_dist < 3:
        score += min_ghost_dist * 10
    else:
        score += 5

    if food_left < state.get_num_food():
        score += 20

    if score > best_score:
        best_score = score
        best_action = action

return best_action
