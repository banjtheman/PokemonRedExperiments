valid_actions = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]


def get_action_indices(actions, valid_actions):
    if actions is None:
        print("No actions to process.")
        return []
    indices = []
    for action in actions:
        if action in valid_actions:
            index = valid_actions.index(action)
            indices.append(index)
        else:
            print(f"Invalid action: {action}")
            # Optionally, you can raise an error here instead of just printing.
    return indices


output = """
Thought Process:
In this particular image, we are situated just above Pallet Town, on Route 1, almost reaching our destination. Our goal is to head back to Pallet Town quickly, ensuring we steer clear of unnecessary grassy areas to avoid random encounters. Since we’re in a perfect spot that directly leads to Pallet Town, the plan is straightforward: move "DOWN" along the path, keeping it as efficient as possible. Every step here is about streamlining our travel and staying on clear paths to minimize the time spent battling wild Pokémon.

Twitch Commentary:
"Hey speedrun fans, we're zooming right back to Pallet Town! We’re taking the direct route down, avoiding those grass patches to our right. Let’s get our first Gym encounter in the bag as fast as we can. Watch as we carefully maneuver this path - every second counts in a speed run, and we’re here to slash our time as best we possibly can!"

Next moves:
["DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN"]  # This set of moves takes us straight into Pallet Town, swiftly and efficiently. Each move is chosen to keep us on the path, avoiding unnecessary diversions or Pokémon encounters.
"""


def extract_array(output_text):
    import re

    lines = output_text.split("\n")
    array_pattern = re.compile(r"\[.*?\]")  # Non-greedy match inside brackets
    for line in lines:
        match = array_pattern.search(line)
        if match:
            try:
                return eval(match.group(0))
            except SyntaxError:
                continue  # In case of a syntax error in eval, continue to next line
    return None


actions = extract_array(output)
print(actions)
print(get_action_indices(actions, valid_actions))
