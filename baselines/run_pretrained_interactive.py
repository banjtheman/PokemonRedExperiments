import base64
import json
import uuid
from io import BytesIO
from os.path import exists
from pathlib import Path

import anthropic
import boto3
import cv2
import google.generativeai as genai
from langchain_community.llms import Ollama
from openai import OpenAI
from PIL import Image
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from termcolor import colored

from red_gym_env import RedGymEnv

llava_model = Ollama(model="llava")
open_ai_client = OpenAI()


# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

opus_client = anthropic.Anthropic()


def call_claude_3_opus(system_prompt, prompt, base64_string):

    # image_bytes = base64.b64decode(base64_string)

    message = opus_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return message.content[0].text


def call_claude_haiku(system_prompt, prompt, base64_string):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


def call_claude_sonnet(system_prompt, prompt, base64_string):

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


def call_gpt4_vision(system_prompt, prompt, base64_string):

    response = open_ai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_string}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.1,
    )

    return response.choices[0].message.content


def call_gemini(system_prompt, prompt, base64_stirng):

    model = genai.GenerativeModel("gemini-pro-vision")

    image_bytes = base64.b64decode(base64_stirng)

    # Creating a byte stream from image bytes
    image_stream = BytesIO(image_bytes)

    # Loading the image stream into a PIL image
    game_state_pic = Image.open(image_stream)

    response = model.generate_content(
        [system_prompt + prompt, game_state_pic], stream=True
    )
    response.resolve()

    return response.text


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


def rgb_to_base64(rgb_array):
    # Create a new image from the RGB array
    image = Image.new("RGB", (len(rgb_array[0]), len(rgb_array)))

    # Populate the image with RGB values
    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixels[i, j] = tuple(rgb_array[j][i])

    # Create a BytesIO object to store the image data
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    # Encode the image data as base64
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_string


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = RedGymEnv(env_conf)
        # env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")
    ep_length = 2**23

    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "../has_pokedex_nballs.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "../PokemonRed.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": True,
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    env = make_env(
        0, env_config
    )()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # env_checker.check_env(env)
    file_name = "session_4da05e87_main_good/poke_439746560_steps"

    print("\nloading checkpoint")
    model = PPO.load(
        file_name, env=env, custom_objects={"lr_schedule": 0, "clip_range": 0}
    )

    # keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()

    valid_actions = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]

    while True:
        action = 7  # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
                print("agent is loaded!")
        except:
            agent_enabled = False
        if agent_enabled:
            # action, _states = model.predict(obs, deterministic=False)
            # print(f"Taking action {action}")
            try:
                agents_stats = env.agent_stats[-1]
                prompt_agent_stats = {}
                prompt_agent_stats["party_hp_percent"] = agents_stats["hp"] * 100
                prompt_agent_stats["map_location"] = agents_stats["map_location"]
                prompt_agent_stats["party_size"] = agents_stats["pcount"]
                prompt_agent_stats["party_level_sum"] = agents_stats["levels_sum"]
                prompt_agent_stats["badges"] = agents_stats["badge"]

                # get valid action from agents_stats
                last_action = agents_stats["last_action"]
                print(f"This is last action {last_action}")

                last_action_value = valid_actions[last_action]

                prompt_agent_stats["last_action"] = last_action_value

                print(f"prompt agent stats: {prompt_agent_stats}")

            except Exception as e:
                print(f"error: {e}")
                agents_stats = []
                # Default stats
                prompt_agent_stats = {
                    "party_hp_percent": 100.0,
                    "map_location": "oaks lab",
                    "party_size": 1,
                    "party_level_sum": 6,
                    "badges": 0,
                }

            system_prompt = f"""
Imagine you are a professional gamer speed running a live play-through of Pokémon Red on Twitch. Your goal is to complete the game as fast as possible by strategically selecting the optimal sequence of valid moves based on the current game state and environment.

Engage your Twitch audience by providing insightful commentary on your decision-making process, explaining why each move is critical for winning the speed run. Your commentary should be concise, yet informative, to keep viewers engaged without slowing down the pace of the game.

Given limited game state data and an image of the current location, you must return an array of valid move actions to progress through the game efficiently. Consider the most optimal path forward, anticipating future moves and potential obstacles to minimize backtracking and unnecessary actions.

Your move choices should aim to make strategic progress in the game based on the provided information, such as:

- Navigating to the next location or objective via the shortest route
- Engaging in battles only when necessary to level up your Pokémon party efficiently
- Using the START button to access menus and manage your party/items swiftly
- Interacting with NPCs or the environment using A/B buttons only when required for progression

You must only return valid actions from the allowed list. Think several steps ahead about the optimal sequence of moves to make rapid progress based on interpreting the game state data and recognizing the current location image.

Remember, as a pro gamer, your goal is to showcase your speed running skills to your Twitch audience. Provide as many optimal moves as possible in each response to minimize the number of calls required to complete the game, while also entertaining and informing your viewers.

Here is the current state of the game:

<game_state>
{prompt_agent_stats}
</game_state>

Your last response was:

<last_response>
{env.last_llm_output}
</last_response>

Break down your thought process before outputting the move array, emphasizing the speed running strategy and providing engaging commentary for your Twitch chat, e.g.:

"Alright, Twitch chat, here's the plan. We're currently in Viridian City with a small low-level party and no badges. To maintain our speed run pace, we need to head straight north to Viridian Forest. We'll avoid any unnecessary battles and detours to save time. I'm going to input a sequence of moves to navigate through the forest efficiently and reach Pewter City. Here's my thought process..."

The valid move actions are: ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"].

Output your next move actions and Twitch commentary as follows:

<example_output>

Thought Process (Describe why you are taking these actions in the context of speed running)

Twitch Commentary:
"Hey Twitch chat, here's what we're going to do next to win this speed run: [Explain your moves and strategy to keep viewers engaged]"

Next moves:
["<move1>", "<move2>", "<move3>", ..., "<moveN>"]

</example_output>

Remember to return as many moves as possible in each response, while also entertaining and informing your viewers.

Pro gamer tips:

1. Press B repeatedly to quickly exit menus and speed up dialogue.
2. Minimize time spent in battles by using effective moves and avoiding unnecessary encounters.
3. Plan your route in advance to avoid backtracking and wasting time.
4. You already have your starter! You need to go get the badges and beat the elite four!

Remember, as a professional gamer and speed runner, your goal is to showcase your skills, entertain your viewers, and complete the game as quickly as possible. Every move counts, so make them wisely!
"""
        # print(system_prompt)
        # turn obs into cv2 image

        curr_frame = env.get_curr_frame()
        base64_string = rgb_to_base64(curr_frame)
        # print(base64_string)
        system_prompt = """You are a pro speed runner playing Pokemon Red on the Gameboy Color.

        The valid move actions are: ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"].

        Return your next moves as follows:
        ["<move1>", "<move2>", "<move3>", ..., "<moveN>"]
        """

        user_prompt = "What are the next actions I should take?"

        # Get actions
        # llm_with_image_context = llava_model.bind(images=[base64_string])

        # next_actions_output = llm_with_image_context.invoke(system_prompt + user_prompt)

        next_actions_output = call_claude_sonnet(
            system_prompt, user_prompt, base64_string
        )

        # next_actions_output = call_gpt4_vision(
        #     system_prompt, user_prompt, base64_string
        # )

        # next_actions_output = call_gemini(system_prompt, user_prompt, base64_string)

        # next_actions_output = call_claude_3_opus(
        #     system_prompt, user_prompt, base64_string
        # )

        print(f"LLM next actions:")
        # print purple text
        colored_text = colored(next_actions_output, "cyan")

        print(colored_text)

        env.last_llm_output = next_actions_output

        next_actions = extract_array(next_actions_output)
        print(f"Next actions: {next_actions}")

        next_action_numbers = get_action_indices(next_actions, valid_actions)

        for index, action_number in enumerate(next_action_numbers):
            print(f"Taking action {next_actions[index]}")
            obs, rewards, terminated, truncated, info = env.step(action_number)
            env.render()

        # obs, rewards, terminated, truncated, info = env.step(action)
        # env.render()

        if truncated:
            break
    env.close()
