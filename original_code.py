from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from malmo import MalmoPython
import os
import time
import json
import errno
import math
import tkinter as tk
from env.mob_fun_env import *
import functools
malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)


recordingsDirectory = "FleeRecordings"
try:
    os.makedirs(recordingsDirectory)
except OSError as exception:
    if exception.errno != errno.EEXIST:  # ignore error if already existed
        raise

print = functools.partial(print, flush=True)

root = tk.Tk()
root.wm_title("Collect the " + GOAL_TYPE + "s, dodge the " + MOB_TYPE + "s!")

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                   borderwidth=0, highlightthickness=0, bg="black")
canvas.pack()
root.update()


def findUs(entities):
    for ent in entities:
        if ent["name"] == MOB_TYPE:
            continue
        elif ent["name"] == GOAL_TYPE:
            continue
        else:
            return ent


def getBestAngle(entities, current_yaw, current_health):
    '''Scan through 360 degrees, looking for the best direction in which to take the next step.'''
    us = findUs(entities)
    scores = []
    # Normalise current yaw:
    while current_yaw < 0:
        current_yaw += 360
    while current_yaw > 360:
        current_yaw -= 360

    # Look for best option
    for i in range(agent_search_resolution):
        # Calculate cost of turning:
        ang = 2 * math.pi * (old_div(i, float(agent_search_resolution)))
        yaw = i * 360.0 / float(agent_search_resolution)
        yawdist = min(abs(yaw - current_yaw), 360 - abs(yaw - current_yaw))
        turncost = agent_turn_weight * yawdist
        score = turncost

        # Calculate entity proximity cost for new (x,z):
        x = us["x"] + agent_stepsize - math.sin(ang)
        z = us["z"] + agent_stepsize * math.cos(ang)
        for ent in entities:
            dist = (ent["x"] - x) * (ent["x"] - x) + (ent["z"] - z) * (
                        ent["z"] - z)
            if (dist == 0):
                continue
            weight = 0.0
            if ent["name"] == MOB_TYPE:
                weight = agent_mob_weight
                dist -= 1  # assume mobs are moving towards us
                if dist <= 0:
                    dist = 0.1
            elif ent["name"] == GOAL_TYPE:
                weight = agent_goal_weight * current_health / 20.0
            score += old_div(weight, float(dist))

        # Calculate cost of proximity to edges:
        distRight = (2 + old_div(ARENA_WIDTH, 2)) - x
        distLeft = (-2 - old_div(ARENA_WIDTH, 2)) - x
        distTop = (2 + old_div(ARENA_BREADTH, 2)) - z
        distBottom = (-2 - old_div(ARENA_BREADTH, 2)) - z
        score += old_div(agent_edge_weight,
                         float(distRight * distRight * distRight * distRight))
        score += old_div(agent_edge_weight,
                         float(distLeft * distLeft * distLeft * distLeft))
        score += old_div(agent_edge_weight,
                         float(distTop * distTop * distTop * distTop))
        score += old_div(agent_edge_weight, float(
            distBottom * distBottom * distBottom * distBottom))
        scores.append(score)

    # Find best score:
    i = scores.index(max(scores))
    # Return as an angle in degrees:
    return i * 360.0 / float(agent_search_resolution)


def canvasX(x):
    return (old_div(CANVAS_BORDER, 2)) + (
                0.5 + old_div(x, float(ARENA_WIDTH))) * (
                       CANVAS_WIDTH - CANVAS_BORDER)


def canvasY(y):
    return (old_div(CANVAS_BORDER, 2)) + (
                0.5 + old_div(y, float(ARENA_BREADTH))) * (
                       CANVAS_HEIGHT - CANVAS_BORDER)


def drawMobs(entities, flash):
    canvas.delete("all")
    if flash:
        canvas.create_rectangle(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT,
                                fill="#ff0000")  # Pain.
    canvas.create_rectangle(canvasX(old_div(-ARENA_WIDTH, 2)),
                            canvasY(old_div(-ARENA_BREADTH, 2)),
                            canvasX(old_div(ARENA_WIDTH, 2)),
                            canvasY(old_div(ARENA_BREADTH, 2)), fill="#888888")
    for ent in entities:
        if ent["name"] == MOB_TYPE:
            canvas.create_oval(canvasX(ent["x"]) - 2, canvasY(ent["z"]) - 2,
                               canvasX(ent["x"]) + 2, canvasY(ent["z"]) + 2,
                               fill="#ff2244")
        elif ent["name"] == GOAL_TYPE:
            canvas.create_oval(canvasX(ent["x"]) - 3, canvasY(ent["z"]) - 3,
                               canvasX(ent["x"]) + 3, canvasY(ent["z"]) + 3,
                               fill="#4422ff")
        else:
            canvas.create_oval(canvasX(ent["x"]) - 4, canvasY(ent["z"]) - 4,
                               canvasX(ent["x"]) + 4, canvasY(ent["z"]) + 4,
                               fill="#22ff44")
    root.update()


validate = True
# Create a pool of Minecraft Mod clients.
# By default, mods will choose consecutive mission control ports, starting at 10000,
# so running four mods locally should produce the following pool by default (assuming nothing else
# is using these ports):
my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10002))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10003))

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000

current_yaw = 0
best_yaw = 0
current_life = 0

for iRepeat in range(num_reps):
    max_retries = 3
    # Set up a recording
    my_mission_record = malmoutils.get_default_recording_object(agent_host,
                                                                "Mission_" + str(
                                                                    iRepeat))
    for retry in range(max_retries):
        try:
            # Attempt to start the mission:
            agent_host.startMission(my_mission, my_client_pool,
                                    my_mission_record, 0, "predatorExperiment")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission", e)
                print("Is the game running?")
                exit(1)
            else:
                time.sleep(2)

    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    agent_host.sendCommand("move 1")  # run!
    # main loop:
    total_reward = 0
    total_commands = 0
    flash = False
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)

            if "Yaw" in ob:
                current_yaw = ob[u'Yaw']
            if "Life" in ob:
                life = ob[u'Life']
                if life < current_life:
                    agent_host.sendCommand("chat aaaaaaaaargh!!!")
                    flash = True
                current_life = life
            if "entities" in ob:
                entities = ob["entities"]
                print(entities[0]['life'])
                drawMobs(entities, flash)
                best_yaw = getBestAngle(entities, current_yaw, current_life)
                difference = best_yaw - current_yaw;
                while difference < -180:
                    difference += 360;
                while difference > 180:
                    difference -= 360;
                difference /= 180.0;
                agent_host.sendCommand("turn " + str(difference))
                total_commands += 1
        if world_state.number_of_rewards_since_last_state > 0:
            # A reward signal has come in - see what it is:
            total_reward += world_state.rewards[-1].getValue()
            # print(world_state.rewards[-1].getValue())
        time.sleep(0.02)
        flash = False

    # mission has ended.
    for error in world_state.errors:
        print("Error:", error.text)
    if world_state.number_of_rewards_since_last_state > 0:
        # A reward signal has come in - see what it is:
        total_reward += world_state.rewards[-1].getValue()

    print("We stayed alive for " + str(
        total_commands) + " commands, and scored " + str(total_reward))
    time.sleep(1)  # Give the mod a little time to prepare for the next mission.
