from past.utils import old_div
import random
from utils import malmoutils

# Task parameters:
NUM_GOALS = 20
GOAL_TYPE = "apple"
GOAL_REWARD = 100
ARENA_WIDTH = 60
ARENA_BREADTH = 60
MOB_TYPE = "Endermite"  # Change for fun, but note that spawning conditions have to be correct - eg spiders will require darker conditions.


# Agent parameters:
agent_stepsize = 1
agent_search_resolution = 30  # Smaller values make computation faster, which seems to offset any benefit from the higher resolution.
agent_goal_weight = 100
agent_edge_weight = -100
agent_mob_weight = -10
agent_turn_weight = 0  # Negative values to penalise turning, positive to encourage.


def getItemXML():
    ''' Build an XML string that contains some randomly positioned goal items'''
    xml = ""
    for item in range(NUM_GOALS):
        x = str(
            random.randint(old_div(-ARENA_WIDTH, 2), old_div(ARENA_WIDTH, 2)))
        z = str(random.randint(old_div(-ARENA_BREADTH, 2),
                               old_div(ARENA_BREADTH, 2)))
        xml += '''<DrawItem x="''' + x + '''" y="210" z="''' + z + '''" type="''' + GOAL_TYPE + '''"/>'''
    return xml


def getCorner(index, top, left, expand=0, y=206):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand + old_div(ARENA_WIDTH, 2))) if left else str(
        expand + old_div(ARENA_WIDTH, 2))
    z = str(-(expand + old_div(ARENA_BREADTH, 2))) if top else str(
        expand + old_div(ARENA_BREADTH, 2))
    return 'x' + index + '="' + x + '" y' + index + '="' + str(
        y) + '" z' + index + '="' + z + '"'


def getMissionXML(summary, agent_host):
    ''' Build an XML mission string.'''
    spawn_end_tag = ' type="mob_spawner" variant="' + MOB_TYPE + '"/>'
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>

        <ModSettings>
            <MsPerTick>20</MsPerTick>
        </ModSettings>
        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>13000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <AllowSpawning>true</AllowSpawning>
                <AllowedMobs>''' + MOB_TYPE + '''</AllowedMobs>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid ''' + getCorner("1", True, True,
                                                expand=1) + " " + getCorner("2",
                                                                            False,
                                                                            False,
                                                                            y=226,
                                                                            expand=1) + ''' type="stone"/>
                    <DrawCuboid ''' + getCorner("1", True, True,
                                                y=207) + " " + getCorner("2",
                                                                         False,
                                                                         False,
                                                                         y=226) + ''' type="air"/>

                    <DrawLine ''' + getCorner("1", True,
                                              True) + " " + getCorner("2", True,
                                                                      False) + spawn_end_tag + '''
                    <DrawLine ''' + getCorner("1", True,
                                              True) + " " + getCorner("2",
                                                                      False,
                                                                      True) + spawn_end_tag + '''
                    <DrawLine ''' + getCorner("1", False,
                                              False) + " " + getCorner("2",
                                                                       True,
                                                                       False) + spawn_end_tag + '''
                    <DrawLine ''' + getCorner("1", False,
                                              False) + " " + getCorner("2",
                                                                       False,
                                                                       True) + spawn_end_tag + '''
                    <DrawCuboid x1="-1" y1="206" z1="-1" x2="1" y2="206" z2="1" ''' + spawn_end_tag + '''
                    ''' + getItemXML() + '''
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>The Hunted</Name>
            <AgentStart>
                <Placement x="0.5" y="207.0" z="0.5"/>
                <Inventory>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ChatCommands/>
                <ContinuousMovementCommands turnSpeedDegs="360"/>
                <AbsoluteMovementCommands/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="''' + str(
        ARENA_WIDTH) + '''" yrange="2" zrange="''' + str(ARENA_BREADTH) + '''" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>
                <RewardForCollectingItem>
                    <Item type="''' + GOAL_TYPE + '''" reward="''' + str(
        GOAL_REWARD) + '''"/>
                </RewardForCollectingItem>''' + malmoutils.get_video_xml(
        agent_host) + '''
            </AgentHandlers>
        </AgentSection>

    </Mission>'''
