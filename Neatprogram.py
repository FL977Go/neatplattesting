import os
import argparse

# CLI: allow headless mode and limit generations for quick tests
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--headless', action='store_true', help='Run without Pygame display')
parser.add_argument('--gens', type=int, default=5000, help='Number of generations to run')
parser.add_argument('--difficulty', type=str, 
                    choices=['veryeasy', 'easy', 'medium', 'hard', 'veryhard'],
                    default='veryhard',
                    help='Difficulty level: veryeasy, easy, medium, hard, veryhard (default)')
args, _ = parser.parse_known_args()
if args.headless:
    os.environ['HEADLESS'] = '1'

import neat
# Import the appropriate game module based on difficulty
difficulty_map = {
    'veryeasy': 'neatplat_veryeasy',
    'easy': 'neatplat_easy',
    'medium': 'neatplat_medium',
    'hard': 'neatplat_hard',
    'veryhard': 'neatplat'
}
module_name = difficulty_map[args.difficulty]
print(f"\n🎮 DIFFICULTY: {args.difficulty.upper()} (Module: {module_name})\n")
neatplat = __import__(module_name)
from random import randint
from pygame import display, Rect,font
import math
import visuilise2
import showvis
import stats as sp

# Define pump; use dummy for headless mode
if os.environ.get('HEADLESS') == '1':
    def pump():
        pass  # no-op for headless
else:
    from pygame.event import get as pump
# from sys import file
# import traceback
try:
    # initialize display only when not running headless
    headless = os.environ.get('HEADLESS') == '1'
    if not headless:
        display.init()
        font.init()
        display.set_caption('NeroEATS2 NEAT Visualization')
        screen = display.set_mode((1920*0.8,1080*0.8)) 
        textfont = font.SysFont(None, 48)  # None = default font, 48 = size
        SCwidth = display.get_window_size()[0]
        SChight = display.get_window_size()[1]
    else:
        # fallback values used for normalization when headless
        textfont = None
        SCwidth = 1920*0.8
        SChight = 1080*0.8



    
        
    local_dir = os.path.dirname(__file__)
    # Use appropriate config file based on difficulty
    if args.difficulty == 'veryhard':
        config_file = 'configneatNew.txt'
    else:
        config_file = f'configneat_{args.difficulty}.txt'
        
    config_path = os.path.join(local_dir, config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    
    filename =  'NeatTextOutput.txt'
    osi = 0
    while True:
        if os.path.exists(filename):
            osi += 1
            filename = f'NeatTextOutput{osi}.txt'
        else:
            break
    cgen: int = -1
    filex = open(filename, "a", encoding="utf-8")
    class MyReporter(neat.reporting.BaseReporter):
        def post_evaluate(self, config, population, species, best_genome):
            # The Population object has 'generation' attribute
            # generation = population.generation 
            

            # Generate the NEAT-style stats string
            ws = sp.generation_stats_str(stats, p, generation=cgen)

            # Write to file
            with open(filename, "a", encoding="utf-8") as f:
                # print(ws)       # prints to console
                f.write(ws + "\n")
    def blit(X,Y,enemyX,enemyY,genome_id:str,disition:str, fitnessdr:int = 'n/a'):
        # no-op when headless
        global steps, tempfitness, goaldis,death, emenmydis
        if os.environ.get('HEADLESS') == '1':
            return

        display.flip()

        # Render text
        text = textfont.render(genome_id, True, (0, 0, 0))  # text, antialias, color
        text_decision = textfont.render(disition, True, (0, 0, 0))  # text, antialias, color
        hud = False
        if hud:
            try:
                hud_lines = [
                '🎮 VERY EASY MODE',
                f'Step: {steps}',
                f'Fitness (last step): {tempfitness:.2f}',
                f'Total fitness: {fitnessdr:.2f}',
                f'Enemy distance: ({emenmydis:.2f}) [STATIONARY - FIXED AT 25%]',
                f'Player: ({X:.1f}, {Y:.1f})',
                f'Goal distance: {goaldis:.1f}',
                f'Death: {death}',
                f'Delta: {delta*120}',# Debug: show death status
                ]
            except Exception:
                print('error: error making hud')
        screen.fill((200, 200, 200))
        if hud:
            try:
                for i, line in enumerate(hud_lines):
                    surf = textfont.render(line,True, (0, 0, 0))
                    screen.blit(surf, (10, 8 + i * 35))
            except Exception:
                print('error: error bliting the hud')
        screen.blit(neatplat.player.convert_alpha(), (round(X), round(Y)))
        screen.blit(neatplat.floor1.convert_alpha(),(round(SCwidth * 0.01), round(SChight) * 0.82))
        screen.blit(neatplat.enemy1.convert_alpha(), (enemyX, enemyY))
        screen.blit(neatplat.win.convert_alpha(), (round(SCwidth * 0.74), (round(SChight) * 0.82 - neatplat.win.get_height()))) 
        screen.blit(text.convert_alpha(), ((1920*0.8)*0.4, 50))  # Blit text at position (50, 50)
        screen.blit(text_decision.convert_alpha(), ((1920*0.8)*0.4, 100))  # Blit text at position (50, 100)
    def eval_genomes(genomes, config):
        global stats, cgen, p, steps,tempfitness,goaldis,death,delta, emenmydis
        # stats
        # print(' '.join(stats.generation_statistics))
        # for names in dir(stats):
        #     print(f'stat val: {names}')
        # for namep in dir(p):
        #     print(f'pop val: {namep}')
        cgen += 1
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f'\n**** starting gen {cgen} ****\n')
        for genome_id, genome in genomes:
            neatplat.reset()
            genome.fitness = 0
            # Create a recurrent network if the genome may contain recurrent connections
            try:
                net = neat.nn.RecurrentNetwork.create(genome, config)
            except Exception:
                # Fallback to feed-forward network if recurrent creation fails
                net = neat.nn.FeedForwardNetwork.create(genome, config)
            # test_in = [1.0]*6
            # test_out = net.activate(test_in)
            # print('SANITY test_in ->', test_in, 'out ->', test_out, 'type:', type(test_out), 'len:', len(test_out))
            # initialize environment and read initial state
            neatplat.reset()
            vis = showvis.showvis()
            
            genome.fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(False, False, False)

            visuilise2.draw_net(config, genome, view=False, fmt='png', show_disabled=False,prune_unused=False,node_names={-1:'EX',-2:'EY',-3:'PX',-4:'PY',-5:'ED',-6:'GD',-7:'Y_VE',-8:'OGR',-9:'EDX',0:'A',1:'D',2:'W'})
            vis.go = False
            vis.draw()
            for episode in range(2):
                # start fresh for each short episode
                neatplat.reset()
                genome.fitness += 0  # ensure float
                genome.fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(False, False, False)
                # initialize shaping tracker
                last_goaldis = goaldis
                steps = 0
                # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
                # 
                # visuilise.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
                

                # safety cap for very long episodes
                # max_steps = 3500
                while not death:
                    # if steps >= max_steps:
                    #     break
                    # normalize inputs: positions -> 0..1 roughly, distances scaled
                    inputs = [
                        enemyX / neatplat.SCwidth,
                        enemyY / neatplat.SChight,
                        X / neatplat.SCwidth,
                        Y / neatplat.SChight,
                        emenmydis / (neatplat.SCwidth * 1.5 + 1e-6),
                        goaldis / (neatplat.SCwidth * 1.5 + 1e-6),

                        # NEW — temporal / state info
                        Y_vel / (neatplat.SChight + 1e-6),
                        on_ground,
                        enemy_dx / (neatplat.SCwidth + 1e-6),
                    ]
                    rawoutput = net.activate(inputs)
                    # optional debug printing: set environment variable NEAT_DEBUG=1 to enable
                    
                    if os.environ.get('NEAT_DEBUG') == '1':
                        print(f'GENOME {genome_id} inputs={inputs} raw={rawoutput}')
                        print(f'GENOME:{genome}')
                        print(net.values)
                    # detect NaN or all-zero outputs
                    if any((isinstance(x, float) and math.isnan(x)) for x in rawoutput):
                        print(f'Warning: rawoutput contains NaN for genome {genome_id} at step {steps}:', rawoutput)
                    if all((abs(x) < 1e-12) for x in rawoutput):
                        # print once to help debugging
                        if os.environ.get('NEAT_DEBUG') == '1':
                            print(f'Notice: rawoutput all zeros for genome {genome_id} at step {steps}')
                    # robust argmax to avoid ties
                    decision = int(max(range(len(rawoutput)), key=lambda i: rawoutput[i]))
                    # map decision (0/1/2) to three boolean action inputs
                    action1 = (decision == 0)
                    action2 = (decision == 1)
                    action3 = (decision == 2)
                    # call simulation with the three boolean inputs and update state
                    tempfitness,enemyX,enemyY,X,Y,emenmydis,goaldis,Y_vel,on_ground,enemy_dx,death = neatplat.sim(action1, action2, action3, steps/130)
                    # per-step shaping: reward reduction in distance-to-goal
                    if emenmydis < 150:
                        if not (enemyY - Y) > 30:
                            # STRONGER penalty for being close to enemy at same height (farming behavior)
                            tempfitness -= (150 - emenmydis) * 0.6
                        else:
                            # Lighter penalty when airborne above enemy
                            tempfitness -= (150 - emenmydis) * 0.05
                    # BONUS: Reward for being airborne (encourages jumping)
                    if on_ground < 0.5:
                        tempfitness += 0.3
                    try:
                        delta = last_goaldis - goaldis
                        # print(delta)
                    except Exception:
                        print('error: delta computational error')
                        delta = 0.0
                    if delta <= .25  and delta >= -.25:
                        tempfitness -= 3.5
                    elif delta > 0:
                        # print(delta*120)
                        tempfitness += delta * 120
                    if delta <= 3  and delta >= -3:
                        tempfitness -= 5
                    if delta <= -.5:
                        tempfitness -= .5
                    if (X - enemyX) > 5:
                        tempfitness += (X - enemyX)/12
                    last_goaldis = goaldis
                    if tempfitness >= 99999999999999999999:
                        with open(filename, "a", encoding="utf-8") as f:
                            f.write(f'\n**** Gemome number: {genome_id} of generation: {cgen}, has reached won in {steps} steps ****\n')
                            print(f'**** Gemome number: {genome_id} of generation: {cgen}, has reached won in {steps} steps ****')
                    
                    
                    if os.environ.get('HEADLESS') != '1' and display.get_active():
                        if steps % 4 == 0:
                            blit(X,Y,enemyX,enemyY,str(genome_id),str(cgen),genome.fitness)                                      
                        pump()
                    else:
                        pump()
                    genome.fitness += tempfitness
                    # modest baseline survival reward (neatplat.sim already adds 0.01)
                    genome.fitness -= 0.2
                    steps += 1
                # end episode; add final shaped reward based on goal distance
                # guard against extremely small goal distances causing huge rewards
                safe_goal = max(goaldis, 1.0)
                try:
                    shaped = (((((768000.0 / (safe_goal ** 1.903)) ** 0.7) * 2.8571428) ** 1.3) / 2.0)
                except Exception:
                    shaped = 0.0
                genome.fitness += shaped
        # ws = sp.generation_stats_str(stats, p, generation=cgen)  
    def test_winner(genome, config):
        try:
            net = neat.nn.RecurrentNetwork.create(genome, config)
        except Exception:
                # Fallback to feed-forward network if recurrent creation fails
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        for episode in range(2):
                # start fresh for each short episode
                neatplat.reset()
                genome.fitness += 0  # ensure float
                genome.fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(False, False, False)
                # initialize shaping tracker
                last_goaldis = goaldis
                steps = 0
                # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
                # 
                # visuilise.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
                

                # safety cap for very long episodes
                # max_steps = 3500
                while not death:
                    # if steps >= max_steps:
                    #     break
                    # normalize inputs: positions -> 0..1 roughly, distances scaled
                    inputs = [
                        enemyX / neatplat.SCwidth,
                        enemyY / neatplat.SChight,
                        X / neatplat.SCwidth,
                        Y / neatplat.SChight,
                        emenmydis / (neatplat.SCwidth * 1.5 + 1e-6),
                        goaldis / (neatplat.SCwidth * 1.5 + 1e-6),

                        # NEW — temporal / state info
                        Y_vel / (neatplat.SChight + 1e-6),
                        on_ground,
                        enemy_dx / (neatplat.SCwidth + 1e-6),
                    ]
                    rawoutput = net.activate(inputs)
                    # optional debug printing: set environment variable NEAT_DEBUG=1 to enable
                    
                    # if os.environ.get('NEAT_DEBUG') == '1':
                    #     print(f'GENOME {genome_id} inputs={inputs} raw={rawoutput}')
                    #     print(f'GENOME:{genome}')
                    #     print(net.values)
                    # # detect NaN or all-zero outputs
                    # if any((isinstance(x, float) and math.isnan(x)) for x in rawoutput):
                    #     print(f'Warning: rawoutput contains NaN for genome {genome_id} at step {steps}:', rawoutput)
                    # if all((abs(x) < 1e-12) for x in rawoutput):
                    #     # print once to help debugging
                    #     if os.environ.get('NEAT_DEBUG') == '1':
                    #         print(f'Notice: rawoutput all zeros for genome {genome_id} at step {steps}')
                    # robust argmax to avoid ties
                    decision = int(max(range(len(rawoutput)), key=lambda i: rawoutput[i]))
                    # map decision (0/1/2) to three boolean action inputs
                    action1 = (decision == 0)
                    action2 = (decision == 1)
                    action3 = (decision == 2)
                    # call simulation with the three boolean inputs and update state
                    tempfitness,enemyX,enemyY,X,Y,emenmydis,goaldis,Y_vel,on_ground,enemy_dx,death = neatplat.sim(action1, action2, action3)
                    # per-step shaping: reward reduction in distance-to-goal
                    try:
                        delta = last_goaldis - goaldis
                    except Exception:
                        delta = 0.0
                    if delta > 0:
                        genome.fitness += delta * 0.3
                    last_goaldis = goaldis
                    # if tempfitness >= 600:
                    #     with open(filename, "a", encoding="utf-8") as f:
                    #         f.write(f'\n**** Gemome number: {genome_id} of generation: {cgen}, has reached won in {steps} steps ****\n')
                    #         print(f'**** Gemome number: {genome_id} of generation: {cgen}, has reached won in {steps} steps ****')
                    
                    
                    if os.environ.get('HEADLESS') != '1' and display.get_active():
                        
                        blit(X,Y,enemyX,enemyY,str('Winner'),str(cgen))                                      
                        pump()
                    else:
                        pump()
                    genome.fitness += tempfitness
                    # modest baseline survival reward (neatplat.sim already adds 0.01)
                    genome.fitness += 0.02
                    steps += 0.75
                # end episode; add final shaped reward based on goal distance
                # guard against extremely small goal distances causing huge rewards
                safe_goal = max(goaldis, 1.0)
                try:
                    shaped = (((((768000.0 / (safe_goal ** 1.903)) ** 0.7) * 2.8571428) ** 1.3) / 2.0)
                except Exception:
                    shaped = 0.0
                genome.fitness += shaped
        print(f'Winner fitness: {genome.fitness}')
    
    def save_graph():
        plot_filename = f'training_graph_{args.difficulty}.png'
        print(f"\nSaving training graph to {plot_filename}...")
        visuilise2.plot_stats(stats, ylog=False, view=False, filename=plot_filename, title=f'Training Progress - {args.difficulty.upper()}')

    def run_neat(config):
        global stats,p 
        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-80')
        
        p.add_reporter(neat.StdOutReporter(True))
        
        p.add_reporter(stats)
        p.add_reporter(MyReporter())
        # p.add_reporter(neat.Checkpointer(1))
        p.add_reporter(neat.Checkpointer(1))


        # CHEACKPOUNT 22



        # print(str(stats)+'\n\n')
        # allow overriding generation count via CLI
        winner = p.run(eval_genomes, args.gens)
        print(f'##########\n# winner #\n##########')
        test_winner(winner,config)
        
        # Save graph at the end of successful run
        save_graph()
        
        # visuilise2.plot_stats(stats, ylog=False, view=True)
        # visuilise2.plot_species(stats, view=True)
        # net = neat.nn.FeedForwardNetwork.create(winner, config)
        # for _ in range(10):
        #     x = randint(0, 1)
        #     output = net.activate([x])[0]
        #     target = 1 - x  # expects opposite
        #     score = 12.5 * (1 - abs(target - output))-2.5
        #     print(f'output:{output}\ntarget:{target}\nscore:{score}')
            



    if __name__ == '__main__':
        

        try:
            run_neat(config)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving graph...")
            save_graph()
            
except Exception:
    print(Exception.with_traceback())
    pass


# #anoher version of the code


# # run_neat_seeded.py
# import os
# import argparse
# import math

# import neat

# # ----------------------------
# # CLI options
# # ----------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument('--headless', action='store_true', help='Run without Pygame display')
# parser.add_argument('--gens', type=int, default=5000, help='Number of generations to run')
# parser.add_argument('--difficulty', type=str,
#                     choices=['veryeasy', 'easy', 'medium', 'hard', 'veryhard'],
#                     default='veryhard',
#                     help='Difficulty level: veryeasy, easy, medium, hard, veryhard (default)')
# parser.add_argument('--episodes', type=int, default=2, help='Episodes per genome (default: 2)')
# parser.add_argument('--seeded', action='store_true',
#                     help='Use deterministic seeds for reset() so Best vs Winner is comparable')
# parser.add_argument('--winner_trials', type=int, default=10,
#                     help='How many random/seeded trials to test winner at the end (default: 10)')
# parser.add_argument('--debug', action='store_true', help='Enable extra debug logging')
# args = parser.parse_args()

# if args.headless:
#     os.environ['HEADLESS'] = '1'
# if args.debug:
#     os.environ['NEAT_DEBUG'] = '1'

# # ----------------------------
# # Import the appropriate game module based on difficulty
# # ----------------------------
# difficulty_map = {
#     'veryeasy': 'neatplat_veryeasy',
#     'easy': 'neatplat_easy',
#     'medium': 'neatplat_medium',
#     'hard': 'neatplat_hard',
#     'veryhard': 'neatplat'
# }
# module_name = difficulty_map[args.difficulty]
# print(f"\n🎮 DIFFICULTY: {args.difficulty.upper()} (Module: {module_name})\n")

# neatplat = __import__(module_name)

# # ----------------------------
# # Pygame display (only if not headless)
# # ----------------------------
# headless = os.environ.get('HEADLESS') == '1'
# if not headless:
#     from pygame import display, font
#     if not display.get_init():
#         display.init()
#     if not font.get_init():
#         font.init()
#     display.set_caption('NeroEATS2 NEAT Visualization (Seeded)')
#     screen = display.set_mode((int(1920 * 0.8), int(1080 * 0.8)))
#     textfont = font.SysFont(None, 48)
#     SCwidth, SChight = display.get_window_size()
#     from pygame.event import get as pump
# else:
#     textfont = None
#     SCwidth = int(1920 * 0.8)
#     SChight = int(1080 * 0.8)

#     def pump():
#         pass  # no-op in headless

# # Optional imports you already use
# import visuilise2
# import showvis
# import stats as sp


# # ----------------------------
# # Helpers
# # ----------------------------
# def safe_reset(seed=None):
#     """
#     Calls neatplat.reset(seed=...) if supported, otherwise falls back to neatplat.reset().
#     """
#     try:
#         # If your reset() supports seed, this works:
#         return neatplat.reset(seed=seed)
#     except TypeError:
#         # Your reset() doesn't accept seed yet
#         return neatplat.reset()


# def blit(X, Y, enemyX, enemyY, genome_id: str, decision_text: str):
#     """
#     Render one frame (only when not headless).
#     """
#     if headless:
#         return

#     display.flip()
#     text = textfont.render(str(genome_id), True, (0, 0, 0))
#     text_decision = textfont.render(str(decision_text), True, (0, 0, 0))

#     screen.fill((200, 200, 200))
#     screen.blit(neatplat.player.convert_alpha(), (round(X), round(Y)))
#     screen.blit(neatplat.floor1.convert_alpha(), (round(SCwidth * 0.01), round(SChight) * 0.82))
#     screen.blit(neatplat.enemy1.convert_alpha(), (enemyX, enemyY))
#     screen.blit(neatplat.win.convert_alpha(),
#                 (round(SCwidth * 0.74), (round(SChight) * 0.82 - neatplat.win.get_height())))
#     screen.blit(text.convert_alpha(), (int((1920 * 0.8) * 0.01), 50))
#     screen.blit(text_decision.convert_alpha(), (int((1920 * 0.8) * 0.1), 100))


# def build_net(genome, config):
#     """
#     Create a network for the genome. Prefer recurrent; fall back to feed-forward.
#     """
#     try:
#         return neat.nn.RecurrentNetwork.create(genome, config)
#     except Exception:
#         return neat.nn.FeedForwardNetwork.create(genome, config)


# def run_one_episode(net, genome_id, config, episode_index, render_label=None):
#     """
#     Runs a single episode and returns the total fitness for that episode.
#     This uses the SAME logic you already wrote, just packaged neatly.
#     """
#     # Deterministic seed option: makes evaluation repeatable
#     seed = None
#     if args.seeded:
#         seed = (int(genome_id) * 1000 + int(episode_index))

#     safe_reset(seed=seed)

#     # Prime the environment (your sim returns initial state)
#     episode_fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(
#         False, False, False
#     )

#     last_goaldis = goaldis
#     steps = 0.0
#     max_steps = 3500

#     while not death and steps < max_steps:
#         # Normalize inputs
#         inputs = [
#             enemyX / neatplat.SCwidth,
#             enemyY / neatplat.SChight,
#             X / neatplat.SCwidth,
#             Y / neatplat.SChight,
#             emenmydis / (neatplat.SCwidth * 1.5 + 1e-6),
#             goaldis / (neatplat.SCwidth * 1.5 + 1e-6),
#             Y_vel / (neatplat.SChight + 1e-6),
#             on_ground,
#             enemy_dx / (neatplat.SCwidth + 1e-6),
#         ]

#         rawoutput = net.activate(inputs)

#         if os.environ.get('NEAT_DEBUG') == '1':
#             print(f'GENOME {genome_id} inputs={inputs} raw={rawoutput}')

#         # argmax decision
#         decision = int(max(range(len(rawoutput)), key=lambda i: rawoutput[i]))
#         actionA = (decision == 0)  # A
#         actionD = (decision == 1)  # D
#         actionW = (decision == 2)  # W

#         tempfitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(
#             actionA, actionD, actionW
#         )

#         # reward moving closer to goal (your existing shaping)
#         delta = last_goaldis - goaldis
#         if delta > 0:
#             episode_fitness += delta * 0.3
#         last_goaldis = goaldis

#         # Add step fitness + small survival bonus
#         episode_fitness += tempfitness
#         episode_fitness += 0.02

#         # Render occasionally
#         if (not headless) and display.get_active():
#             if int(steps) % 10 == 0:
#                 label = render_label if render_label is not None else ""
#                 blit(X, Y, enemyX, enemyY, str(genome_id), label)
#             pump()
#         else:
#             pump()

#         steps += 0.75

#     # End-of-episode shaped reward based on goal distance (your existing formula)
#     safe_goal = max(goaldis, 1.0)
#     try:
#         shaped = (((((768000.0 / (safe_goal ** 1.903)) ** 0.7) * 2.8571428) ** 1.3) / 2.0)
#     except Exception:
#         shaped = 0.0

#     episode_fitness += shaped
#     return float(episode_fitness)


# # ----------------------------
# # Main NEAT wiring
# # ----------------------------
# local_dir = os.path.dirname(__file__)

# if args.difficulty == 'veryhard':
#     config_file = 'configneatNew.txt'
# else:
#     config_file = f'configneat_{args.difficulty}.txt'

# config_path = os.path.join(local_dir, config_file)
# config = neat.Config(
#     neat.DefaultGenome,
#     neat.DefaultReproduction,
#     neat.DefaultSpeciesSet,
#     neat.DefaultStagnation,
#     config_path
# )

# p = neat.Population(config)
# stats = neat.StatisticsReporter()

# # Output file naming like your original
# filename = 'NeatTextOutput.txt'
# osi = 0
# while os.path.exists(filename):
#     osi += 1
#     filename = f'NeatTextOutput{osi}.txt'

# cgen = -1


# class MyReporter(neat.reporting.BaseReporter):
#     def post_evaluate(self, config_, population, species, best_genome):
#         # Uses global stats + p + cgen for formatted output
#         ws = sp.generation_stats_str(stats, p, generation=cgen)
#         with open(filename, "a", encoding="utf-8") as f:
#             f.write(ws + "\n")


# def eval_genomes(genomes, config_):
#     """
#     Evaluates all genomes for a generation.
#     Key changes:
#       - Deterministic seeds (optional) -> makes Best vs Winner comparable.
#       - No duplicate reset calls.
#       - Per-genome fitness is sum across episodes.
#     """
#     global cgen
#     cgen += 1
#     with open(filename, "a", encoding="utf-8") as f:
#         f.write(f'\n****** Running generation {cgen} ******\n')

#     for genome_id, genome in genomes:
#         genome.fitness = 0.0
#         net = build_net(genome, config_)

#         # OPTIONAL: draw net once per genome (can be slow; disable in headless)
#         if not headless:
#             try:
#                 vis = showvis.showvis()
#                 vis.go = False
#                 vis.draw()
#                 visuilise2.draw_net(
#                     config_,
#                     genome,
#                     view=False,
#                     fmt='png',
#                     show_disabled=False,
#                     prune_unused=False,
#                     node_names={
#                         -1: 'EX', -2: 'EY', -3: 'PX', -4: 'PY', -5: 'ED', -6: 'GD',
#                         -7: 'Y_VE', -8: 'OGR', -9: 'EDX',
#                         0: 'A', 1: 'D', 2: 'W'
#                     }
#                 )
#             except Exception:
#                 pass

#         total = 0.0
#         for episode in range(args.episodes):
#             ep_fit = run_one_episode(net, genome_id, config_, episode_index=episode, render_label=str(cgen))
#             total += ep_fit

#         genome.fitness = float(total)


# def test_winner(winner_genome, config_):
#     """
#     Test the final winner. Important changes:
#       - winner_genome.fitness resets to 0.0
#       - runs multiple trials so you see consistency
#       - can be seeded deterministically (if --seeded) or more variable otherwise
#     """
#     net = build_net(winner_genome, config_)
#     winner_genome.fitness = 0.0

#     scores = []
#     for t in range(args.winner_trials):
#         # For winner trials, use stable seeds if --seeded, else let environment randomize.
#         # If seeded, we use a fixed base so runs are repeatable.
#         if args.seeded:
#             # Use a fixed base seed that doesn't depend on genome_id (we don't have it here).
#             # Each trial gets a different, but repeatable seed.
#             safe_reset(seed=(999999 + t))
#         else:
#             safe_reset(seed=None)

#         # run_one_episode calls reset itself, so we just call it with unique episode_index=t
#         # to vary seed in a repeatable way when --seeded is on.
#         score = run_one_episode(net, genome_id=999999, config=config_, episode_index=t, render_label="Winner")
#         scores.append(score)

#     winner_genome.fitness = float(sum(scores) / max(1, len(scores)))
#     print("\n##########\n# winner #\n##########")
#     print(f"Winner avg fitness over {len(scores)} trials: {winner_genome.fitness:.6f}")
#     print(f"Winner trial min/max: {min(scores):.6f} / {max(scores):.6f}")


# def save_graph():
#     plot_filename = f'training_graph_{args.difficulty}.png'
#     print(f"\nSaving training graph to {plot_filename}...")
#     visuilise2.plot_stats(
#         stats,
#         ylog=False,
#         view=False,
#         filename=plot_filename,
#         title=f'Training Progress - {args.difficulty.upper()}'
#     )


# def run_neat():
#     # Reporters
#     p.add_reporter(neat.StdOutReporter(True))
#     p.add_reporter(stats)
#     p.add_reporter(MyReporter())
#     p.add_reporter(neat.Checkpointer(1))

#     # Train
#     winner = p.run(eval_genomes, args.gens)

#     # Test winner and save graph
#     test_winner(winner, config)
#     save_graph()


# if __name__ == '__main__':
#     try:
#         run_neat()
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user. Saving graph...")
#         save_graph()
