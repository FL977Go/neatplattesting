# Reinforcement Learning with NEAT

A platformer game AI trained using **NEAT (NeuroEvolution of Augmating Topologies)** algorithm. Neural networks evolve over generations to learn how to navigate obstacles and reach the goal.

## 🎮 What This Does

This project uses neuroevolution to train AI agents that:
- Navigate a 2D platformer environment
- Avoid moving enemies
- Reach the goal position
- Learn through evolutionary selection over generations

## 🚀 Quick Start

### 1. Create Virtual Environment

**PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**CMD:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run Training

**With visualization:**
```powershell
python Neatprogram.py
```

**Headless mode (faster):**
```powershell
python Neatprogram.py --headless --gens 100
```

**Command-line options:**
- `--headless` - Run without Pygame display (faster training)
- `--gens N` - Limit to N generations (default: 5000)

## 🧠 How It Works

### Neural Network Architecture

**Inputs (9 sensors):**
- Enemy X/Y position
- Player X/Y position
- Distance to enemy
- Distance to goal
- Y velocity
- On-ground status
- Enemy velocity

**Outputs (3 actions):**
- Action 0: Move left (A)
- Action 1: Move right (D)
- Action 2: Jump (W)

### Training Process

1. **Population**: 300 genomes evolve each generation
2. **Fitness Goal**: Reach 400+ fitness points
3. **Evolution**: Networks mutate and crossover based on performance
4. **Selection**: Best performers pass genes to next generation

### Fitness Rewards

- **Distance reduction**: Rewarded for getting closer to goal
- **Survival**: Small reward for each step survived
- **Goal proximity**: Exponential reward based on final distance to goal
- **Winning**: 600+ fitness for reaching the goal

## 📁 Project Structure

```
reinforcement_learning_neat/
├── Neatprogram.py          # Main training script
├── neatplat.py             # Platformer game environment
├── configneatNew.txt       # NEAT configuration
├── requirements.txt        # Python dependencies
├── test_winner.py          # Test trained genomes
├── play_manual.py          # Manually play the game
├── visuilise2.py           # Visualization utilities
├── showvis.py              # Network visualization
├── stats.py                # Statistics tracking
├── NeatTextOutput*.txt     # Training logs (generated)
└── neat-checkpoint-*       # Saved checkpoints (generated)
```

## 🎯 Configuration

Edit `configneatNew.txt` to adjust:
- **Population size**: `pop_size = 300`
- **Fitness threshold**: `fitness_threshold = 400`
- **Mutation rates**: Various `*_mutate_rate` parameters
- **Network structure**: `num_hidden`, `num_inputs`, `num_outputs`

## 📊 Monitoring Progress

During training you'll see:
- **Pygame window**: Current genome playing in real-time
- **Console output**: Generation statistics
- **Log files**: `NeatTextOutput*.txt` with detailed progress
- **Checkpoints**: Saved every generation as `neat-checkpoint-*`

## 🧪 Testing Trained Models

After training completes, statistics plots are automatically generated. To test a saved checkpoint:

```powershell
python test_winner.py
```

## 🎮 Manual Play

Want to try the game yourself?

```powershell
python play_manual.py
```

Controls:
- **A** - Move left
- **D** - Move right  
- **W** - Jump

## 📈 Typical Training Results

- **Early generations (0-50)**: Random behavior, low fitness
- **Mid training (50-200)**: Basic movement patterns emerge
- **Late training (200+)**: Sophisticated strategies, goal-reaching behavior
- **Success**: Genomes consistently reach 400+ fitness

## 🔧 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'distutils'`
- **Solution**: Make sure you're using the updated `requirements.txt` with pygame 2.6.1+

**Issue**: Training is too slow
- **Solution**: Use `--headless` flag to disable visualization

**Issue**: Genomes not improving
- **Solution**: Adjust mutation rates or fitness rewards in configuration

## 📚 Learn More

- [NEAT Algorithm](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) - Original paper
- [NEAT-Python Documentation](https://neat-python.readthedocs.io/)
- [Pygame Documentation](https://www.pygame.org/docs/)

## 📝 License

This project is open source and available for educational purposes.
