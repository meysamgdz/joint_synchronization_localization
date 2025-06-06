# Joint Synchronization and Localization System

A Python implementation for joint time synchronization and localization ([paper](https://ieeexplore.ieee.org/document/9787497)) using:
- Particle Filtering (PF)
- Bayesian Recursive Filtering (BRF)
- Angle of Arrival (AoA) estimation
- Network synchronization protocols
- Timestamp exchange simulation

## Key Features
- Comparative analysis of PF and BRF methods
- Position and offset error visualization through CDF plots
- Mobile trajectory generation with configurable parameters
- Access Point (AP) positioning system
- Network clock synchronization with Belief Propagation
- NLOS/LOS identification capability
- Hybrid synchronization (BP + Kalman BRF) support

## Key Algorithms
### Localization System
- Linearized BRF with PV model
- Clock parameter estimation
- Particle Filter (Gaussian mixture location updates)

### Network Synchronization
- Belief Propagation (Factor graph implementation & Message passing algorithm)
- Hybrid Synchronization (BP + BRF combination

## Files Overview
### `main.py`
The main simulation script that:
- Configures simulation parameters (iterations, time steps, bins, particles)
- Runs both PF and BRF localization methods
- Generates comparative CDF plots for position and offset errors

### `helper.py`
Provides utility functions for:
- Angle of Arrival (AoA) calculations
- Particle filter operations (initialization, resampling)
- AoA standard deviation modeling

### `jointSyncLoc.py` (Core Module)
Contains the `jointSyncLoc` class with key functionality for joint sync & loc:
- BRF and PF localization implementations
- AP positioning and trajectory generation
- Timestamp exchange simulation
- LOS/NLOS scenario handling

### `SyncNet.py` (Network Synchronization)
Implements network clock synchronization with:
- Belief Propagation algorithm
- Hybrid synchronization (BP + Kalman Filter)
- Timestamp exchange modeling
- Grandmaster clock reference support

## Output 
The system generates:
- Position Estimation CDF (Comparative PF vs BRF)
- Offset Estimation CDF (Comparative PF vs BRF)
- Console output of RMSE metrics
- Network synchronization statistics
## Installation

```bash
git clone [your-repository-url]
cd [your-repository-name]
pip install numpy matplotlib
```
Run the `main.py` with your desired parameters.