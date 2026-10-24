# f1-aero-efficiency-explorer-
Web dashboard comparing F1 teams' aerodynamic efficiency (drag vs. downforce) from public telemetry. Python (FastF1, FastAPI), PostgreSQL, React + TypeScript. Work in progress.

# F1 Aero Efficiency Explorer

>  Work in progress

A web dashboard that compares the aerodynamic efficiency of Formula 1 teams using public telemetry data.

## The idea

Every F1 car is a trade-off between **drag** (straight-line speed) and **downforce** (cornering speed). This project estimates where each team sits on that trade-off:

- **Drag proxy:** top speed on straights / speed trap
- **Downforce proxy:** minimum speed through high-speed corners

Plotting both per team, per race, shows how each car's aerodynamic setup changes between low- and high-downforce tracks.

## Planned features

- Drag vs. downforce scatter plot per team and race
- Speed trace comparison between two drivers on the same lap
- Season view: how teams shift across different circuits
- Documented methodology and known limitations

## Stack

Python (FastF1, Pandas) · FastAPI · PostgreSQL · React + TypeScript · Docker

## Data

Public timing and telemetry data via [FastF1](https://github.com/theOehrly/Fast-F1). Not affiliated with Formula 1 or any team.
