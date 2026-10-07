"""Real-robot bridge: drive a physical dual-arm robot from the pi0.5 policy server.

The server (RoboDojo/XPolicyLab/client_server/ws/model_server.py) is unchanged;
this package replaces the Isaac Sim env loop with hardware I/O plus a safety
filter. See docs/robot.md.
"""
