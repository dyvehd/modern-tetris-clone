"""The 100L cheese-race trainer: external bot backends, AI shadows,
automove and live feedback.

Module map:
- ``backends`` — the uniform Backend/BotAdvice seam over four external
  engines plus our own beam (placements in, placements out; navigation is
  always ours).
- ``advisor``  — the worker thread that keeps advice fresh for the
  current piece (backends are too slow to call inline: 50-250 ms).
- ``annotation`` — chess-style move quality: rank the player's placement
  among the bot's scored candidates, z-scored against the candidate
  distribution.
- ``app``      — pygame wiring: mode config, keybinds, the trainer state
  machine (advice display, automove, live feedback) hooked into the game
  loop.
"""
