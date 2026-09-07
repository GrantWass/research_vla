"""Template for a RoboDojo task. Copy to task/RoboDojo/tasks/<task>.py.

TODOs:
  1. Rename MyTaskCommon -> <Your>Common and my_task -> <task> (must equal
     the YAML basename and module basename; the registry imports
     task.RoboDojo.tasks.<task> and expects class <task>).
  2. Fill in run_reward() with real reward_manager predicates (never
     trivially-True) and gen_instruction().
  3. Tune step_lim to the task horizon.
"""
from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


# TODO: rename to <Your>Common, e.g. WipeTableCommon for task wipe_table.
class MyTaskCommon:
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400  # TODO: tune to the task horizon

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def soft_reset(self, seed=None, options=None):
        # TODO: override only if the task holds state between episodes that
        # reset() does not clear. Otherwise delete this method.
        super().soft_reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        # TODO: replace with the real success condition. Labels must match
        # the YAML `label:` list exactly.
        self.reward_manager.check(
            [
                self.reward_manager.is_axis_up(label="target", axis=[0, 0, 1], threshold=45),
                self.reward_manager.all_robot_back_to_origin(),
            ]
        )

    def get_score(self):
        # TODO: optional dense/partial scoring; mirror run_reward predicates.
        # See task/RoboDojo/tasks/stack_bowls.py get_score() for the pattern.
        self.run_reward()

    def gen_instruction(self, env_idx):
        # TODO: language instruction(s) shown to the policy.
        templates = ["Put the bowl upright on the table."]
        return templates


# TODO: rename to the task name (== file basename == YAML basename).
class my_task(MyTaskCommon, TaskEnv):
    pass
