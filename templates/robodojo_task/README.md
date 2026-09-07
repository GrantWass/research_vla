# RoboDojo task template

Copy these two files into the RoboDojo checkout, renaming `my_task` to your
task name (lowercase `snake_case`, unless an uppercase asset name forces an
exception like `push_T` / `play_Xylophone`):

```bash
TASK=my_new_task   # <-- rename
cp templates/robodojo_task/my_task.yml RoboDojo/task/RoboDojo/config/${TASK}.yml
cp templates/robodojo_task/my_task.py  RoboDojo/task/RoboDojo/tasks/${TASK}.py
# then rename the classes inside my_task.py: MyTaskCommon + <TASK> (see TODOs)
```

## Naming rule (all four must match)

- `task/RoboDojo/config/<task>.yml`
- `task/RoboDojo/tasks/<task>.py` (module basename)
- exported env class `<task>` (snake_case, same as module — the registry does
  `load_task_class(task_name)` and imports `task.RoboDojo.tasks.<task_name>`)
- result path `eval_result/RoboDojo/<task>/` (automatic once names match)

## Checklist before requesting review

- [ ] `run_reward()` calls `self.reward_manager.check(...)` with real predicates —
      never leave it trivially always `True` (corrupts success metrics silently).
- [ ] YAML `label: [...]` entries match the `label=` strings in Python exactly.
- [ ] `reset()` resets `reward_manager`; override `soft_reset()` if the task
      holds episode state that must clear between episodes.
- [ ] `step_lim` fits the horizon (long-horizon tasks need more; truncation
      looks like policy failure).
- [ ] Task inherits `TaskEnv` (not `SyncCollectEnv` / `SyncRobotEnv`).
- [ ] No `print` / `breakpoint` in task logic; use the shared logger if needed.
- [ ] Validate Isaac-free first: `python scripts/internal/task_inventory.py --format json --check`
      then `bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/demo_policy
      --ckpt demo --policy-env demo-env --only <task> --dry-run`.
      Runtime acceptance needs the GPU box: same `smoke` without `--dry-run`,
      exit `0` **and** `_result.json` with `eval_time >= 1`.
