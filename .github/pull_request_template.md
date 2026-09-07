## Summary

<!-- One or two sentences: what and why. -->

## Type

<!-- tests / task template / policy template / finetune / docs / chore -->

## Test plan

- [ ] `make test` passes locally
- [ ] CI smoke run linked: <!-- paste run URL -->
- [ ] GPU-box validation (if applicable): <!-- e.g. robodojo.sh smoke --only <task> -->

## Checklist

- [ ] New/changed template covered by `tests/test_templates.py`
- [ ] No secrets, weights, or generated artifacts in the diff
- [ ] Docs updated (`OPENVLA_ROBODOJO_SETUP.md` or template README)
- [ ] Fine-tune/policy runs record checkpoint location + dataset revision (if applicable)
