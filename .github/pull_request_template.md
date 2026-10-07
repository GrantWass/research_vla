## Summary

<!-- One or two sentences: what and why. -->

## Type

<!-- robot / sim / policies / serving / templates / tests / docs / chore -->

## Test plan

- [ ] `make test` passes locally
- [ ] CI smoke run linked: <!-- paste run URL -->
- [ ] Sim or robot validation (if applicable): <!-- e.g. make smoke POLICY=pi05, or a dry run on the arm -->

## Checklist

- [ ] New/changed template covered by `tests/test_templates.py`
- [ ] No secrets, weights, or generated artifacts in the diff
- [ ] Docs updated (`docs/` or template README)
- [ ] Fine-tune/policy runs record checkpoint location + dataset revision (if applicable)
