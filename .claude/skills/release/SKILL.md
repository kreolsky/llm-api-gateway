---
name: release
description: Tag a version and fast-forward dev into main. Релиз, выпусти версию, мерж dev в main, слить dev в main, fast-forward main, новая версия, release, tag a version.
---

# Release — tag `vX.Y.Z`, fast-forward `dev` → `main`

A release = release note on `dev` + annotated tag `vX.Y.Z` on the audited `origin/dev` SHA +
fast-forward of `origin/main` to that SHA. **A release does not deploy** — the prod server
is updated by the separate `deploy-server` skill, which stamps the tag into `/health`.

**Invoking this skill IS the operator's go-ahead for the tag and the push to `main`.** The
operator is asked exactly one thing, the version level (§2). The only other halts: a `STOP`
from the pre-flight, a red unit suite, a `WARN` whose decision cannot be made from the tree.

## 1. Pre-flight

```bash
git push origin dev                                  # the audit reads origin/dev
python3 .claude/scripts/release-audit.py --fetch     # STOP => exit 1, blocks the release
.venv/bin/python -m pytest tests/unit/ -q            # the released SHA is green
```

The script is read-only: fast-forward possible, local `HEAD == origin/dev`, commit types
since the last tag + recommended level, `requirements.txt`/`Dockerfile` changed (the next
deploy is a rebuild), new `Settings` fields (new env vars), the newest note. Each `WARN`
gets a line in the note. A non-ff `main` (it has commits `dev` lacks) → halt and ask; never
`--force`. Tracked edits in the working tree are not part of the release — the tag goes on
`origin/dev`, never on the working tree.

## 2. Version (ask the operator)

Versions are annotated git tags `vMAJOR.MINOR.PATCH`; no file in the tree carries a version.
Recommendation by **presence** of a commit type since the last tag: `BREAKING CHANGE` in a
body or `type!:` in a subject → major; else any `feat` → minor; else patch. Major = a client
of the router breaks: a response shape, an error envelope, an access rule, a removed endpoint
or config key.

Ask with `AskUserQuestion`, recommended level FIRST, each option with its evidence (counts
from the pre-flight). A level BELOW the recommendation gets one line in the note:
`version: v1.2.1 — feat×3 ruled internal`. Do not create the tag yet — it is created in
§4, so an abandoned release leaves no orphan tag.

## 3. Release note

`docs/release/release-YYYY-MM-DD-vX.Y.Z.md`, **always English** — the whole note, whatever the
language of the request (`cyrillic-src-gate.py` blocks Cyrillic in `docs/release/`); shaped
like the newest existing note (`ls docs/release/*.md | tail -1`); H1 `# Release vX.Y.Z — YYYY-MM-DD`.

- One-paragraph summary: commit count since the previous tag
  (`git log v<LAST>..origin/dev --oneline --no-merges | wc -l`, taken ONCE — the note's own
  commit changes it) + the headline theme.
- Changes grouped by theme, from the commit bodies — client-visible first, written as what
  a client of the router can now do (`.claude/rules/testing.md` → *Reports are user
  scenarios*), never a raw log.
- `# Upgrade actions` (mandatory): restart or full rebuild, each new env var with
  its default and whether prod needs a value, new/changed keys in `config/*.yaml` (prod
  configs are authoritative and are edited by hand), usage DB schema changes (additive?).
- `# Known limitations` when by-design caveats ship.

Commit on `dev` with an English message `docs(release): vX.Y.Z note`, after
`.claude/scripts/pre-commit-gates.sh` (unpiped), then `git push origin dev`.

## 4. Tag and fast-forward

Push both by the SHA of the audited `origin/dev`, never by a local branch name. Never
create a local merge commit on `main` (`.claude/rules/git-strategy.md` → *Release*).

```bash
git fetch origin
REL=$(git rev-parse origin/dev)
test -z "$(git log origin/dev..origin/main --oneline)"   # still a fast-forward
VER=v1.0.0                                               # the level chosen in §2
git tag -a "$VER" "$REL" -m "Release $VER — $(date +%F)

See docs/release/release-$(date +%F)-$VER.md"
git push origin "refs/tags/$VER"
git push origin "$REL:refs/heads/main"
git rev-parse "$VER^{}" origin/main                       # both lines are $REL
```

## Output

Version tag (and the overridden recommendation, if any), note path, commit count, the new
`origin/main` SHA, the GitHub Release page of the tag (created by the `release` job of
`.github/workflows/publish.yml` from the note; missing ⇒ re-run it by hand with `tag:`) — and that prod is NOT updated until `deploy-server` runs, after which
`/health` on the server reports `vX.Y.Z`.
