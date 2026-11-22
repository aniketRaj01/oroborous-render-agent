# Instance runbook

How the GPU instance gets, checks and runs the render agent, and what to do by hand. This page uses
placeholders (`<region>`, `<bucket>`, `<ssm-prefix>`, ...): fill them in from your own AWS account;
nothing real belongs in this repository.

## What happens at every boot

```
systemd starts render-agent.service
  ExecStartPre: bootstrap.py   (stdlib only, runs as the service user, has a 5 minute budget)
    1. lock, then fetch main from the public repository into a bare mirror (retries with backoff)
    2. pick the version: the head of origin/main (default, RENDER_AGENT_TRACK=main), or with
       RENDER_AGENT_TRACK=tags the newest tag vX.Y.Z whose commit is on main and whose pyproject
       version matches it; RENDER_AGENT_PIN overrides both
    3. export it to releases/<name>, build venvs/<hash of requirements.lock> only if that is new
    4. run the new release's selfcheck inside that venv
    5. only then switch current / current-venv (atomically) and keep the old pair as previous
    6. write /run/render-agent/version.env with AGENT_VERSION=main+<12 hex> (tags mode:
       <tag>+<12 hex>; a pinned commit: commit+<12 hex>; at most 64 characters)
  ExecStart: current-venv/bin/python -m render_agent run
```

Any failure in steps 1 to 5 (no network, GitHub down, a bad commit, a failed install or selfcheck, the
time budget) is logged and the agent starts on the version that already works. The bootstrap fails
the start only when nothing usable is installed. The version that runs is sent with every claim
(`agentVersion`) and is in the journal.

**What updates by itself and what does not.** Whatever is merged to `main` reaches the instance at
its next boot (so a merge is a deployment: the self-check and the fallback are the only gate, and CI
must be green before you merge). But `bootstrap.py` and the systemd units are installed by
`deploy/install.sh` and are *not* part of the exported source: a change under `deploy/` takes effect
only when you re-run `install.sh` (and rebuild the image).

**Moving an existing instance to main tracking.** The bootstrap that follows `main` is itself
`bootstrap.py`, so an instance installed earlier still follows tags until you run
`sudo deploy/install.sh` once from a checkout that contains this change (with the same options you
used for the first install; it is idempotent and keeps your `agent.env` and `deploy.env`). The next
boot then tracks `main`. Until then it keeps running the newest tag, which is safe. Remove any
`RENDER_AGENT_PIN` line from `deploy.env` unless you want it.

## Layout on the instance

```
/opt/render-agent/
  bootstrap.py              installed by install.sh
  repo/                     bare mirror of the public repository (never executed from)
  releases/<name>/          exported source, plus RELEASE (tag and commit); <name> is the tag, or
                            main-<12 hex> / commit-<12 hex> when following a commit
  venvs/<hash>/             one virtualenv per sha256-12 of requirements.lock
  current -> releases/<name>         current-venv -> venvs/<hash>
  previous -> ...                    previous-venv -> ...     (the last good pair)
/etc/render-agent/agent.env   AGENT_ID and the SSM location of the Job API URL (mode 0600)
/etc/render-agent/deploy.env  optional bootstrap settings, e.g. RENDER_AGENT_PIN (mode 0600)
/run/render-agent/version.env AGENT_VERSION=..., rewritten at every start
/var/lib/render-agent/        the service's working directory
```

## First-time setup

### 1. AWS items (once, by hand)

1. **SSM parameter.** Create a `SecureString` named `<ssm-prefix>/job-api-url` holding the Job API
   function URL. The agent reads every parameter directly under `<ssm-prefix>` at start-up and maps
   the last path segment to a setting (`job-api-url` becomes `JOB_API_URL`). Nothing else sensitive
   is needed.
2. **Instance role** (an IAM role attached to the instance profile) needs:
   - read access to that parameter: `ssm:GetParametersByPath` on `<ssm-prefix>` (and
     `kms:Decrypt` on the key if you use a customer-managed key);
   - on the clip bucket: `s3:PutObject`, `s3:GetObject` (the agent verifies each upload with a HEAD
     request, which needs it) and `s3:AbortMultipartUpload` on `arn:aws:s3:::<bucket>/plans/*`,
     plus `s3:ListBucket` on `arn:aws:s3:::<bucket>`;
   - to call the Job API: **both** `lambda:InvokeFunctionUrl` **and** `lambda:InvokeFunction` on the
     function. AWS requires both for function URLs. A 403 from the agent (exit code 3) almost always
     means one of them is missing.
3. **Outbound access** from the instance to `github.com` (HTTPS), `pypi.org` and
   `files.pythonhosted.org` (only when a new `requirements.lock` has to be installed), and the AWS
   endpoints for SSM, S3 and Lambda.

### 2. Prepare the machine image (once)

On the instance (or the machine you will turn into an image), as a user with `sudo`:

```sh
git clone <repository-url> render-agent-src      # the public repository
cd render-agent-src && git checkout <a release tag>   # for example the latest vX.Y.Z
sudo deploy/install.sh --service-user <login-user>
```

`install.sh` checks for `git`, `python3` 3.10 or newer and the `venv` module and stops with the exact
missing packages (add `--install-packages` to let it call `apt-get`). It creates the directories,
copies `bootstrap.py` and the units, creates the two env files from the examples **only if missing**
(owned by the service user, mode 0600: the agent refuses a config file readable by group or others),
runs `systemctl daemon-reload` and enables `render-agent.service`. It does **not** enable the
dead-man timer. Preview with `--dry-run`. It is safe to re-run.

### 3. Configure the agent

Edit `/etc/render-agent/agent.env` and replace the three placeholder values (until you do, the agent
stops at start-up naming the setting):

```
AGENT_ID=<instance-id>                     # the EC2 instance id; stable, never change it
RENDER_AGENT_SSM_PREFIX=<ssm-prefix>       # where job-api-url lives
RENDER_AGENT_SSM_REGION=<region>
```

`AGENT_ID` must stay the same across restarts: the Job API resumes a crashed render only for the
same id (the contract defines it as the EC2 instance id).

### 4. Pre-load the first release (recommended)

So that even the very first boot does not depend on the network, run the bootstrap once while the
image is still online:

```sh
sudo -u <login-user> python3 /opt/render-agent/bootstrap.py
```

It fetches the head of `main` and builds its environment; the last line says which version will run, for
example `bootstrap: running main+0123456789ab (new; first install)`. The model server (a separate
unit, `ltx-server.service`) is installed by its own procedure; the agent waits for its `/health`.

### 5. Verify (see "First boot checklist" below), then make the image.

## Cutting a release

By default the instance follows `main`, so cutting a tag is not needed for it to get a change. Tags
are still useful as named rollback points, and they are what an instance follows with
`RENDER_AGENT_TRACK=tags` (the old behaviour, the fast way back if main tracking misbehaves).

1. In a pull request, set `version` in `pyproject.toml` to the new number (for example `1.2.3`),
   run `uv lock` and `scripts/export_requirements.sh`, and merge when CI is green. CI fails if
   `requirements.lock` is not exactly what `uv.lock` exports.
2. On a clean checkout of `main` that equals the remote:

   ```sh
   python scripts/release.py v1.2.3 --dry-run    # runs every check, changes nothing
   python scripts/release.py v1.2.3              # creates the annotated tag and pushes only it
   ```

   It refuses unless: the tag is `vMAJOR.MINOR.PATCH`; `pyproject.toml` has that version; the tree is
   clean; you are on `main` and equal to the remote; `requirements.lock` is in sync; CI is green for
   `HEAD` (`gh` must be logged in, or pass `--skip-ci-check`); the tag does not exist yet. All
   failures are listed together.
3. Instances in tags mode pick the tag up at their next boot (or `sudo systemctl restart render-agent`).

Released versions are immutable: a tag that is moved to another commit is refused by instances in
tags mode. To fix a bad release, cut a new, higher version.

## Rolling back

- **Fastest, per instance:** put `RENDER_AGENT_PIN=<tag or commit hash>` (for example `v1.2.2`, or a
  commit hash of at least 7 characters from `git log`) in `/etc/render-agent/deploy.env` and restart
  the service (or reboot). The bootstrap then runs exactly that version (a tag must match its
  pyproject version; any pin must be on `main`). Remove the line to follow `main` again. A pin stays
  until removed, so main merges stop arriving while it is set.
- **Back to released tags:** put `RENDER_AGENT_TRACK=tags` in `deploy.env` and restart: the instance
  runs the newest valid tag again and ignores later merges until you cut the next tag.
- **For main tracking:** revert the bad commit on `main` through a PR; the next boot follows it. The
  self-check refuses a commit that does not import, and the agent then stays on the last good one.
- **For every instance in tags mode:** delete the bad tag from the remote
  (`git push origin :refs/tags/v1.2.3`). The next bootstrap prunes it from its mirror and moves to the
  newest remaining valid tag, keeping `previous` as the version it came from.
- If a new tag only *looks* bad on one boot (network, PyPI) nothing needs doing: the agent stayed on
  the working version.

## Chained clips (continuity between the clips of a plan)

For a chained plan the Job API gives each clip a `startImage` (an S3 location of the previous clip's
last frame) and a `lastFrameUpload` (where to put this clip's own last frame). Both are optional; an
API that does not send them changes nothing.

- **ffmpeg** extracts the last frame, so the instance needs it: `sudo apt-get install -y ffmpeg`
  (add it to the image). Without it a chained clip fails with `upload_error` ("ffmpeg is required ...").
  `python -m render_agent check` prints a WARNING when the claim contains chained clips and ffmpeg is
  missing, and the self-check logs `ffmpeg_not_found` as a warning only.
- **Order and failure.** Clips run in position order. After a chained clip fails or is given up, later
  chained clips of that plan are not started in this run (logged as `clip_skipped`, reason
  `earlier_clip_of_chained_plan_failed`); they wait for a later claim.
- **Where frames live.** The previous clip's PNG is kept in `SCRATCH_DIR` (`p<plan>-<n>-last.png`) until
  the next clip has used it, so normally nothing is downloaded. After a restart or when a plan is
  resumed, the start frame is fetched from S3 (`p<plan>-start-<n>.png`, mode 0644, under the real path
  of `SCRATCH_DIR`). The model server only accepts an image inside its `ALLOWED_OUTPUT_DIR`, so
  `SCRATCH_DIR` and the model server's `ALLOWED_OUTPUT_DIR` must be the same directory (both default to
  `/tmp/render-agent`). A missing start frame fails the clip as `render_error` ("start frame is
  missing"): it is never rendered from text alone.
- **IAM.** The frame objects are under `plans/<id>/` in the clip bucket, which the instance role
  already covers (`s3:PutObject`, `s3:GetObject` on `plans/*`).
- **Extra condition images.** A clip may also carry `conditionImages` in the claim: `{bucket, region, key,
  frameIdx, strength}` entries (an anchor image of a recurring character at frame 0 or at a later frame).
  They are downloaded per clip into `SCRATCH_DIR` (`p<plan>-<n>-cond-<i>.<ext>`), sent to the model server
  as `conditionImages` (together with the chained start frame, which keeps frame 0), and deleted when the
  clip is over. A missing or unreadable one, more than 3 images, or two at the same frame fails the clip as
  `render_error`; the clip is never rendered without an image it asked for.
- **Logs.** `start_frame_downloaded` / `start_frame_reused` / `frame_extracted` / `frame_uploaded`, with
  keys, sizes and seconds, never the prompt.

## Reading logs

```sh
journalctl -u render-agent -b             # this boot: the bootstrap and then the agent
journalctl -u render-agent -b | grep 'bootstrap:'
journalctl -u render-agent -b | grep run_summary
```

Bootstrap lines start with `bootstrap:` and the last one states the outcome:

| Last bootstrap line | Meaning |
|---|---|
| `running main+<sha> (new; was main+<sha>)` | moved to a newer commit of `main` (tags mode: `v1.2.3+<sha> (new; was v1.2.2)`) |
| `running main+<sha> (unchanged)` | already the head of `main` (tags mode: the newest valid tag) |
| `running v1.2.2+<sha> (fallback: <reason>)` | a newer version was refused or failed; still running the old one |
| `no usable version (<reason>)` | nothing installed and nothing could be installed: the start fails |

Agent lines are `key=value`. `run_summary` ends every run with `done`, `failed`, `skipped`,
`runtime_s` and `reason`. Exit codes: 0 normal end, 2 configuration error, 3 Job API auth error, 4
unexpected error, 5 systemic failure (model server never ready, or too many failed clips in a row).
The unit does not restart on 2, 3 or 5; fix the cause, then `sudo systemctl restart render-agent`.

## First boot checklist

Do this on the real instance, after setup and the image is built from it:

1. Stop the instance, then start it. Do not log in or touch anything.
2. After a few minutes, SSH in and check:
   - `systemctl status render-agent` shows it running (or a normal exit if there was no work);
   - `journalctl -u render-agent -b | grep 'bootstrap:' | tail -3` ends with `running main+<sha>`
     and the sha is the head of `main` (`cat /run/render-agent/version.env`);
   - `readlink /opt/render-agent/current` is `releases/main-<that sha>`;
   - `journalctl -u render-agent -b | grep -E 'claim|run_start'` shows the agent started and claimed
     (or an empty claim);
   - the Job API shows your `AGENT_ID` and the same `agentVersion` on the jobs it claimed.
3. Optionally test the safety nets: set `RENDER_AGENT_PIN` to an older tag or commit, restart,
   confirm the version changes; remove it, restart, confirm it returns.

## Opt-in: dead-man timer

`render-agent-deadman.timer` powers the instance off 65 minutes after boot, as a last-resort safety
net. It is **not enabled by default**. Enable it with `sudo deploy/install.sh --enable-deadman ...`
or `sudo systemctl enable --now render-agent-deadman.timer`. It does nothing while
`/etc/render-agent/no-shutdown` exists, so before working on a machine over SSH run
`sudo touch /etc/render-agent/no-shutdown`. Disable with
`sudo systemctl disable --now render-agent-deadman.timer`.

## Opt-in: power off when the agent has no work

After `IDLE_EXIT_MINUTES` (5) with nothing to claim the agent exits 0 and the instance keeps
running (and billing). With `AUTO_SHUTDOWN_ON_IDLE=true` that idle exit also powers the instance
off. The instance's shutdown behaviour is "stop", so an operating-system poweroff stops it (it is
never terminated); no AWS call is made and the instance role needs no EC2 permission. It is **off by
default**.

**Turn it on** (once per instance):

1. Put `AUTO_SHUTDOWN_ON_IDLE=true` in `/etc/render-agent/deploy.env` (optionally
   `AUTO_SHUTDOWN_GRACE_SECONDS=10`, the wait before the poweroff so the last log lines are written).
2. Run `sudo deploy/install.sh` (same options as the first install) from a checkout that has this
   feature. It installs the root-owned helper `/usr/local/sbin/render-agent-poweroff` and the updated
   unit, keeps your `deploy.env` as it is, and never turns the flag on by itself. Re-running it is
   harmless.
3. The setting is read when the service starts (next boot, or `sudo systemctl restart render-agent`).

**How it works.** The agent runs unprivileged (and with `NoNewPrivileges`, so `sudo` would not work).
After an idle exit it writes `/run/render-agent/poweroff-requested`; the unit's `ExecStopPost` (the
`+` prefix runs it as root) starts the helper, which powers off only if that file exists, the service
ended with exit 0, and the guard file below is absent. Nothing else writes the marker: a crash, a
signal, `systemctl stop`, the run deadline, `--max-clips`, a Job API or configuration error never
power off. The instance also stays up when the bootstrap fell back to an older version
(`AGENT_BOOTSTRAP_FALLBACK=true` in `/run/render-agent/version.env`), when a clip output file is still
in the scratch directory, when claimed clips were left unstarted, or when the model server is not
ready: the log says why. The marker lives in `/run`, so it is gone after any stop or reboot and can
never power off the next boot.

**Keep the instance up** (debugging over SSH): `sudo touch /etc/render-agent/no-shutdown`; remove the
file to re-enable. The agent checks it before and after the grace wait, and the helper checks it
again as root. The dead-man timer honours the same file.

**What it looks like in the journal** (`journalctl -u render-agent -b`):

```
idle_exit idle_seconds=300
run_summary ... reason=idle
idle exit: powering off instance
poweroff-if-requested: idle exit, powering off instance
```

and when it chooses to stay up: `idle_exit_staying_up why=...`. If the poweroff command itself
fails the helper logs `poweroff failed; the instance stays up` and the service still ends normally.

A stopped instance is started again by the boot Lambda at its next scheduled boot; the agent then
starts as usual (nothing is kept on disk between boots that could stop it again). The boot Lambda's
alarm for an instance that has run too long remains the backstop if a poweroff ever fails.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Agent exits 2 right after start | `agent.env` still has the placeholders, or a value is invalid; the journal names the setting |
| Exit 3 | missing `lambda:InvokeFunctionUrl` or `lambda:InvokeFunction` on the instance role, wrong region, or no credentials |
| `fallback: fetch failed` on every boot | no route to GitHub; the agent keeps running the installed version |
| `commit is not on main` (a pin) | the pinned commit is not part of `main`; pin a commit from `main` |
| `RENDER_AGENT_TRACK must be main or tags` | a typo in `deploy.env`; the instance stays on the version it has |
| `rejected vX: commit is not on main` | the tag was made on a branch; release from `main` with `release.py` |
| `rejected vX: pyproject version does not match the tag` | the tag and `pyproject.toml` disagree; cut a new version |
| `fallback: ... different commit than the installed release` | a tag was moved; releases are immutable, cut a new version |
| `fallback: selfcheck failed (...)` | the new release does not import on this instance; the message names the import |
| `fallback: time budget exceeded` | a slow network; the next boot tries again (or raise `RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS`) |
| `no usable version` | first boot with no network and nothing pre-loaded; run step 4 of the setup |
