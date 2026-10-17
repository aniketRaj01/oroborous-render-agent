# Instance runbook

How the GPU instance gets, checks and runs the render agent, and what to do by hand. This page uses
placeholders (`<region>`, `<bucket>`, `<ssm-prefix>`, ...): fill them in from your own AWS account;
nothing real belongs in this repository.

## What happens at every boot

```
systemd starts render-agent.service
  ExecStartPre: bootstrap.py   (stdlib only, runs as the service user, has a 5 minute budget)
    1. lock, then fetch tags from the public repository into a bare mirror (retries with backoff)
    2. pick the newest tag vX.Y.Z whose commit is on main and whose pyproject version matches it
    3. export it to releases/<tag>, build venvs/<hash of requirements.lock> only if that is new
    4. run the new release's selfcheck inside that venv
    5. only then switch current / current-venv (atomically) and keep the old pair as previous
    6. write /run/render-agent/version.env with AGENT_VERSION=<tag>+<short sha>
  ExecStart: current-venv/bin/python -m render_agent run
```

Any failure in steps 1 to 5 (no network, GitHub down, a bad tag, a failed install or selfcheck, the
time budget) is logged and the agent starts on the version that already works. The bootstrap fails
the start only when nothing usable is installed. The version that runs is sent with every claim
(`agentVersion`) and is in the journal.

**What updates by itself and what does not.** New releases of the agent arrive by tag. But
`bootstrap.py` and the systemd units are installed by `deploy/install.sh` and are *not* part of a
release: a change under `deploy/` takes effect only when you re-run `install.sh` (and rebuild the
image).

## Layout on the instance

```
/opt/render-agent/
  bootstrap.py              installed by install.sh
  repo/                     bare mirror of the public repository (never executed from)
  releases/<tag>/           exported source of a tag, plus RELEASE (tag and commit)
  venvs/<hash>/             one virtualenv per sha256-12 of requirements.lock
  current -> releases/<tag>          current-venv -> venvs/<hash>
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

It fetches the newest tag and builds its environment; the last line says which version will run, for
example `bootstrap: running v1.0.0+0123456789ab (new; first install)`. The model server (a separate
unit, `ltx-server.service`) is installed by its own procedure; the agent waits for its `/health`.

### 5. Verify (see "First boot checklist" below), then make the image.

## Cutting a release

A release is a deliberate act on your machine; the instance never follows `main` directly.

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
3. Instances pick the tag up at their next boot (or `sudo systemctl restart render-agent`).

Released versions are immutable: a tag that is moved to another commit is refused by the instances.
To fix a bad release, cut a new, higher version.

## Rolling back

- **Fastest, per instance:** put `RENDER_AGENT_PIN=v1.2.2` in `/etc/render-agent/deploy.env` and
  restart the service (or reboot). The bootstrap then runs exactly that tag (it must still be on
  `main` and match its pyproject version). Remove the line to follow the newest tag again.
- **For every instance:** delete the bad tag from the remote
  (`git push origin :refs/tags/v1.2.3`). The next bootstrap prunes it from its mirror and moves to the
  newest remaining valid tag, keeping `previous` as the version it came from.
- If a new tag only *looks* bad on one boot (network, PyPI) nothing needs doing: the agent stayed on
  the working version.

## Reading logs

```sh
journalctl -u render-agent -b             # this boot: the bootstrap and then the agent
journalctl -u render-agent -b | grep 'bootstrap:'
journalctl -u render-agent -b | grep run_summary
```

Bootstrap lines start with `bootstrap:` and the last one states the outcome:

| Last bootstrap line | Meaning |
|---|---|
| `running v1.2.3+<sha> (new; was v1.2.2)` | upgraded |
| `running v1.2.3+<sha> (unchanged)` | already the newest valid tag |
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
   - `journalctl -u render-agent -b | grep 'bootstrap:' | tail -3` ends with `running <tag>+<sha>`
     and the tag is the one you expect (`cat /run/render-agent/version.env`);
   - `readlink /opt/render-agent/current` is `releases/<that tag>`;
   - `journalctl -u render-agent -b | grep -E 'claim|run_start'` shows the agent started and claimed
     (or an empty claim);
   - the Job API shows your `AGENT_ID` and the same `agentVersion` on the jobs it claimed.
3. Optionally test the safety nets: set `RENDER_AGENT_PIN` to an older tag, restart, confirm the
   version changes; remove it, restart, confirm it returns.

## Opt-in: dead-man timer

`render-agent-deadman.timer` powers the instance off 65 minutes after boot, as a last-resort safety
net. It is **not enabled by default**. Enable it with `sudo deploy/install.sh --enable-deadman ...`
or `sudo systemctl enable --now render-agent-deadman.timer`. It does nothing while
`/etc/render-agent/no-shutdown` exists, so before working on a machine over SSH run
`sudo touch /etc/render-agent/no-shutdown`. Disable with
`sudo systemctl disable --now render-agent-deadman.timer`.

### Unverified: powering off when the agent ends

Stopping the instance when the agent finishes is handled outside this repository. systemd offers
`SuccessAction=poweroff` for a service. It has **not** been tried here, and it is not known whether a
manual `systemctl stop render-agent` also triggers it. Treat it as an opt-in experiment, on a
throwaway instance first, never as a default.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Agent exits 2 right after start | `agent.env` still has the placeholders, or a value is invalid; the journal names the setting |
| Exit 3 | missing `lambda:InvokeFunctionUrl` or `lambda:InvokeFunction` on the instance role, wrong region, or no credentials |
| `fallback: fetch failed` on every boot | no route to GitHub; the agent keeps running the installed version |
| `rejected vX: commit is not on main` | the tag was made on a branch; release from `main` with `release.py` |
| `rejected vX: pyproject version does not match the tag` | the tag and `pyproject.toml` disagree; cut a new version |
| `fallback: ... different commit than the installed release` | a tag was moved; releases are immutable, cut a new version |
| `fallback: selfcheck failed (...)` | the new release does not import on this instance; the message names the import |
| `fallback: time budget exceeded` | a slow network; the next boot tries again (or raise `RENDER_AGENT_BOOTSTRAP_BUDGET_SECONDS`) |
| `no usable version` | first boot with no network and nothing pre-loaded; run step 4 of the setup |
