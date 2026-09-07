# Remote access: SSH into the Windows 11 Pro box

Goal: `ssh winbox` from this Mac (or VS Code Remote-SSH) on **any** network —
home LAN, different WiFi, phone hotspot. The cross-network part rides on
**Tailscale** (no router/port-forwarding config, punches through NATs).

Mac side is already ready: system `ssh` + an `id_ed25519` keypair in `~/.ssh/`.
Do the steps in order; each one isolates a layer, so a failure tells you
exactly what broke.

## 1. Enable SSH on Windows (Administrator PowerShell)

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
Get-NetFirewallRule -Name *ssh*   # confirm port 22 is allowed
```

Get the LAN address for the first test: `ipconfig` → IPv4 (e.g. `192.168.1.50`).

## 2. Test on the same WiFi (password auth)

```bash
ssh <winuser>@192.168.1.50
```

Succeed here before touching keys or Tailscale.

## 3. Key auth (before exposing anything remotely)

On the Mac, print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

On Windows, append it to `C:\Users\<winuser>\.ssh\authorized_keys` (create
`.ssh` if needed), then fix ACLs — Windows OpenSSH rejects the key otherwise:

```powershell
icacls "$env:USERPROFILE\.ssh\authorized_keys" /inheritance:r /grant:r "$($env:USERNAME):R"
```

Retest `ssh <winuser>@192.168.1.50` — no password prompt expected. Then harden
`C:\ProgramData\ssh\sshd_config` (Admin editor, restart `sshd` after):

```
PasswordAuthentication no
PermitRootLogin no
```

## 4. Cross-network via Tailscale

Install Tailscale on **both** machines (`tailscale.com/download`, or
`brew install tailscale` on the Mac) and sign in with the **same account**.

**Option A — plain SSH over the tailnet (recommended).** Keeps standard SSH
semantics (keys from step 3), which fits split-machine flows like RoboDojo's
`robodojo.sh server` / `client` across machines.

```bash
tailscale status         # note the Windows box's 100.x.y.z address
ssh <winuser>@100.x.y.z  # works from any network
```

Pin it in `~/.ssh/config`:

```
Host winbox
  HostName 100.x.y.z
  User <winuser>
  IdentityFile ~/.ssh/id_ed25519
```

Then `ssh winbox` from anywhere. VS Code's Remote-SSH extension picks up
`winbox` automatically.

**Option B — Tailscale SSH (no keys).** On Windows: `tailscale up --ssh`.
From the Mac: `tailscale ssh <winuser>@winbox`. Auth rides on the Tailscale
identity instead of SSH keys.

## 5. Verify from a different network

Join a phone hotspot on the Mac and run `ssh winbox`. If it connects, the
setup is complete.

## Alternatives (only if Tailscale is off the table)

- **Port forwarding + dynamic DNS**: forward a non-standard external port
  (e.g. 2222 → 22) on the home router to the PC; use DuckDNS/No-IP for the
  rotating residential IP. Exposes SSH to the internet — key-only auth is
  mandatory. More fragile than Tailscale.
- **Cloudflare Tunnel** (`cloudflared access ssh`): no open ports, but more
  moving parts than Tailscale for two machines.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Connection timed out` on LAN | Windows Firewall or wrong IP; recheck step 1, `ping` the IP first |
| `Permission denied (publickey)` | `authorized_keys` ACLs (step 3 `icacls`), wrong username, or key not appended as one line |
| `Connection refused` on tailnet IP | Tailscale not `up` / not same account on both ends; `tailscale status` on both |
| Works on LAN, fails on hotspot | Tailscale not connected on one side; classic SSH can't cross NATs — this is the step Tailscale solves |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` | Rebuilt/reimaged box; `ssh-keygen -R winbox` (or the IP) and reconnect |
