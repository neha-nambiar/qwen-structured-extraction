<#
.SYNOPSIS
Open the SSH tunnel between this machine and the GPU pod; it stays open until Ctrl+C.

.DESCRIPTION
Reverse forwards (pod loopback to this machine), so the GPU worker reaches the local
stack at its own 127.0.0.1:
  pod 127.0.0.1:6379  -> host 127.0.0.1:6379   Redis
  pod 127.0.0.1:5432  -> host 127.0.0.1:5432   Postgres (jobs, page images, records)

Local forwards (this machine to pod loopback), so Prometheus scrapes the pod through
host.docker.internal:
  host 127.0.0.1:18000 -> pod 127.0.0.1:8000   vLLM /metrics
  host 127.0.0.1:19102 -> pod 127.0.0.1:9102   GPU worker /metrics

Every forward binds 127.0.0.1 on both ends; nothing is published on a public interface.
Closing the tunnel leaves the pod idle, and it deletes itself after 15 minutes.

.PARAMETER PodIp
The pod's public IP, as printed by deploy/launch_pod.ps1.

.PARAMETER Port
The pod's external port mapped to 22, as printed by deploy/launch_pod.ps1.

.PARAMETER IdentityFile
Optional path to the SSH private key matching PUBLIC_KEY.

.EXAMPLE
deploy/tunnel.ps1 -PodIp 203.0.113.10 -Port 40022 -IdentityFile ~/.ssh/esci_runpod
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9.:-]+$')]
    [string]$PodIp,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$Port,

    [string]$IdentityFile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Every pod is new, so its host key is unknown.
$identity = if ($IdentityFile) { @('-i', $IdentityFile) } else { @() }
ssh @identity -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -p $Port root@$PodIp -R 127.0.0.1:6379:127.0.0.1:6379 -R 127.0.0.1:5432:127.0.0.1:5432 -L 127.0.0.1:18000:127.0.0.1:8000 -L 127.0.0.1:19102:127.0.0.1:9102
exit $LASTEXITCODE
