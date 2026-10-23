<#
.SYNOPSIS
Create one RunPod GPU pod from the eroll GPU image and print its id and SSH details.

.DESCRIPTION
Calls the RunPod REST API (POST /v1/pods): RTX 4090, secure cloud,
only 22/tcp exposed. Every other service on the pod binds to loopback; the local stack
is reached only through deploy/tunnel.ps1.

Worker mode runs the GPU worker; when it exits, the pod deletes itself.

Eval mode runs no worker. The pod deletes itself after -LifetimeMinutes, so a forgotten
benchmark pod stops billing.

Keys are read from the environment and are never printed:
  RUNPOD_API_KEY   both modes (API call, and passed to the pod for self-deletion)
  PUBLIC_KEY       both modes (SSH public key written to authorized_keys)
  GHCR_OWNER       both modes (image is ghcr.io/<owner>/eroll-gpu:<tag>)
  DATABASE_URL, REDIS_URL   worker mode only

RUNPOD_POD_ID is not sent: RunPod sets it inside every pod.

This call starts GPU billing.

.EXAMPLE
deploy/launch_pod.ps1 -Tag v3.0.0 -Mode eval -LifetimeMinutes 90
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$')]
    [string]$Tag,

    [ValidateSet('worker', 'eval')]
    [string]$Mode = 'worker',

    [ValidateRange(1, 1440)]
    [int]$LifetimeMinutes = 120
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$API = 'https://rest.runpod.io/v1'
$GPU_TYPE = 'NVIDIA GeForce RTX 4090'
# Image is ~10 GB plus ~17 GB of Qwen weights downloaded at start.
$CONTAINER_DISK_GB = 60
# The SSH address appears only after the image pull finishes and the container starts.
$SSH_WAIT_SECONDS = 2400
$SSH_POLL_SECONDS = 10

function Get-RequiredEnv([string]$Name) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "missing required environment variable $Name"
    }
    return $value
}

$runpodKey = Get-RequiredEnv 'RUNPOD_API_KEY'
$publicKey = Get-RequiredEnv 'PUBLIC_KEY'
$owner = (Get-RequiredEnv 'GHCR_OWNER').ToLowerInvariant()

$podEnv = [ordered]@{
    RUNPOD_API_KEY        = $runpodKey
    PUBLIC_KEY            = $publicKey
    POD_MODE              = $Mode
    EVAL_LIFETIME_MINUTES = "$LifetimeMinutes"
}

# Optional: turns on the low_confidence flag in the GPU worker
if ($env:CONFIDENCE_THRESHOLD) { $podEnv['CONFIDENCE_THRESHOLD'] = $env:CONFIDENCE_THRESHOLD }

if ($Mode -eq 'worker') {
    # The pod reaches Postgres and Redis at its own 127.0.0.1 via the tunnel's reverse forwards.
    foreach ($name in 'DATABASE_URL', 'REDIS_URL') {
        $value = Get-RequiredEnv $name
        if ($value -notmatch '@127\.0\.0\.1:') {
            throw "$name must point at 127.0.0.1 (the pod reaches it through the tunnel)"
        }
        $podEnv[$name] = $value
    }
}

$image = "ghcr.io/$owner/eroll-gpu:$Tag"
$body = [ordered]@{
    name              = "eroll-$Mode-$Tag"
    imageName         = $image
    computeType       = 'GPU'
    gpuTypeIds        = @($GPU_TYPE)
    gpuCount          = 1
    cloudType         = 'SECURE'
    interruptible     = $false
    ports             = @('22/tcp')
    containerDiskInGb = $CONTAINER_DISK_GB
    volumeInGb        = 0
    supportPublicIp   = $true
    env               = $podEnv
} | ConvertTo-Json -Depth 5

$headers = @{ Authorization = "Bearer $runpodKey" }

try {
    $pod = Invoke-RestMethod -Method Post -Uri "$API/pods" -Headers $headers `
        -ContentType 'application/json' -Body $body
} catch {
    $status = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { 'none' }
    $detail = if ($_.ErrorDetails) { "$($_.ErrorDetails.Message)" } else { $_.Exception.Message }
    foreach ($value in $podEnv.Values) {
        if ($value.Length -ge 4) { $detail = $detail.Replace($value, '<redacted>') }
    }
    $detail = $detail.Replace($runpodKey, '<redacted>')
    throw "RunPod create pod failed: HTTP $status $detail"
}

$podId = $pod.id
Write-Output "pod id: $podId"
Write-Output "image:  $image ($Mode mode)"

# publicIp and the 22 -> external port mapping are filled in only once the pod runs.
$deadline = (Get-Date).AddSeconds($SSH_WAIT_SECONDS)
while ((Get-Date) -lt $deadline) {
    try {
        $pod = Invoke-RestMethod -Method Get -Uri "$API/pods/$podId" -Headers $headers
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        throw "RunPod get pod $podId failed: HTTP $status"
    }
    $ipProp = $pod.PSObject.Properties['publicIp']
    $mapProp = $pod.PSObject.Properties['portMappings']
    $publicIp = if ($ipProp) { $ipProp.Value } else { $null }
    $sshPort = $null
    if ($mapProp -and $mapProp.Value) {
        $portProp = $mapProp.Value.PSObject.Properties['22']
        if ($portProp) { $sshPort = $portProp.Value }
    }
    if ($publicIp -and $sshPort) {
        Write-Output "ssh:    ssh -p $sshPort root@$publicIp"
        Write-Output "tunnel: deploy/tunnel.ps1 -PodIp $publicIp -Port $sshPort"
        exit 0
    }
    Start-Sleep -Seconds $SSH_POLL_SECONDS
}

throw "pod $podId has no SSH address after $SSH_WAIT_SECONDS s; check the RunPod console (the pod is still billed)"
