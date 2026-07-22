# Netflix Recommender - Environment Activation Script
# Usage: . .\activate.ps1  (note the leading dot to run in current scope)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate the virtual environment
& "$projectRoot\venv\Scripts\Activate.ps1"

# Load .env file and set AWS credentials as environment variables
$envFile = "$projectRoot\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        # Skip blank lines and comments
        if ($_ -match '^\s*$' -or $_ -match '^\s*#') { return }
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
            Write-Host "  Set $key" -ForegroundColor DarkGray
        }
    }
    Write-Host "AWS credentials loaded from .env" -ForegroundColor Green
} else {
    Write-Warning ".env file not found at $envFile"
}

Write-Host ""
Write-Host "Netflix Recommender venv activated (Python $(python --version))" -ForegroundColor Cyan
Write-Host "Project: $projectRoot" -ForegroundColor Cyan
