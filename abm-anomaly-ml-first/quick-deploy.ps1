# Quick SSH Deployment to DigitalOcean
# This PowerShell script syncs your local changes to DigitalOcean and restarts services

param(
    [string]$ServerIP = "64.227.16.180",
    [string]$Username = "root",
    [switch]$OnlyAPI,
    [switch]$OnlyDashboard,
    [switch]$OnlyML,
    [switch]$RestartAll,
    [switch]$ViewLogs,
    [string]$KeyPath = "~\.ssh\id_rsa"
)

Write-Host "🚀 Quick Deploy to DigitalOcean Server" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green

# Configuration
$RemoteProjectPath = "/root/Capstone-Project/abm-anomaly-ml-first"
$LocalProjectPath = Get-Location

# Function to run SSH commands
function Invoke-SSHCommand {
    param([string]$Command)
    Write-Host "🔧 Executing: $Command" -ForegroundColor Yellow
    ssh -i $KeyPath $Username@$ServerIP $Command
}

# Function to sync files using SCP
function Sync-Files {
    param([string]$LocalPath, [string]$RemotePath, [string]$Description)
    Write-Host "📁 Syncing $Description..." -ForegroundColor Cyan
    scp -i $KeyPath -r $LocalPath $Username@$ServerIP`:$RemotePath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $Description synced successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to sync $Description" -ForegroundColor Red
    }
}

try {
    # Check if SSH key exists
    if (!(Test-Path $KeyPath)) {
        Write-Host "❌ SSH key not found at $KeyPath" -ForegroundColor Red
        Write-Host "💡 Generate SSH key with: ssh-keygen -t rsa -b 4096 -C 'your_email@domain.com'" -ForegroundColor Yellow
        Write-Host "💡 Copy to server with: ssh-copy-id -i $KeyPath $Username@$ServerIP" -ForegroundColor Yellow
        exit 1
    }

    # Test SSH connection
    Write-Host "🔍 Testing SSH connection..." -ForegroundColor Cyan
    $connectionTest = ssh -i $KeyPath -o ConnectTimeout=5 $Username@$ServerIP "echo 'Connection successful'"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Cannot connect to server. Check SSH key and server status." -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ SSH connection successful" -ForegroundColor Green

    # Sync specific components or all
    if ($OnlyAPI) {
        Write-Host "🔧 Deploying API only..." -ForegroundColor Magenta
        Sync-Files "services\api\*" "$RemoteProjectPath/services/api/" "API files"
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose restart api"
    }
    elseif ($OnlyDashboard) {
        Write-Host "🎨 Deploying Dashboard only..." -ForegroundColor Magenta
        Sync-Files "services\dashboard\*" "$RemoteProjectPath/services/dashboard/" "Dashboard files"
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose restart dashboard"
    }
    elseif ($OnlyML) {
        Write-Host "🧠 Deploying ML components only..." -ForegroundColor Magenta
        Sync-Files "services\anomaly-detector\*" "$RemoteProjectPath/services/anomaly-detector/" "ML Analyzer files"
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose restart anomaly-detector"
    }
    else {
        Write-Host "📦 Full deployment - syncing all files..." -ForegroundColor Magenta
        
        # Sync all major components
        Sync-Files "services\api\*" "$RemoteProjectPath/services/api/" "API files"
        Sync-Files "services\dashboard\*" "$RemoteProjectPath/services/dashboard/" "Dashboard files"
        Sync-Files "services\anomaly-detector\*" "$RemoteProjectPath/services/anomaly-detector/" "ML Analyzer files"
        Sync-Files "docker-compose.yml" "$RemoteProjectPath/" "Docker Compose"
        
        # Sync SVM debug components
        Write-Host "🔍 Syncing SVM Debug components..." -ForegroundColor Cyan
        Sync-Files "svm_requirements.txt" "$RemoteProjectPath/" "SVM Requirements"
        Sync-Files "debug_svm_cli.py" "$RemoteProjectPath/" "SVM CLI Tool"
        Sync-Files "example_sessions.json" "$RemoteProjectPath/" "Example Sessions"
        
        # Install dependencies on server
        Write-Host "📦 Installing Python dependencies on server..." -ForegroundColor Cyan
        Invoke-SSHCommand "cd $RemoteProjectPath && pip install -r svm_requirements.txt"
    }

    # Restart services
    if ($RestartAll -or (!$OnlyAPI -and !$OnlyDashboard -and !$OnlyML)) {
        Write-Host "🔄 Restarting all services..." -ForegroundColor Cyan
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose down"
        Start-Sleep -Seconds 3
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose up -d"
        Start-Sleep -Seconds 10
        
        # Check service status
        Write-Host "🔍 Checking service status..." -ForegroundColor Cyan
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose ps"
    }

    # Show logs if requested
    if ($ViewLogs) {
        Write-Host "📋 Recent logs..." -ForegroundColor Cyan
        Invoke-SSHCommand "cd $RemoteProjectPath && docker-compose logs --tail=50"
    }

    # Test endpoints
    Write-Host "🌐 Testing endpoints..." -ForegroundColor Cyan
    Write-Host "   Dashboard: http://$ServerIP:3000" -ForegroundColor White
    Write-Host "   API Health: http://$ServerIP:8000/health" -ForegroundColor White
    Write-Host "   SVM Debug: http://$ServerIP:8000/api/v1/svm-debug/model-info" -ForegroundColor White
    Write-Host "   API Docs: http://$ServerIP:8000/docs" -ForegroundColor White

    # Quick health check
    Write-Host "🏥 Quick health check..." -ForegroundColor Cyan
    $healthCheck = Invoke-RestMethod -Uri "http://$ServerIP:8000/health" -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($healthCheck) {
        Write-Host "✅ API is responding" -ForegroundColor Green
    } else {
        Write-Host "⚠️  API not responding yet (may still be starting)" -ForegroundColor Yellow
    }

    Write-Host "🎉 Deployment complete!" -ForegroundColor Green
    Write-Host "💡 Quick commands:" -ForegroundColor Yellow
    Write-Host "   View logs: .\quick-deploy.ps1 -ViewLogs" -ForegroundColor White
    Write-Host "   API only: .\quick-deploy.ps1 -OnlyAPI" -ForegroundColor White
    Write-Host "   Dashboard only: .\quick-deploy.ps1 -OnlyDashboard" -ForegroundColor White
    Write-Host "   Restart all: .\quick-deploy.ps1 -RestartAll" -ForegroundColor White

} catch {
    Write-Host "❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
