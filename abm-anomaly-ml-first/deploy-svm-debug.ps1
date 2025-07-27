# Simple Deploy Script for SVM Debug System
# Quick deployment to DigitalOcean without Docker dependency

param(
    [string]$ServerIP = "64.227.16.180"
)

Write-Host "🚀 Deploying SVM Debug System to DigitalOcean" -ForegroundColor Green

# Files to deploy
$filesToDeploy = @{
    "services\api\svm_debug_api.py" = "/root/Capstone-Project/abm-anomaly-ml-first/services/api/svm_debug_api.py"
    "services\api\main.py" = "/root/Capstone-Project/abm-anomaly-ml-first/services/api/main.py"
    "services\dashboard\src\SVMDebugDashboard.js" = "/root/Capstone-Project/abm-anomaly-ml-first/services/dashboard/src/SVMDebugDashboard.js"
    "services\dashboard\src\Dashboard.js" = "/root/Capstone-Project/abm-anomaly-ml-first/services/dashboard/src/Dashboard.js"
    "services\anomaly-detector\svm_visualizer.py" = "/root/Capstone-Project/abm-anomaly-ml-first/services/anomaly-detector/svm_visualizer.py"
    "svm_requirements.txt" = "/root/Capstone-Project/abm-anomaly-ml-first/svm_requirements.txt"
    "debug_svm_cli.py" = "/root/Capstone-Project/abm-anomaly-ml-first/debug_svm_cli.py"
    "example_sessions.json" = "/root/Capstone-Project/abm-anomaly-ml-first/example_sessions.json"
}

try {
    # Deploy each file
    foreach ($localFile in $filesToDeploy.Keys) {
        $remoteFile = $filesToDeploy[$localFile]
        if (Test-Path $localFile) {
            Write-Host "📁 Deploying $localFile..." -ForegroundColor Cyan
            scp $localFile root@${ServerIP}:$remoteFile
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ $localFile deployed" -ForegroundColor Green
            } else {
                Write-Host "❌ Failed to deploy $localFile" -ForegroundColor Red
            }
        } else {
            Write-Host "⚠️  $localFile not found locally" -ForegroundColor Yellow
        }
    }

    # Install dependencies and restart services on server
    Write-Host "🔧 Installing dependencies and restarting services..." -ForegroundColor Cyan
    
    $commands = @(
        "cd /root/Capstone-Project/abm-anomaly-ml-first",
        "pip install -r svm_requirements.txt",
        "docker-compose restart api",
        "docker-compose restart dashboard",
        "docker-compose restart anomaly-detector",
        "sleep 10",
        "docker-compose ps"
    )
    
    foreach ($cmd in $commands) {
        ssh root@$ServerIP $cmd
    }

    Write-Host "🎉 SVM Debug System deployed successfully!" -ForegroundColor Green
    Write-Host "🌐 Test at: http://$ServerIP:3000 (SVM Debug tab)" -ForegroundColor White

} catch {
    Write-Host "❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
}
