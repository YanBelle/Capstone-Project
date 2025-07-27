# Remote Testing Script for DigitalOcean Server
# Test SVM Debug System without local Docker

param(
    [string]$ServerIP = "64.227.16.180",
    [switch]$FullTest,
    [switch]$QuickTest,
    [switch]$ViewLogs,
    [switch]$CheckStatus
)

Write-Host "🧪 Testing SVM Debug System on DigitalOcean" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

function Test-Endpoint {
    param([string]$Url, [string]$Description, [int]$TimeoutSeconds = 10)
    
    Write-Host "🔍 Testing $Description..." -ForegroundColor Cyan
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        Write-Host "✅ $Description: OK" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ $Description: FAILED ($($_.Exception.Message))" -ForegroundColor Red
        return $false
    }
}

function Test-SVMEndpoints {
    Write-Host "🔍 Testing SVM Debug API endpoints..." -ForegroundColor Magenta
    
    $svmEndpoints = @{
        "Model Info" = "http://$ServerIP:8000/api/v1/svm-debug/model-info"
        "Performance Metrics" = "http://$ServerIP:8000/api/v1/svm-debug/performance-metrics"
    }
    
    $results = @{}
    foreach ($endpoint in $svmEndpoints.Keys) {
        $results[$endpoint] = Test-Endpoint $svmEndpoints[$endpoint] $endpoint
    }
    
    return $results
}

function Test-SVMSession {
    Write-Host "🧠 Testing SVM session analysis..." -ForegroundColor Magenta
    
    $testData = @{
        session_id = "test_svm_debug_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        raw_text = "Normal ATM transaction: user inserted card, entered PIN, checked balance, withdrew $100, transaction completed successfully"
        include_visualization = $true
    }
    
    try {
        $response = Invoke-RestMethod -Uri "http://$ServerIP:8000/api/v1/svm-debug/analyze-session" -Method POST -Body ($testData | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
        Write-Host "✅ SVM session analysis: SUCCESS" -ForegroundColor Green
        Write-Host "   Decision Score: $($response.decision_score)" -ForegroundColor White
        Write-Host "   Anomaly Detected: $($response.is_anomaly)" -ForegroundColor White
        return $true
    } catch {
        Write-Host "❌ SVM session analysis: FAILED ($($_.Exception.Message))" -ForegroundColor Red
        return $false
    }
}

try {
    if ($CheckStatus) {
        Write-Host "📊 Checking service status..." -ForegroundColor Cyan
        ssh root@$ServerIP "cd /root/Capstone-Project/abm-anomaly-ml-first && docker-compose ps"
        return
    }

    if ($ViewLogs) {
        Write-Host "📋 Viewing recent logs..." -ForegroundColor Cyan
        ssh root@$ServerIP "cd /root/Capstone-Project/abm-anomaly-ml-first && docker-compose logs --tail=100"
        return
    }

    # Basic connectivity tests
    Write-Host "🌐 Testing basic connectivity..." -ForegroundColor Cyan
    $basicTests = @{
        "API Health" = "http://$ServerIP:8000/health"
        "Dashboard" = "http://$ServerIP:3000"
        "API Docs" = "http://$ServerIP:8000/docs"
    }
    
    $basicResults = @{}
    foreach ($test in $basicTests.Keys) {
        $basicResults[$test] = Test-Endpoint $basicTests[$test] $test
    }

    # SVM-specific tests
    if ($QuickTest -or $FullTest) {
        $svmResults = Test-SVMEndpoints
        
        if ($FullTest) {
            $sessionResult = Test-SVMSession
        }
    }

    # Summary
    Write-Host "`n📊 TEST SUMMARY" -ForegroundColor Yellow
    Write-Host "===============" -ForegroundColor Yellow
    
    Write-Host "`n🔧 Basic Services:" -ForegroundColor Cyan
    foreach ($test in $basicResults.Keys) {
        $status = if ($basicResults[$test]) { "✅ PASS" } else { "❌ FAIL" }
        Write-Host "   $test`: $status" -ForegroundColor White
    }
    
    if ($QuickTest -or $FullTest) {
        Write-Host "`n🔍 SVM Debug API:" -ForegroundColor Cyan
        foreach ($test in $svmResults.Keys) {
            $status = if ($svmResults[$test]) { "✅ PASS" } else { "❌ FAIL" }
            Write-Host "   $test`: $status" -ForegroundColor White
        }
        
        if ($FullTest -and $sessionResult) {
            Write-Host "`n🧠 SVM Session Analysis: ✅ PASS" -ForegroundColor Green
        } elseif ($FullTest) {
            Write-Host "`n🧠 SVM Session Analysis: ❌ FAIL" -ForegroundColor Red
        }
    }

    # Access URLs
    Write-Host "`n🌐 ACCESS URLS:" -ForegroundColor Yellow
    Write-Host "   Dashboard: http://$ServerIP:3000" -ForegroundColor White
    Write-Host "   SVM Debug Tab: http://$ServerIP:3000 (click SVM Debug)" -ForegroundColor White
    Write-Host "   API Docs: http://$ServerIP:8000/docs" -ForegroundColor White
    Write-Host "   SVM Debug API: http://$ServerIP:8000/api/v1/svm-debug/" -ForegroundColor White

    # Quick commands
    Write-Host "`n💡 QUICK COMMANDS:" -ForegroundColor Yellow
    Write-Host "   Full test: .\test-remote.ps1 -FullTest" -ForegroundColor White
    Write-Host "   Quick test: .\test-remote.ps1 -QuickTest" -ForegroundColor White
    Write-Host "   View logs: .\test-remote.ps1 -ViewLogs" -ForegroundColor White
    Write-Host "   Check status: .\test-remote.ps1 -CheckStatus" -ForegroundColor White

} catch {
    Write-Host "❌ Testing failed: $($_.Exception.Message)" -ForegroundColor Red
}
