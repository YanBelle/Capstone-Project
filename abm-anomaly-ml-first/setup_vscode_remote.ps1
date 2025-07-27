# PowerShell script to set up VS Code Remote Development Environment
# Run this from your Windows machine to configure DigitalOcean server

param(
    [string]$ServerIP = "64.227.16.180",
    [string]$Username = "yc",
    [string]$Password = "yc"
)

Write-Host "🚀 Setting up VS Code Remote Development Environment" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green

# Function to run commands on remote server
function Invoke-RemoteCommand {
    param([string]$Command, [string]$Description)
    
    Write-Host "🔧 $Description..." -ForegroundColor Cyan
    
    # Use plink (PuTTY) if available, otherwise suggest alternative
    if (Get-Command plink -ErrorAction SilentlyContinue) {
        echo "y" | plink -ssh -pw $Password $Username@$ServerIP $Command
    } else {
        Write-Host "💡 Please run this command on your DigitalOcean server:" -ForegroundColor Yellow
        Write-Host "   $Command" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter when completed"
    }
}

function Copy-FileToServer {
    param([string]$LocalFile, [string]$RemotePath, [string]$Description)
    
    Write-Host "📁 $Description..." -ForegroundColor Cyan
    
    if (Get-Command pscp -ErrorAction SilentlyContinue) {
        pscp -pw $Password $LocalFile $Username@${ServerIP}:$RemotePath
    } else {
        Write-Host "💡 Please copy this file to your server:" -ForegroundColor Yellow
        Write-Host "   Local: $LocalFile" -ForegroundColor White
        Write-Host "   Remote: $RemotePath" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter when completed"
    }
}

try {
    Write-Host "📋 Server Information:" -ForegroundColor Yellow
    Write-Host "   IP: $ServerIP" -ForegroundColor White
    Write-Host "   Username: $Username" -ForegroundColor White
    Write-Host "   Password: $Password" -ForegroundColor White
    Write-Host ""

    # Check if setup script exists locally
    $setupScript = "setup_digitalocean_dev.sh"
    if (Test-Path $setupScript) {
        Write-Host "✅ Setup script found locally" -ForegroundColor Green
        
        # Copy setup script to server
        Copy-FileToServer $setupScript "/tmp/setup_digitalocean_dev.sh" "Copying setup script to server"
        
        # Make script executable and run it
        Invoke-RemoteCommand "chmod +x /tmp/setup_digitalocean_dev.sh" "Making setup script executable"
        Invoke-RemoteCommand "sudo /tmp/setup_digitalocean_dev.sh" "Running setup script (this may take several minutes)"
        
    } else {
        Write-Host "❌ Setup script not found locally" -ForegroundColor Red
        Write-Host "💡 Please ensure 'setup_digitalocean_dev.sh' is in the current directory" -ForegroundColor Yellow
        exit 1
    }

    Write-Host ""
    Write-Host "🎉 Setup Complete!" -ForegroundColor Green
    Write-Host "==================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📱 VS Code Remote SSH Configuration:" -ForegroundColor Yellow
    Write-Host "   Host: $ServerIP" -ForegroundColor White
    Write-Host "   User: $Username" -ForegroundColor White
    Write-Host "   Password: $Password" -ForegroundColor White
    Write-Host "   Folder: /home/yc/development/Capstone-Project/abm-anomaly-ml-first" -ForegroundColor White
    Write-Host ""
    Write-Host "🌐 Development Environment URLs:" -ForegroundColor Yellow
    Write-Host "   Dashboard: http://$ServerIP:3001" -ForegroundColor White
    Write-Host "   API: http://$ServerIP:8001" -ForegroundColor White
    Write-Host "   API Docs: http://$ServerIP:8001/docs" -ForegroundColor White
    Write-Host ""
    Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Install VS Code 'Remote - SSH' extension" -ForegroundColor White
    Write-Host "   2. Connect to: $Username@$ServerIP" -ForegroundColor White
    Write-Host "   3. Open folder: /home/yc/development/Capstone-Project/abm-anomaly-ml-first" -ForegroundColor White
    Write-Host "   4. Run in terminal: dev-start" -ForegroundColor White
    Write-Host "   5. Start developing!" -ForegroundColor White
    Write-Host ""
    Write-Host "📖 For detailed instructions, see: VSCODE_REMOTE_INSTRUCTIONS.md" -ForegroundColor Yellow

} catch {
    Write-Host "❌ Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Manual Setup Instructions:" -ForegroundColor Yellow
    Write-Host "   1. SSH to your server: ssh $Username@$ServerIP" -ForegroundColor White
    Write-Host "   2. Copy and run the setup_digitalocean_dev.sh script" -ForegroundColor White
    Write-Host "   3. Follow the VS Code Remote SSH instructions" -ForegroundColor White
}

Write-Host ""
Write-Host "🔧 Development Workflow:" -ForegroundColor Cyan
Write-Host "   Edit Code → Test (Dev: :3001) → Git Push → Auto Deploy (Prod: :3000)" -ForegroundColor White
