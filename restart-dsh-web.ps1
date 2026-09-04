$ErrorActionPreference = 'Continue'
$result = @()
$result += ('old pid 17132 exists: ' + [bool](Get-Process -Id 17132 -ErrorAction SilentlyContinue))
Stop-Process -Id 17132 -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
$result += ('after kill, port 3080 listening: ' + [bool](Get-NetTCPConnection -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue))
$bin = 'C:\Users\AhrixMarin\AppData\Roaming\npm\node_modules\@deepseek-ai\dsh\lib\bin.js'
$outLog = "$env:USERPROFILE\.dsh\web.stdout.log"
$errLog = "$env:USERPROFILE\.dsh\web.stderr.log"
$p = Start-Process -FilePath 'C:\Program Files\nodejs\node.exe' -ArgumentList @($bin, 'web') -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$result += ('started new dsh web pid: ' + $p.Id)
$up = $false
for ($i = 0; $i -lt 90; $i++) {
  Start-Sleep -Seconds 1
  if (Get-NetTCPConnection -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue) { $up = $true; break }
}
$result += ('port 3080 up: ' + $up + ' after ' + ($i + 1) + 's')
Start-Sleep -Seconds 3
try { $rb = Invoke-WebRequest -Uri 'http://127.0.0.1:3080/dsh-whale/balance.json' -UseBasicParsing -TimeoutSec 10; $result += ('balance.json status: ' + $rb.StatusCode + ' hasTotalBalance: ' + $rb.Content.Contains('totalBalance')) } catch { $result += ('balance.json ERR: ' + $_.Exception.Message) }
try { $rw = Invoke-WebRequest -Uri 'http://127.0.0.1:3080/dsh-whale/widget.js' -UseBasicParsing -TimeoutSec 10; $result += ('widget.js status: ' + $rw.StatusCode + ' len: ' + $rw.Content.Length) } catch { $result += ('widget.js ERR: ' + $_.Exception.Message) }
$result | Out-File "$env:USERPROFILE\.dsh\web-restart-result.txt" -Encoding utf8
$result -join "`n"