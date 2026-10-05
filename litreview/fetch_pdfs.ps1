$ErrorActionPreference = "Continue"
$dir = "C:\Users\AhrixMarin\Desktop\otus\litreview\pdfs"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$ids = @(
  "2101.08944",
  "2501.18674",
  "1912.00477",
  "2310.17897",
  "1911.09107",
  "1804.01947",
  "1904.05877",
  "2001.06449"
)

foreach ($id in $ids) {
  $out = Join-Path $dir "$id.pdf"
  if (Test-Path $out) { Write-Host "have $id"; continue }
  $url = "https://arxiv.org/pdf/$id"
  try {
    Invoke-WebRequest -Uri $url -OutFile $out -UserAgent "Mozilla/5.0 (research lit review)" -TimeoutSec 90
    $len = (Get-Item $out).Length
    Write-Host "ok $id  $len bytes"
  } catch {
    Write-Host "FAIL $id : $($_.Exception.Message)"
  }
}
