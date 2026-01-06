# Windows PowerShell script - WSL olmadan Git push
# GitHub'a SSH ile push etmek için

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GitHub'a Push İşlemi (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Proje dizinine git
$projectPath = "\\wsl$\Ubuntu\home\goksu\code\bosch-kaggle-pipeline"
if (-not (Test-Path $projectPath)) {
    Write-Host "Hata: Proje dizini bulunamadı: $projectPath" -ForegroundColor Red
    Write-Host "WSL bağlantısını kontrol edin veya WSL'i başlatın: wsl" -ForegroundColor Yellow
    exit 1
}

Set-Location $projectPath

Write-Host "1. Git durumu kontrol ediliyor..." -ForegroundColor Yellow
git status

Write-Host ""
Write-Host "2. Data klasörü git'ten kaldırılıyor..." -ForegroundColor Yellow
git rm -r --cached data/ 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Data klasörü zaten track edilmiyor." -ForegroundColor Gray
}

Write-Host ""
Write-Host "3. Tüm değişiklikler stage'e ekleniyor..." -ForegroundColor Yellow
git add -A

Write-Host ""
Write-Host "4. Stage'deki dosyalar:" -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "5. Commit ediliyor..." -ForegroundColor Yellow
git commit -m "Cleanup: Bosch pipeline dosyalarını temizle ve düzenle (data klasörü hariç)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Commit edilecek değişiklik yok." -ForegroundColor Gray
}

Write-Host ""
Write-Host "6. Main branch'e geçiliyor..." -ForegroundColor Yellow
git checkout main 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Main branch oluşturuluyor..." -ForegroundColor Gray
    git checkout -b main
}

Write-Host ""
Write-Host "7. Main branch'e push ediliyor..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Tamamlandı!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Push işlemi başarısız oldu." -ForegroundColor Red
    Write-Host "SSH key ayarlarını kontrol edin." -ForegroundColor Yellow
}

