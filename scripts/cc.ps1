Write-Host -NoNewLine "Removing __pycache__/ recursively..."
Get-ChildItem -Include __pycache__ -Recurse -force | Remove-Item -Force -Recurse
Write-Host "OK"
