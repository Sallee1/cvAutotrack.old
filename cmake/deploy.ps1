# deploy.ps1
# 在 POST_BUILD 时复制 DLL/PDB 到游戏目录
# 独立脚本文件保证 UTF-8，避免 cmd 传参中文编码问题
param(
    [Parameter(Mandatory)]
    [string]$DllPath,
    [Parameter(Mandatory)]
    [string]$PdbPath,
    [Parameter(Mandatory)]
    [string]$GameDir,
    [Parameter(Mandatory)]
    [string]$Config
)

$dllDst = Join-Path $GameDir "DLL" $Config
$pdbDst = Join-Path $GameDir "PDB" $Config

mkdir $dllDst -Force -ErrorAction SilentlyContinue | Out-Null
mkdir $pdbDst -Force -ErrorAction SilentlyContinue | Out-Null

Copy-Item $DllPath -Destination $dllDst -Force -Verbose
if (Test-Path $PdbPath) {
    Copy-Item $PdbPath -Destination $pdbDst -Force -Verbose
} else {
    Write-Warning "PDB 文件不存在，跳过: $PdbPath"
}
