# 批量生成所有图片的二进制嵌入头文件
# 在单个 PowerShell 进程中处理全部图片，避免进程启动开销
# PowerShell 进程启动 ~700ms，15 次分开启动浪费 ~10s，合并后只需 ~1s

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResourceDir = Resolve-Path "$ScriptDir\..\..\..\resource"
$BinImgDir  = "$ScriptDir\image"

$images = @(
    @{ src = "$ResourceDir\GI_ICON_SIGHT.png";  type = "Png"; name = "icon_sight"; dst = "$BinImgDir\icon\resources.binary.image.icon_sight.h" }
    @{ src = "$ResourceDir\GI_ICON_QUEST.png";   type = "Png"; name = "icon_quest"; dst = "$BinImgDir\icon\resources.binary.image.icon_quest.h" }
    @{ src = "$ResourceDir\uid\uid_.png";         type = "Png"; name = "uid_";      dst = "$BinImgDir\uid\resources.binary.image.uid_.h" }
    @{ src = "$ResourceDir\uid\uid0.png";         type = "Png"; name = "uid0";     dst = "$BinImgDir\uid\resources.binary.image.uid0.h" }
    @{ src = "$ResourceDir\uid\uid1.png";         type = "Png"; name = "uid1";     dst = "$BinImgDir\uid\resources.binary.image.uid1.h" }
    @{ src = "$ResourceDir\uid\uid2.png";         type = "Png"; name = "uid2";     dst = "$BinImgDir\uid\resources.binary.image.uid2.h" }
    @{ src = "$ResourceDir\uid\uid3.png";         type = "Png"; name = "uid3";     dst = "$BinImgDir\uid\resources.binary.image.uid3.h" }
    @{ src = "$ResourceDir\uid\uid4.png";         type = "Png"; name = "uid4";     dst = "$BinImgDir\uid\resources.binary.image.uid4.h" }
    @{ src = "$ResourceDir\uid\uid5.png";         type = "Png"; name = "uid5";     dst = "$BinImgDir\uid\resources.binary.image.uid5.h" }
    @{ src = "$ResourceDir\uid\uid6.png";         type = "Png"; name = "uid6";     dst = "$BinImgDir\uid\resources.binary.image.uid6.h" }
    @{ src = "$ResourceDir\uid\uid7.png";         type = "Png"; name = "uid7";     dst = "$BinImgDir\uid\resources.binary.image.uid7.h" }
    @{ src = "$ResourceDir\uid\uid8.png";         type = "Png"; name = "uid8";     dst = "$BinImgDir\uid\resources.binary.image.uid8.h" }
    @{ src = "$ResourceDir\uid\uid9.png";         type = "Png"; name = "uid9";     dst = "$BinImgDir\uid\resources.binary.image.uid9.h" }
)

function Convert-ImageToHeader {
    param([string]$src, [string]$type, [string]$name, [string]$dst)

    $data = [System.IO.File]::ReadAllBytes($src)

    $sb = [System.Text.StringBuilder]::new()
    $null = $sb.AppendLine("#pragma once")
    $null = $sb.AppendLine("")
    $null = $sb.AppendLine("namespace TianLi::Resources::Binary::Image::$type")
    $null = $sb.AppendLine("{")
    $null = $sb.Append("    const unsigned char $name[] = {")

    for ($i = 0; $i -lt $data.Length; $i++) {
        if ($i % 16 -eq 0) {
            $null = $sb.AppendLine()
            $null = $sb.Append("        ")
        }
        $null = $sb.AppendFormat("0x{0:X2},", $data[$i])
    }

    $null = $sb.AppendLine()
    $null = $sb.AppendLine("    };")
    $null = $sb.AppendLine("}")

    # 确保目标目录存在
    $dstDir = Split-Path $dst -Parent
    if (-not (Test-Path $dstDir)) {
        $null = New-Item -ItemType Directory -Path $dstDir -Force
    }

    [System.IO.File]::WriteAllText($dst, $sb.ToString())
}

foreach ($img in $images) {
    Convert-ImageToHeader -src $img.src -type $img.type -name $img.name -dst $img.dst
}
